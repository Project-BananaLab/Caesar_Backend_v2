# app/features/admin/services/file_ingest_service.py
# -*- coding: utf-8 -*-
import os
import tempfile
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import datetime
import traceback

from app.features.admin.models.docs import Doc
from app.features.login.company.models import Company
from app.features.admin.services.s3_service import (
    put_file_and_checksum,
    delete_object_by_url,
)
from app.rag.internal_data_rag.internal_ingest import IngestService


def _get_company_code(db: Session, company_id: int) -> str:
    """
    company.id -> company.code(=회사코드) 로 변환.
    - 회사코드는 Chroma 컬렉션명으로 사용된다.
    - 없으면 company_{id} fallback.
    """
    stmt = select(Company.code).where(Company.id == company_id).limit(1)
    code = db.execute(stmt).scalars().first()
    return (code or "").strip() or f"company_{company_id}"


def handle_upload_and_ingest(
    db: Session,
    *,
    company_id: int,
    employee_id: Optional[int],
    is_private: bool,
    file_bytes: bytes,
    file_name: str,
) -> dict:
    """
    업로드 전체 파이프라인:
      1) S3 업로드
      2) docs INSERT ('processing')
      3) IngestService 로 Chroma 인덱싱 (회사코드 컬렉션)
      4) docs UPDATE (succeeded/failed)
      5) (선택) 실패 시 S3 롤백 시도
    반환: {"ok": bool, "docId": int, "chunks"?: int, "url"?: str, "error"?: str}
    """
    # ── 1) S3 업로드 (인자명 file_bytes 사용!)
    s3_url, size, checksum = put_file_and_checksum(
        file_bytes=file_bytes,
        orig_name=file_name,
    )

    # ── 2) docs INSERT (processing)
    doc = Doc(
        company_id=company_id,
        employee_id=employee_id,          # 관리자 업로드면 None
        is_private=is_private,            # False=회사공개, True=개인문서
        file_name=file_name,
        file_url=s3_url,
        file_size=size,
        checksum_sha256=checksum,
        ingest_status="processing",       # 초기 상태
        chunks_count=0,
        error_text=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        ingested_at=None,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)  # doc.id 확보(→ VectorDB 메타 doc_id 로 사용)

    # ── 3) ingest (회사코드 컬렉션 + extra 메타 병합)
    try:
        company_code = _get_company_code(db, company_id)

        # 🔴 None이 들어가면 Chroma가 메타데이터 에러를 냅니다.
        #     → None은 아예 넣지 말고, 기본 타입만 사용
        extra_meta = {
            "doc_id": int(doc.id),
            "company_id": int(company_id),
            "is_private": bool(is_private),
        }
        if employee_id is not None:
            extra_meta["user_id"] = int(employee_id)

        with tempfile.TemporaryDirectory() as td:
            local_path = os.path.join(td, file_name)
            with open(local_path, "wb") as f:
                f.write(file_bytes)

            svc = IngestService()
            chunks_count, ok = svc.ingest_single_file_with_metadata(
                local_path,
                collection_name=company_code,  # ← 회사코드 컬렉션
                extra_meta=extra_meta,
                show_preview=False
            )

        if not ok:
            # 실패 → 상태 저장, 에러 메시지
            doc.ingest_status = "failed"
            doc.error_text = "embedding_or_chroma_error"
            doc.updated_at = datetime.utcnow()
            db.add(doc)
            db.commit()

            # (선택) 롤백: S3 지우기
            try:
                delete_object_by_url(s3_url)
            except Exception:
                pass

            return {"ok": False, "docId": doc.id, "error": "ingest_failed"}

        # ── 4) docs UPDATE (성공)
        doc.ingest_status = "succeeded"
        doc.chunks_count = chunks_count
        doc.ingested_at = datetime.utcnow()
        doc.updated_at = datetime.utcnow()
        db.add(doc)
        db.commit()

        return {"ok": True, "docId": doc.id, "chunks": chunks_count, "url": s3_url}

    except Exception as e:
        # 실패 시 상태/에러 메시지 남김
        doc.ingest_status = "failed"
        doc.error_text = f"{type(e).__name__}: {e}"
        doc.updated_at = datetime.utcnow()
        db.add(doc)
        db.commit()

        # (선택) 롤백: S3 지우기
        try:
            delete_object_by_url(s3_url)
        except Exception:
            pass

        traceback.print_exc()
        return {"ok": False, "docId": doc.id, "error": str(e)}


def delete_doc_everywhere(db: Session, *, doc_id: int) -> dict:
    """
    문서를 DB/S3/VectorDB에서 동시 정리.
    - VectorDB는 해당 회사코드 컬렉션에서 where={"doc_id": doc_id} 로 삭제.
    """
    doc = db.get(Doc, doc_id)
    if not doc:
        return {"ok": False, "error": "not_found"}

    # S3 객체 삭제
    try:
        delete_object_by_url(doc.file_url)
    except Exception:
        pass

    # VectorDB 삭제
    try:
        company_code = _get_company_code(db, doc.company_id)
        svc = IngestService()
        col = svc.get_chroma_collection(company_code)
        col.delete(where={"doc_id": int(doc_id)})
    except Exception:
        pass

    # DB 삭제
    db.delete(doc)
    db.commit()
    return {"ok": True, "deleted": doc_id}
