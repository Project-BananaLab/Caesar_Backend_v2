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
    put_file_if_absent,
    sha256_bytes,
    delete_object_by_url,
)

from sqlalchemy.exc import IntegrityError  # ← 경합 처리용 (선택)

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
      1) 바이트 해시 계산 → 동일(회사+해시) 문서 존재 시 재사용
      2) (필요 시) S3 업로드 (내용주소화)
      3) docs INSERT ('processing')
      4) IngestService 로 Chroma 인덱싱 (회사코드 컬렉션)
      5) docs UPDATE (succeeded/failed)
      6) (선택) 실패 시 S3 롤백 시도
    반환: {"ok": bool, "docId": int, "chunks"?: int, "url"?: str, "error"?: str, "duplicated"?: bool}
    """
    # 0) 해시 선계산
    checksum = sha256_bytes(file_bytes)

    # 1) 동일(회사, 해시) 문서 선조회 → 있으면 기존 레코드 재사용
    existing = db.execute(
        select(Doc).where(Doc.company_id == company_id, Doc.checksum_sha256 == checksum)
    ).scalars().first()

    if existing:
        # 이미 같은 내용이 인덱싱되어 있는 케이스
        return {
            "ok": True,
            "duplicated": True,
            "docId": existing.id,
            "chunks": existing.chunks_count,
            "url": existing.file_url,
            "message": f"동일한 파일이 이미 존재합니다: {existing.file_name}",
            "existingFileName": existing.file_name,
        }

    # ── 2) S3 업로드 (내용주소화 방식으로: 같은 내용은 물리적으로 1회만 업로드)
    s3_url, size, checksum_hex, uploaded_new = put_file_if_absent(
        file_bytes=file_bytes,
        orig_name=file_name,
        checksum_hex=checksum,
    )

    # ── 3) docs INSERT (processing)
    doc = Doc(
        company_id=company_id,
        employee_id=employee_id,          # 관리자 업로드면 None
        is_private=is_private,            # False=회사공개, True=개인문서
        file_name=file_name,
        file_url=s3_url,
        file_size=size,
        checksum_sha256=checksum_hex,
        ingest_status="processing",       # 초기 상태
        chunks_count=0,
        error_text=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        ingested_at=None,
    )
    db.add(doc)

    # UNIQUE 제약(회사+해시)과 경합 시 안전 처리
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # 레이스: 다른 트랜잭션이 방금 생성했다면 그대로 재사용
        winner = db.execute(
            select(Doc).where(Doc.company_id == company_id, Doc.checksum_sha256 == checksum)
        ).scalars().first()
        if winner:
            return {
                "ok": True,
                "duplicated": True,
                "docId": winner.id,
                "chunks": winner.chunks_count,
                "url": winner.file_url,
                "message": f"동일한 파일이 이미 존재합니다: {winner.file_name}",
                "existingFileName": winner.file_name,
            }
        # 예외가 다른 원인이라면 다시 던져도 됨
        raise

    db.refresh(doc)  # doc.id 확보(→ VectorDB 메타 doc_id 로 사용)

    # ── 4) ingest (회사코드 컬렉션 + extra 메타 병합)
    try:
        company_code = _get_company_code(db, company_id)

        # Chroma 메타데이터는 기본 타입만 허용 (None 값 제외)
        extra_meta = {
            "doc_id": int(doc.id),
            "company_id": int(company_id),
            "is_private": bool(is_private),
        }
        if employee_id is not None:
            extra_meta["user_id"] = int(employee_id)

        # 임시 디렉터리에 파일 저장 후 청킹 진행
        import time
        import gc
        
        td = tempfile.mkdtemp()
        try:
            local_path = os.path.join(td, file_name)
            with open(local_path, "wb") as f:
                f.write(file_bytes)

            svc = IngestService()
            # 회사별 컬렉션으로 청킹 및 임베딩 저장
            chunks_count, ok = svc.ingest_single_file_with_metadata(
                local_path,
                collection_name=company_code,  # 회사코드별 컬렉션
                extra_meta=extra_meta,
                show_preview=False
            )
            
            # Excel 파일 처리 후 리소스 정리 대기
            time.sleep(0.1)  # 잠시 대기하여 파일 핸들 해제
            gc.collect()     # 가비지 컬렉션 강제 실행
            
        finally:
            # 임시 디렉터리 안전하게 정리
            try:
                import shutil
                shutil.rmtree(td, ignore_errors=True)
            except Exception:
                pass

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

        # ── 5) docs UPDATE (성공) - 안전한 업데이트
        try:
            # 최신 상태로 새로고침
            db.refresh(doc)
            doc.ingest_status = "succeeded"
            doc.chunks_count = chunks_count
            doc.ingested_at = datetime.utcnow()
            doc.updated_at = datetime.utcnow()
            db.add(doc)
            db.commit()
        except Exception as update_error:
            print(f"⚠️ docs 업데이트 실패: {update_error}")
            # 업데이트 실패해도 ChromaDB 저장은 성공했으므로 계속 진행
            db.rollback()

        return {
            "ok": True,
            "duplicated": False,
            "docId": doc.id,
            "chunks": chunks_count,
            "url": s3_url,
            "uploadedNewToS3": uploaded_new,  # 참고용
        }

    except Exception as e:
        # 실패 시 상태/에러 메시지 남김 - 안전한 업데이트
        try:
            db.refresh(doc)
            doc.ingest_status = "failed"
            doc.error_text = f"{type(e).__name__}: {e}"
            doc.updated_at = datetime.utcnow()
            db.add(doc)
            db.commit()
        except Exception as update_error:
            print(f"⚠️ 실패 상태 업데이트 실패: {update_error}")
            db.rollback()

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
