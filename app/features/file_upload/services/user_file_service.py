# app/features/file_upload/services/user_file_service.py
# -*- coding: utf-8 -*-
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import datetime

from app.features.admin.models.docs import Doc
from app.features.admin.services.file_ingest_service import (
    handle_upload_and_ingest,
    delete_doc_everywhere,
)


def upload_user_file(
    db: Session,
    *,
    company_id: int,
    user_id: int,
    file_bytes: bytes,
    file_name: str,
) -> dict:
    """
    개인 파일 업로드 처리
    - 항상 is_private=True로 설정
    - 사용자별 중복 체크 (동일 사용자의 동일 파일만 중복 방지)
    - company_id, user_id, is_private 정보 포함
    """
    from app.features.admin.services.s3_service import sha256_bytes, put_file_if_absent
    from app.features.admin.models.docs import Doc
    from sqlalchemy import select
    from datetime import datetime
    import tempfile
    import os
    import time
    import gc
    
    # 해시 계산
    checksum = sha256_bytes(file_bytes)
    
    # 개인 파일 중복 체크: 같은 사용자 + 같은 해시 조합만 중복으로 판단
    existing = db.execute(
        select(Doc).where(
            Doc.company_id == company_id,
            Doc.employee_id == user_id,  # 같은 사용자
            Doc.checksum_sha256 == checksum,  # 같은 내용
            Doc.is_private == True  # 개인 문서
        )
    ).scalars().first()
    
    if existing:
        # 같은 사용자가 이미 동일한 파일을 업로드한 경우
        return {
            "ok": True,
            "duplicated": True,
            "docId": existing.id,
            "chunks": existing.chunks_count,
            "url": existing.file_url,
        }
    
    # S3 업로드 (내용주소화 방식)
    s3_url, size, checksum_hex, uploaded_new = put_file_if_absent(
        file_bytes=file_bytes,
        orig_name=file_name,
        checksum_hex=checksum,
    )
    
    # DB에 문서 메타데이터 저장
    doc = Doc(
        company_id=company_id,
        employee_id=user_id,  # 개인 파일이므로 업로더 기록
        is_private=True,  # 개인 파일은 항상 private
        file_name=file_name,
        file_url=s3_url,
        file_size=size,
        checksum_sha256=checksum_hex,
        ingest_status="processing",
    )
    
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    # Chroma 인덱싱
    try:
        from app.features.admin.services.file_ingest_service import _get_company_code
        from app.rag.internal_data_rag.internal_ingest import IngestService
        
        company_code = _get_company_code(db, company_id)
        
        # 메타데이터 설정
        extra_meta = {
            "doc_id": int(doc.id),
            "company_id": int(company_id),
            "is_private": True,
            "user_id": int(user_id),
        }
        
        # 임시 파일로 저장 후 인덱싱
        td = tempfile.mkdtemp()
        try:
            local_path = os.path.join(td, file_name)
            with open(local_path, "wb") as f:
                f.write(file_bytes)
            
            svc = IngestService()
            chunks_count, ok = svc.ingest_single_file_with_metadata(
                local_path,
                collection_name=company_code,
                extra_meta=extra_meta,
                show_preview=False
            )
            
            time.sleep(0.1)
            gc.collect()
            
        finally:
            import shutil
            shutil.rmtree(td, ignore_errors=True)
        
        if ok:
            # 성공
            doc.ingest_status = "succeeded"
            doc.chunks_count = chunks_count
            doc.ingested_at = datetime.utcnow()
        else:
            # 실패
            doc.ingest_status = "failed"
            doc.error_text = "embedding_or_chroma_error"
        
        doc.updated_at = datetime.utcnow()
        db.add(doc)
        db.commit()
        
        return {
            "ok": ok,
            "duplicated": False,
            "docId": doc.id,
            "chunks": chunks_count if ok else 0,
            "url": s3_url,
            "uploadedNewToS3": uploaded_new,
        }
        
    except Exception as e:
        # 인덱싱 실패
        doc.ingest_status = "failed"
        doc.error_text = str(e)
        doc.updated_at = datetime.utcnow()
        db.add(doc)
        db.commit()
        
        return {
            "ok": False,
            "docId": doc.id,
            "error": str(e)
        }


def get_user_files(
    db: Session,
    *,
    company_id: int,
    user_id: int,
    limit: int = 50,
    offset: int = 0,
) -> List[Doc]:
    """
    사용자의 개인 파일 목록 조회
    - 해당 사용자가 업로드한 private 파일만 조회
    """
    query = (
        db.query(Doc)
        .filter(
            Doc.company_id == company_id,
            Doc.employee_id == user_id,
            Doc.is_private == True
        )
        .order_by(Doc.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    
    return query.all()


def delete_user_file(
    db: Session,
    *,
    doc_id: int,
    company_id: int,
    user_id: int,
) -> dict:
    """
    사용자 개인 파일 삭제
    - 본인이 업로드한 private 파일만 삭제 가능
    """
    # 파일 소유권 확인
    doc = (
        db.query(Doc)
        .filter(
            Doc.id == doc_id,
            Doc.company_id == company_id,
            Doc.employee_id == user_id,
            Doc.is_private == True
        )
        .first()
    )
    
    if not doc:
        return {"ok": False, "error": "파일을 찾을 수 없거나 삭제 권한이 없습니다."}
    
    # 파일 삭제 (DB/S3/VectorDB에서 모두 삭제)
    return delete_doc_everywhere(db, doc_id=doc_id)


def get_user_file_count(
    db: Session,
    *,
    company_id: int,
    user_id: int,
) -> int:
    """사용자의 개인 파일 총 개수 조회"""
    return (
        db.query(Doc)
        .filter(
            Doc.company_id == company_id,
            Doc.employee_id == user_id,
            Doc.is_private == True
        )
        .count()
    )
