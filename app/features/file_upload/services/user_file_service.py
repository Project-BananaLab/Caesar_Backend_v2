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
    - company_id, user_id, is_private 정보 포함
    """
    return handle_upload_and_ingest(
        db,
        company_id=company_id,
        employee_id=user_id,
        is_private=True,  # 개인 파일은 항상 private
        file_bytes=file_bytes,
        file_name=file_name,
    )


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
