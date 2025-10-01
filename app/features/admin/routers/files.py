# app/features/admin/routers/files.py
# -*- coding: utf-8 -*-
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session

from app.utils.db import get_db
from app.features.admin.models.docs import Doc
from app.features.admin.services.file_ingest_service import (
    handle_upload_and_ingest,
    delete_doc_everywhere,
)
from app.features.auth.company_auth import get_current_company_admin
# ↑ 회사 토큰 디코딩 의존성: { company_id or companyId, code?, role? } 반환 가정

router = APIRouter(prefix="/api/admin/files", tags=["admin-files"])


@router.post("/upload")
async def upload_files(
    db: Session = Depends(get_db),
    admin = Depends(get_current_company_admin),  # 관리자 토큰(회사계정)
    files: List[UploadFile] = File(...),
    isPrivate: Optional[bool] = Form(False),     # 기본은 회사공개 문서
    employeeId: Optional[int] = Form(None),      # 개인문서 업로드 시 직원 id (관리자는 보통 None)
):
    """
    관리자 업로드 엔드포인트
    - 단일/다중 파일 업로드 모두 지원
    - 회사공개 문서: isPrivate=false, employeeId=None
    - 개인문서로 업로드: isPrivate=true + employeeId 지정(필수)
    """
    # 1) 관리자 권한 체크 (의존성에서 미리 검증되어 왔다고 가정)
    role = admin.get("role") or admin.get("Role") or "admin"
    if role != "admin":
        raise HTTPException(status_code=403, detail="관리자만 업로드 가능")

    # 2) 회사 ID 추출(스네이크/카멜 호환)
    company_id = admin.get("company_id") or admin.get("companyId")
    if not company_id:
        raise HTTPException(status_code=401, detail="토큰에 company_id가 없습니다.")

    # 3) 개인문서 유효성 체크
    if bool(isPrivate) and not employeeId:
        raise HTTPException(
            status_code=400,
            detail="개인문서 업로드는 employeeId가 필요합니다."
        )

    results = []
    for uf in files:
        content = await uf.read()
        if not content:
            results.append({"ok": False, "file": uf.filename, "error": "empty_file"})
            continue

        r = handle_upload_and_ingest(
            db,
            company_id=int(company_id),
            employee_id=employeeId,
            is_private=bool(isPrivate),
            file_bytes=content,
            file_name=uf.filename,
        )
        results.append({"file": uf.filename, **r})

    ok_count = sum(1 for r in results if r.get("ok"))
    if ok_count == 0:
        # 전부 실패한 경우 400으로 상세 반환
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=results)
    return {"uploaded": results}


@router.get("/list")
def list_docs(
    db: Session = Depends(get_db),
    admin = Depends(get_current_company_admin),
    limit: int = 50,
    offset: int = 0,
):
    """
    관리자용 문서 목록
    - 관리자 회사의 공개 문서만 조회 (개인 문서 제외)
    """
    company_id = admin.get("company_id") or admin.get("companyId")
    if not company_id:
        raise HTTPException(status_code=401, detail="토큰에 company_id가 없습니다.")

    q = (
        db.query(Doc)
        .filter(
            Doc.company_id == int(company_id),
            Doc.is_private == False  # 회사 공개 문서만 조회
        )
        .order_by(Doc.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        {
            "id": d.id,
            "fileName": d.file_name,
            "url": d.file_url,
            "isPrivate": d.is_private,
            "employeeId": d.employee_id,
            "status": d.ingest_status,
            "size": d.file_size,
            "createdAt": d.created_at,
        }
        for d in q
    ]


@router.delete("/{doc_id}")
def delete_doc(
    doc_id: int,
    db: Session = Depends(get_db),
    admin = Depends(get_current_company_admin),
):
    """
    관리자 문서 삭제 (S3/VectorDB/DB 동시 삭제)
    - 본인 회사의 공개 문서만 삭제 가능 (개인 문서 삭제 불가)
    """
    company_id = admin.get("company_id") or admin.get("companyId")
    if not company_id:
        raise HTTPException(status_code=401, detail="토큰에 company_id가 없습니다.")

    d = (
        db.query(Doc)
        .filter(
            Doc.id == doc_id, 
            Doc.company_id == int(company_id),
            Doc.is_private == False  # 회사 공개 문서만 삭제 가능
        )
        .first()
    )
    if not d:
        raise HTTPException(status_code=404, detail="not found")

    return delete_doc_everywhere(db, doc_id=doc_id)
