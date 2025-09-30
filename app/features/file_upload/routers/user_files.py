# app/features/file_upload/routers/user_files.py
# -*- coding: utf-8 -*-
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Header
from sqlalchemy.orm import Session

from app.utils.db import get_db
from app.features.login.employee_google import crud as employee_crud
from app.features.file_upload.services.user_file_service import (
    upload_user_file,
    get_user_files,
    delete_user_file,
    get_user_file_count,
)
from app.features.file_upload.schemas.file_schemas import (
    FileUploadResponse,
    UserFileInfo,
    UserFileListResponse,
    FileDeleteResponse,
)

router = APIRouter(prefix="/api/user/files", tags=["user-files"])


def get_current_user_info(authorization: str = Header(None), db: Session = Depends(get_db)):
    """
    현재 로그인된 사용자 정보 추출
    - Authorization 헤더에서 Google User ID를 추출하여 사용자 정보 반환
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required",
        )

    # Authorization 헤더에서 Google User ID 추출
    try:
        auth_scheme, google_user_id = authorization.split(" ", 1)
        if auth_scheme not in ["Bearer", "GoogleAuth"]:
            raise ValueError("Invalid auth scheme")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'GoogleAuth {google_user_id}'",
        )

    # 사용자 정보 조회
    db_employee = employee_crud.get_employee_by_google_id(db, google_user_id=google_user_id)
    if not db_employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Employee not found"
        )

    return db_employee


@router.post("/upload", response_model=FileUploadResponse)
async def upload_personal_file(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_info),
    file: UploadFile = File(...),
):
    """
    개인 파일 업로드
    - 사용자 개인 문서로 업로드 (is_private=True)
    - company_id, user_id, is_private 정보 자동 설정
    """
    # 파일 내용 읽기
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail="빈 파일은 업로드할 수 없습니다."
        )

    # 파일 업로드 처리
    result = upload_user_file(
        db,
        company_id=current_user.company_id,
        user_id=current_user.id,
        file_bytes=content,
        file_name=file.filename,
    )

    if not result.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "파일 업로드에 실패했습니다.")
        )

    return FileUploadResponse(**result)


@router.get("/list", response_model=UserFileListResponse)
def list_personal_files(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_info),
    limit: int = 50,
    offset: int = 0,
):
    """
    개인 파일 목록 조회
    - 현재 사용자가 업로드한 개인 파일만 조회
    """
    # 파일 목록 조회
    files = get_user_files(
        db,
        company_id=current_user.company_id,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
    )

    # 총 개수 조회
    total = get_user_file_count(
        db,
        company_id=current_user.company_id,
        user_id=current_user.id,
    )

    # 응답 형식으로 변환
    file_list = [
        UserFileInfo(
            id=f.id,
            fileName=f.file_name,
            url=f.file_url,
            isPrivate=f.is_private,
            status=f.ingest_status,
            size=f.file_size,
            chunksCount=f.chunks_count,
            errorText=f.error_text,
            createdAt=f.created_at,
            updatedAt=f.updated_at,
            ingestedAt=f.ingested_at,
        )
        for f in files
    ]

    return UserFileListResponse(files=file_list, total=total)


@router.delete("/{file_id}", response_model=FileDeleteResponse)
def delete_personal_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_info),
):
    """
    개인 파일 삭제
    - 본인이 업로드한 파일만 삭제 가능
    - DB/S3/VectorDB에서 모두 삭제
    """
    result = delete_user_file(
        db,
        doc_id=file_id,
        company_id=current_user.company_id,
        user_id=current_user.id,
    )

    if not result.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.get("error", "파일을 찾을 수 없거나 삭제 권한이 없습니다.")
        )

    return FileDeleteResponse(**result)
