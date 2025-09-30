# app/features/file_upload/schemas/file_schemas.py
# -*- coding: utf-8 -*-
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime


class FileUploadResponse(BaseModel):
    """파일 업로드 응답 스키마"""
    ok: bool
    docId: Optional[int] = None
    chunks: Optional[int] = None
    url: Optional[str] = None
    error: Optional[str] = None
    duplicated: Optional[bool] = False


class UserFileInfo(BaseModel):
    """사용자 파일 정보 스키마"""
    id: int
    fileName: str
    url: str
    isPrivate: bool
    status: str  # 'pending'|'processing'|'succeeded'|'failed'
    size: Optional[int] = None
    chunksCount: int = 0
    errorText: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime
    ingestedAt: Optional[datetime] = None


class UserFileListResponse(BaseModel):
    """사용자 파일 목록 응답 스키마"""
    files: List[UserFileInfo]
    total: int


class FileDeleteResponse(BaseModel):
    """파일 삭제 응답 스키마"""
    ok: bool
    deleted: Optional[int] = None
    error: Optional[str] = None
