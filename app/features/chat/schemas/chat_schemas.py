# app/features/chat/schemas/chat_schemas.py
"""
채팅 관련 데이터베이스 스키마 모듈
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ChatBase(BaseModel):
    """Chat 기본 스키마"""

    channel_id: int = Field(..., description="채널 ID (FK)")
    user_id: Optional[int] = Field(None, description="사용자 ID (FK) - 익명 채팅 지원")
    message: str = Field(
        ..., min_length=1, max_length=2000, description="채팅 메시지 내용"
    )
    message_type: str = Field(
        default="text", description="메시지 타입 (text, image, file, etc.)"
    )
    is_edited: bool = Field(default=False, description="수정 여부")
    is_deleted: bool = Field(default=False, description="삭제 여부 (소프트 삭제)")


class ChatCreate(ChatBase):
    """Chat 생성 요청 스키마"""

    pass


class ChatUpdate(BaseModel):
    """Chat 수정 요청 스키마"""

    message: str = Field(
        ..., min_length=1, max_length=2000, description="수정할 메시지 내용"
    )


class ChatResponse(ChatBase):
    """Chat 응답 스키마 (DB 조회 결과)"""

    id: int = Field(..., description="채팅 ID (PK)")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: Optional[datetime] = Field(None, description="수정 시간")

    # 관계 데이터 (JOIN 결과)
    channel_name: Optional[str] = Field(None, description="채널 이름")
    user_name: Optional[str] = Field(None, description="사용자 이름")

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class ChatListResponse(BaseModel):
    """Chat 목록 응답 스키마"""

    chats: List[ChatResponse]
    total: int = Field(..., description="총 채팅 개수")
    page: int = Field(..., description="현재 페이지")
    page_size: int = Field(..., description="페이지 크기")
    total_pages: int = Field(..., description="총 페이지 수")


class ChatSearchRequest(BaseModel):
    """Chat 검색 요청 스키마"""

    channel_id: Optional[int] = Field(None, description="채널 ID 필터")
    user_id: Optional[int] = Field(None, description="사용자 ID 필터")
    message_keyword: Optional[str] = Field(None, description="메시지 키워드 검색")
    start_date: Optional[datetime] = Field(None, description="검색 시작 날짜")
    end_date: Optional[datetime] = Field(None, description="검색 종료 날짜")
    message_type: Optional[str] = Field(None, description="메시지 타입 필터")
    limit: int = Field(default=20, ge=1, le=100, description="조회 개수")
    offset: int = Field(default=0, ge=0, description="건너뛸 개수")


# 데이터베이스 테이블 스키마 (SQLAlchemy 참고용)
"""
CREATE TABLE chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL,
    user_id INTEGER,
    message TEXT NOT NULL,
    message_type VARCHAR(50) DEFAULT 'text',
    is_edited BOOLEAN DEFAULT FALSE,
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL,
    
    FOREIGN KEY (channel_id) REFERENCES channels(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    
    INDEX idx_channel_id (channel_id),
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_message_search (message(100))
);
"""
