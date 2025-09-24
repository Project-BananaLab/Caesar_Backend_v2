# app/features/channel/schemas/channel_schemas.py
"""
채널 관련 데이터베이스 스키마 모듈
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ChannelBase(BaseModel):
    """Channel 기본 스키마"""

    name: str = Field(..., min_length=1, max_length=100, description="채널 이름")
    description: Optional[str] = Field(None, max_length=500, description="채널 설명")
    channel_type: str = Field(
        default="public", description="채널 타입 (public, private, direct)"
    )
    is_active: bool = Field(default=True, description="활성화 상태")
    max_members: Optional[int] = Field(
        None, gt=0, description="최대 멤버 수 (None은 무제한)"
    )


class ChannelCreate(ChannelBase):
    """Channel 생성 요청 스키마"""

    created_by: Optional[int] = Field(None, description="생성자 사용자 ID")


class ChannelUpdate(BaseModel):
    """Channel 수정 요청 스키마"""

    name: Optional[str] = Field(
        None, min_length=1, max_length=100, description="채널 이름"
    )
    description: Optional[str] = Field(None, max_length=500, description="채널 설명")
    channel_type: Optional[str] = Field(None, description="채널 타입")
    is_active: Optional[bool] = Field(None, description="활성화 상태")
    max_members: Optional[int] = Field(None, gt=0, description="최대 멤버 수")


class ChannelResponse(ChannelBase):
    """Channel 응답 스키마 (DB 조회 결과)"""

    id: int = Field(..., description="채널 ID (PK)")
    created_by: Optional[int] = Field(None, description="생성자 사용자 ID")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: Optional[datetime] = Field(None, description="수정 시간")

    # 추가 통계 정보
    member_count: Optional[int] = Field(None, description="현재 멤버 수")
    message_count: Optional[int] = Field(None, description="총 메시지 수")
    last_message_at: Optional[datetime] = Field(None, description="마지막 메시지 시간")

    # 관계 데이터
    creator_name: Optional[str] = Field(None, description="생성자 이름")

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class ChannelListResponse(BaseModel):
    """Channel 목록 응답 스키마"""

    channels: List[ChannelResponse]
    total: int = Field(..., description="총 채널 개수")
    page: int = Field(..., description="현재 페이지")
    page_size: int = Field(..., description="페이지 크기")
    total_pages: int = Field(..., description="총 페이지 수")


class ChannelDeleteResponse(BaseModel):
    """Channel 삭제 응답 스키마"""

    message: str = Field(..., description="삭제 결과 메시지")
    deleted_channel_id: int = Field(..., description="삭제된 채널 ID")
    deleted_channel_name: str = Field(..., description="삭제된 채널 이름")


class ChannelSearchRequest(BaseModel):
    """Channel 검색 요청 스키마"""

    name_keyword: Optional[str] = Field(None, description="채널 이름 키워드 검색")
    channel_type: Optional[str] = Field(None, description="채널 타입 필터")
    is_active: Optional[bool] = Field(None, description="활성화 상태 필터")
    created_by: Optional[int] = Field(None, description="생성자 ID 필터")
    start_date: Optional[datetime] = Field(None, description="생성일 시작 날짜")
    end_date: Optional[datetime] = Field(None, description="생성일 종료 날짜")
    limit: int = Field(default=20, ge=1, le=100, description="조회 개수")
    offset: int = Field(default=0, ge=0, description="건너뛸 개수")


class ChannelMember(BaseModel):
    """Channel 멤버 스키마"""

    id: int = Field(..., description="멤버 ID (PK)")
    channel_id: int = Field(..., description="채널 ID")
    user_id: int = Field(..., description="사용자 ID")
    role: str = Field(default="member", description="역할 (owner, admin, member)")
    joined_at: datetime = Field(..., description="가입 시간")
    is_muted: bool = Field(default=False, description="음소거 상태")

    # 관계 데이터
    user_name: Optional[str] = Field(None, description="사용자 이름")

    class Config:
        from_attributes = True


class ChannelMemberRequest(BaseModel):
    """Channel 멤버 추가/수정 요청 스키마"""

    user_id: int = Field(..., description="사용자 ID")
    role: str = Field(default="member", description="역할")


# 데이터베이스 테이블 스키마 (SQLAlchemy 참고용)
"""
-- 채널 테이블
CREATE TABLE channels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    channel_type VARCHAR(20) DEFAULT 'public' CHECK (channel_type IN ('public', 'private', 'direct')),
    is_active BOOLEAN DEFAULT TRUE,
    max_members INTEGER,
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL,
    
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
    
    INDEX idx_name (name),
    INDEX idx_channel_type (channel_type),
    INDEX idx_is_active (is_active),
    INDEX idx_created_by (created_by)
);

-- 채널 멤버 테이블
CREATE TABLE channel_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role VARCHAR(20) DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member')),
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_muted BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (channel_id) REFERENCES channels(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    
    UNIQUE(channel_id, user_id),
    INDEX idx_channel_id (channel_id),
    INDEX idx_user_id (user_id),
    INDEX idx_role (role)
);

-- 사용자 테이블 (참고용)
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(100),
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL
);
"""
