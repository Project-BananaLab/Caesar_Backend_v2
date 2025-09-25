# app/features/channel/schemas/channel_schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChannelBase(BaseModel):
    """채널 기본 스키마"""

    title: Optional[str] = Field(None, max_length=255, description="대화목록 제목")


class ChannelCreate(ChannelBase):
    """채널 생성 스키마"""

    employee_id: int = Field(..., description="채널 주인 employee ID")


class ChannelUpdate(BaseModel):
    """채널 수정 스키마"""

    title: Optional[str] = Field(None, max_length=255, description="수정할 제목")


class ChannelResponse(ChannelBase):
    """채널 응답 스키마"""

    id: int = Field(..., description="채널 ID")
    employee_id: int = Field(..., description="채널 주인 employee ID")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="수정 시간")

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class ChannelListResponse(BaseModel):
    """채널 목록 응답 스키마"""

    channels: List[ChannelResponse]
    total: int = Field(..., description="총 채널 개수")


class ChannelDeleteResponse(BaseModel):
    """채널 삭제 응답 스키마"""

    message: str
    deleted_channel_id: int
