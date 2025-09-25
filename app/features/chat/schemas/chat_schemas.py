# app/features/chat/schemas/chat_schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Message(BaseModel):
    """메시지 스키마"""
    role: str = Field(..., description="메시지 역할 (user, agent, assistant)")
    content: str = Field(..., min_length=1, description="메시지 내용")


class ChatBase(BaseModel):
    """채팅 기본 스키마"""
    channel_id: int = Field(..., description="채널 ID")
    messages: List[Message] = Field(default_factory=list, description="메시지 배열")


class ChatCreate(ChatBase):
    """채팅 생성 스키마"""
    pass


class ChatResponse(ChatBase):
    """채팅 응답 스키마"""
    id: int = Field(..., description="채팅 ID")
    created_at: datetime = Field(..., description="생성 시간")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatListResponse(BaseModel):
    """채팅 목록 응답 스키마"""
    chats: List[ChatResponse]
    total: int = Field(..., description="총 채팅 개수")
    channel_id: Optional[int] = Field(None, description="필터링된 채널 ID")