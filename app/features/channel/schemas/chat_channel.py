# app/features/user/channel/chat_channel.py
"""
채팅 채널 스키마 모듈
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ChatChannel(BaseModel):
    """채팅 채널 정보 스키마"""
    channel_id: str  # varchar(100)
    user_id: int
    chat_title: str  # varchar(255)
    created_at: datetime