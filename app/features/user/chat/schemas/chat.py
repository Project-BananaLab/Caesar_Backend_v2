# app/features/user/chat/test.py
"""
사용자 채팅 모듈 테스트 파일
"""

from datetime import datetime
from pydantic import BaseModel


class Chat(BaseModel):
    """채팅 정보 스키마"""
    id: int
    channel_id: int
    chat_history: str
    created_at: datetime