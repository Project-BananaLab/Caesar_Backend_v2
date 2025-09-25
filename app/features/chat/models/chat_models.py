# app/features/chat/models/chat_models.py
from sqlalchemy import Column, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.utils.db import Base


class Chat(Base):
    """채팅 테이블 - 메시지 세션"""

    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(
        Integer, ForeignKey("channels.id"), nullable=False, comment="채널 ID"
    )
    messages = Column(JSON, default=list, comment="메시지 배열 (JSONB)")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), comment="생성 시간"
    )

    # 관계 설정
    channel = relationship("Channel", back_populates="chats")

    def __repr__(self):
        return f"<Chat(id={self.id}, channel_id={self.channel_id}, messages_count={len(self.messages or [])})>"
