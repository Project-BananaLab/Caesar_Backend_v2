# app/features/chat/models/chat_models.py
"""
Chat 데이터베이스 모델 (SQLAlchemy)
실제 DB 연동 시 사용할 모델 정의
"""

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Chat(Base):
    """채팅 테이블 모델"""

    __tablename__ = "chats"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign Keys
    channel_id = Column(
        Integer, ForeignKey("channels.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Chat Data
    message = Column(Text, nullable=False)
    message_type = Column(String(50), default="text", nullable=False)

    # Status Fields
    is_edited = Column(Boolean, default=False, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.utcnow, nullable=True)

    # Relationships
    channel = relationship("Channel", back_populates="chats")
    user = relationship("User", back_populates="chats")

    # Indexes (추가 설정 필요)
    __table_args__ = (
        # 인덱스는 SQLAlchemy migration 또는 직접 SQL로 생성
    )

    def __repr__(self):
        return f"<Chat(id={self.id}, channel_id={self.channel_id}, message='{self.message[:50]}...')>"


# 사용 예시 (실제 DB 연동 시)
"""
# 데이터베이스 생성 쿼리
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
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- 인덱스 생성
CREATE INDEX idx_chats_channel_id ON chats(channel_id);
CREATE INDEX idx_chats_user_id ON chats(user_id);
CREATE INDEX idx_chats_created_at ON chats(created_at);
CREATE INDEX idx_chats_message_search ON chats(message);
"""
