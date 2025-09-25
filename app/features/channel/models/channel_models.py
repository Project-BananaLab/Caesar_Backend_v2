# app/features/channel/models/channel_models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.utils.db import Base


class Channel(Base):
    """채널 테이블 - 대화방"""

    __tablename__ = "channels"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(
        Integer, ForeignKey("employee.id"), nullable=False, comment="채널 주인"
    )
    title = Column(String(255), nullable=True, comment="대화목록 제목")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), comment="생성 시간"
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="수정 시간",
    )

    # 관계 설정
    # employee = relationship("Employee", back_populates="channels")
    chats = relationship("Chat", back_populates="channel", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Channel(id={self.id}, title='{self.title}', employee_id={self.employee_id})>"
