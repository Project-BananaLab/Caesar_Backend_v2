# app/features/channel/models/channel_models.py
"""
Channel 데이터베이스 모델 (SQLAlchemy)
실제 DB 연동 시 사용할 모델 정의
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    Table,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Channel(Base):
    """채널 테이블 모델"""

    __tablename__ = "channels"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Channel Info
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    channel_type = Column(
        String(20), default="public", nullable=False
    )  # public, private, direct

    # Channel Settings
    is_active = Column(Boolean, default=True, nullable=False)
    max_members = Column(Integer, nullable=True)  # None = unlimited

    # Creator Info
    created_by = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.utcnow, nullable=True)

    # Relationships
    chats = relationship("Chat", back_populates="channel", cascade="all, delete-orphan")
    creator = relationship("User", back_populates="created_channels")
    members = relationship(
        "ChannelMember", back_populates="channel", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (
            f"<Channel(id={self.id}, name='{self.name}', type='{self.channel_type}')>"
        )


class ChannelMember(Base):
    """채널 멤버 테이블 모델"""

    __tablename__ = "channel_members"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign Keys
    channel_id = Column(
        Integer, ForeignKey("channels.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # Member Info
    role = Column(String(20), default="member", nullable=False)  # owner, admin, member
    joined_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_muted = Column(Boolean, default=False, nullable=False)

    # Relationships
    channel = relationship("Channel", back_populates="members")
    user = relationship("User", back_populates="channel_memberships")

    def __repr__(self):
        return f"<ChannelMember(channel_id={self.channel_id}, user_id={self.user_id}, role='{self.role}')>"


# 사용자 모델 (참고용)
class User(Base):
    """사용자 테이블 모델 (참고용)"""

    __tablename__ = "users"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # User Info
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(100), nullable=True)
    avatar_url = Column(Text, nullable=True)

    # Status
    is_active = Column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.utcnow, nullable=True)

    # Relationships
    chats = relationship("Chat", back_populates="user")
    created_channels = relationship("Channel", back_populates="creator")
    channel_memberships = relationship("ChannelMember", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


# 사용 예시 (실제 DB 연동 시)
"""
# 데이터베이스 생성 쿼리

-- 사용자 테이블
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
    
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
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
    
    UNIQUE(channel_id, user_id)
);

-- 채팅 테이블
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
CREATE INDEX idx_channels_name ON channels(name);
CREATE INDEX idx_channels_type ON channels(channel_type);
CREATE INDEX idx_channels_created_by ON channels(created_by);

CREATE INDEX idx_channel_members_channel_id ON channel_members(channel_id);
CREATE INDEX idx_channel_members_user_id ON channel_members(user_id);

CREATE INDEX idx_chats_channel_id ON chats(channel_id);
CREATE INDEX idx_chats_user_id ON chats(user_id);
CREATE INDEX idx_chats_created_at ON chats(created_at);
"""
