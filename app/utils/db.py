# app/utils/db.py
from app.utils.env_loader import env_tokens

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL is not set (check your .env)")

engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 사용자별 OAuth 토큰 저장소
user_tokens = {
    "user_123": {
        "google": env_tokens["google"],
        "slack": env_tokens["slack"],
        "notion": env_tokens["notion"],
    }
}

def get_user_tokens(user_id: str) -> dict:
    """사용자 ID로 토큰 정보 조회"""
    return user_tokens.get(user_id, {})


def save_user_tokens(user_id: str, service: str, tokens: dict):
    """사용자 토큰 저장"""
    if user_id not in user_tokens:
        user_tokens[user_id] = {}
    user_tokens[user_id][service] = tokens


def get_service_token(user_id: str, service: str) -> dict:
    """특정 서비스의 토큰 조회"""
    return user_tokens.get(user_id, {}).get(service, {})
