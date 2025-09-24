# app/utils/db.py
from app.utils.env_loader import env_tokens

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.core.config import settings

# SQLAlchemy 엔진 (pre-ping으로 끊어진 연결 자동 감지)
engine = create_engine(settings.DB_URL, pool_pre_ping=True)

# 세션 팩토리                                               # 선택: 커밋 후 객체 즉시 재사용할 때 편함
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


# Base 클래스 (모든 모델이 상속)
class Base(DeclarativeBase):
    # 기본 스키마를 caesar로 설정
    metadata = MetaData(schema="caesar")


# FastAPI 의존성: 요청당 세션 제공
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
