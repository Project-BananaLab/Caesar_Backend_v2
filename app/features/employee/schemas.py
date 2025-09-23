# schemas.py

from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any

# --- 사용자 관련 스키마 ---
class UserCreate(BaseModel):
    """회원가입 시 사용할 스키마"""
    name: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    """사용자 정보 응답 시 사용할 스키마 (비밀번호 제외)"""
    id: int
    name: str
    email: EmailStr

    class Config:
        from_attributes = True # SQLAlchemy 모델을 Pydantic 모델로 변환 허용

# --- 인증 관련 스키마 ---
class Token(BaseModel):
    """로그인 응답 시 토큰 정보를 담을 스키마"""
    access_token: str
    token_type: str

# --- 외부 서비스 연동 정보 스키마 ---
class IntegrationsUpdate(BaseModel):
    """외부 서비스 API 키/JSON 업데이트 시 사용할 스키마"""
    google_calendar_json: Optional[Dict[str, Any]] = None
    google_drive_json: Optional[Dict[str, Any]] = None
    notion_api: Optional[str] = None
    slack_api: Optional[str] = None

class IntegrationsResponse(IntegrationsUpdate):
    """외부 서비스 정보 응답 시 사용할 스키마 (복호화된 데이터)"""
    pass