# app/features/login/company/security.py
import jwt
from datetime import datetime, timedelta, timezone
from app.core.config import settings

ALGO = settings.ALGORITHM

def _now() -> datetime:
    return datetime.now(timezone.utc)

def create_access_token(company_id: int, co_id: str, role: str) -> str:
    iat = _now()
    exp = iat + timedelta(minutes=settings.ACCESS_EXPIRES_MIN)
    payload = {
        "co_id": co_id,               # 회사 계정ID
        "companyId": company_id,    # 회사 primary key
        "role": role,               # 'admin'
        "iat": int(iat.timestamp()),
        "exp": int(exp.timestamp()),
        "typ": "access",
    }
    # ✅ 설정에서 단일화한 secret 사용
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=ALGO)
