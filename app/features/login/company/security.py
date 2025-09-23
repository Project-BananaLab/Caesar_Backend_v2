# app/features/login/company/security.py
import os, jwt
from datetime import datetime, timedelta, timezone

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-to-32+bytes-secret")
ACCESS_MIN = int(os.getenv("ACCESS_EXPIRES_MIN", "30"))
ALGO = "HS256"

def _now() -> datetime:
    return datetime.now(timezone.utc)

def create_access_token(company_id: int, co_id: str, role: str) -> str:
    iat = _now()
    exp = iat + timedelta(minutes=ACCESS_MIN)
    payload = {
        "sub": co_id,
        "companyId": company_id,
        "role": role,
        "iat": int(iat.timestamp()),
        "exp": int(exp.timestamp()),
        "typ": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGO)
