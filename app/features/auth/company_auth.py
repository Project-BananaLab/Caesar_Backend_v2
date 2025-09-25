# app/features/auth/company_auth.py
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

bearer = HTTPBearer(auto_error=True)

ALGO = settings.ALGORITHM

def get_current_company_admin(token: HTTPAuthorizationCredentials = Depends(bearer)):
    """
    관리자(회사계정) 토큰 파싱
    - payload.role == "admin" 인지 확인
    - companyId, co_id를 downstream에서 사용
    """
    try:
        payload = jwt.decode(token.credentials, settings.JWT_SECRET_KEY, algorithms=ALGO)
        if payload.get("typ") != "access":
            raise ValueError("invalid token type")
        role = payload.get("role")
        if role != "admin":
            raise ValueError("admin only")
        return {
            "company_id": payload.get("companyId"),
            "co_id": payload.get("co_id"),
            "role": role,
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e}")
