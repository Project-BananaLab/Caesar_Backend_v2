# auth.py

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from . import crud, schemas, models, security
from .database import get_db

# "/employee/login" 경로에서 토큰을 가져오도록 설정
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/employee/login")

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    """
    JWT 토큰을 검증하고 해당 사용자를 DB에서 찾아 반환합니다.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증 정보를 확인할 수 없습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 토큰을 디코딩하여 payload를 얻음
        payload = jwt.decode(token, security.SECRET_KEY, algorithms=[security.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # DB에서 해당 이메일의 사용자를 찾음
    user = crud.get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: models.Employee = Depends(get_current_user)):
    """
    현재 사용자가 활성 상태인지 확인합니다. (확장성을 위해 분리)
    예: is_active 플래그가 있는 경우 여기서 확인 가능
    """
    # if not current_user.is_active:
    #     raise HTTPException(status_code=400, detail="비활성화된 계정입니다.")
    return current_user