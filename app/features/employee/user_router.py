# user_router.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from . import crud, schemas, security, models
from .database import get_db
from .auth import get_current_active_user # 인증을 위한 dependency

router = APIRouter(
    prefix="/employee", # 이 라우터의 모든 경로는 /employee 로 시작
    tags=["employee"],   # FastAPI 문서에서 그룹화하기 위한 태그
)

@router.post("/signup", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """회원가입 API"""
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="이미 등록된 이메일입니다.")
    return crud.create_user(db=db, user=user)


@router.post("/login", response_model=schemas.Token)
def login(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    """로그인 API (JWT 토큰 발급)"""
    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = security.create_access_token(
        data={"sub": user.email}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.put("/me/integrations", response_model=schemas.UserResponse)
def update_integrations(
    integrations: schemas.IntegrationsUpdate,
    db: Session = Depends(get_db),
    current_user: models.Employee = Depends(get_current_active_user)
):
    """
    로그인한 사용자의 외부 서비스 연동 정보(API 키 등)를 업데이트합니다.
    데이터는 암호화되어 DB에 저장됩니다.
    """
    return crud.update_user_integrations(db=db, user_id=current_user.id, data=integrations)


@router.get("/me/integrations", response_model=schemas.IntegrationsResponse)
def get_my_integrations(current_user: models.Employee = Depends(get_current_active_user)):
    """
    (백엔드 사용 예시) 로그인한 사용자의 연동 정보를 DB에서 가져와 복호화하여 반환합니다.
    실제 서비스에서는 이 데이터를 클라이언트에 직접 보내기보다,
    이 데이터를 필요로 하는 다른 백엔드 로직에서 사용합니다.
    """
    decrypted_data = {
        "google_calendar_json": security.decrypt_data(current_user.google_calendar_json) if current_user.google_calendar_json else None,
        "google_drive_json": security.decrypt_data(current_user.google_drive_json) if current_user.google_drive_json else None,
        "notion_api": security.decrypt_data(current_user.notion_api) if current_user.notion_api else None,
        "slack_api": security.decrypt_data(current_user.slack_api) if current_user.slack_api else None,
    }
    return decrypted_data