# app/features/login/company/routers.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.utils.db import get_db
from .schemas import CompanyLoginIn, CompanyLoginOut
from .service import login_by_coid

# 프론트 요청 경로와 통일
router = APIRouter(prefix="/api/company", tags=["auth-company"])

@router.post(
    "/login",
    response_model=CompanyLoginOut,
    response_model_by_alias=True,  # 응답을 alias(camelCase)로 직렬화
)
def company_login(payload: CompanyLoginIn, db: Session = Depends(get_db)):
    try:
        # 요청은 camelCase(coId)로 들어오지만, 코드에선 payload.co_id 로 접근
        return login_by_coid(db, payload.co_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
