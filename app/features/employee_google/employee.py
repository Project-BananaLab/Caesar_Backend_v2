# employee.py
# 'employee'와 관련된 API 엔드포인트를 정의하는 라우터입니다.

from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from . import crud, schemas, models
from .database import get_db
from typing import Optional

# APIRouter 인스턴스를 생성합니다.
# main.py에서 이 라우터를 포함시켜 API 엔드포인트를 구성합니다.
router = APIRouter(
    prefix="/employees",  # 이 라우터의 모든 경로 앞에 "/employees"가 붙습니다.
    tags=["employees"],   # FastAPI 문서에서 API를 그룹화하는 태그입니다.
)

@router.post("/google-login", response_model=schemas.Employee)
def google_login(employee: schemas.EmployeeCreate, db: Session = Depends(get_db)):
    """
    구글 소셜 로그인을 처리합니다.
    - 사용자가 DB에 없으면 새로 생성하고, 있으면 기존 정보를 반환합니다.
    - 프론트엔드에서 구글 로그인 후 받은 사용자 정보를 이 엔드포인트로 보냅니다.
    """
    # Google User ID로 사용자가 이미 존재하는지 확인합니다.
    db_employee = crud.get_employee_by_google_id(db, google_user_id=employee.google_user_id)
    if db_employee:
        # 사용자가 이미 존재하면 해당 사용자 정보를 반환합니다.
        return db_employee
    # 사용자가 없으면 새로 생성합니다.
    return crud.create_employee(db=db, employee=employee)


@router.get("/me", response_model=schemas.Employee)
def get_current_employee(
    authorization: Optional[str] = Header(None), 
    db: Session = Depends(get_db)
):
    """
    현재 로그인된 사용자의 정보를 조회합니다.
    - Authorization 헤더에서 Google User ID를 추출하여 사용자 정보를 반환합니다.
    - 보안: URL에 민감한 정보가 노출되지 않습니다.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Authorization header is required"
        )
    
    # Authorization 헤더에서 Google User ID 추출
    # 형식: "Bearer {google_user_id}" 또는 "GoogleAuth {google_user_id}"
    try:
        auth_scheme, google_user_id = authorization.split(' ', 1)
        if auth_scheme not in ['Bearer', 'GoogleAuth']:
            raise ValueError("Invalid auth scheme")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'GoogleAuth {google_user_id}'"
        )
    
    # 해당 사용자가 존재하는지 확인합니다.
    db_employee = crud.get_employee_by_google_id(db, google_user_id=google_user_id)
    if db_employee is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Employee not found"
        )
    
    return db_employee


@router.get("/{google_user_id}", response_model=schemas.Employee)
def get_employee(google_user_id: str, db: Session = Depends(get_db)):
    """
    특정 사용자의 정보를 조회합니다.
    - Google User ID로 사용자 정보를 반환합니다.
    - 주의: 이 엔드포인트는 관리자용이며, 일반 사용자는 /me를 사용해야 합니다.
    """
    # 해당 사용자가 존재하는지 확인합니다.
    db_employee = crud.get_employee_by_google_id(db, google_user_id=google_user_id)
    if db_employee is None:
        # 사용자가 없으면 404 Not Found 에러를 발생시킵니다.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Employee not found")
    
    return db_employee


@router.put("/{google_user_id}/api-keys", response_model=schemas.Employee)
def update_api_keys(google_user_id: str, api_keys: schemas.APIKeysUpdate, db: Session = Depends(get_db)):
    """
    특정 사용자의 Notion 및 Slack API 키를 업데이트합니다.
    - 요청 본문으로 받은 API 키들은 암호화되어 DB에 저장됩니다.
    """
    # 해당 사용자가 존재하는지 먼저 확인합니다.
    db_employee = crud.get_employee_by_google_id(db, google_user_id=google_user_id)
    if db_employee is None:
        # 사용자가 없으면 404 Not Found 에러를 발생시킵니다.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Employee not found")

    # API 키를 업데이트하는 CRUD 함수를 호출합니다.
    updated_employee = crud.update_employee_api_keys(db, google_user_id=google_user_id, api_keys=api_keys)
    return updated_employee
