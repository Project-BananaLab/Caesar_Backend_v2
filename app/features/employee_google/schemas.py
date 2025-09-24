# schemas.py
# Pydantic 모델(스키마)을 정의하여 API의 요청 및 응답 데이터 형식을 지정하고 유효성을 검사합니다.

from pydantic import BaseModel, EmailStr
from typing import Optional

# 직원 정보의 기본 필드를 정의하는 스키마
class EmployeeBase(BaseModel):
    email: EmailStr  # 이메일 형식 유효성 검사
    full_name: str
    google_user_id: str

# 새로운 직원을 생성할 때 요청 본문으로 받을 데이터 스키마
# EmployeeBase의 모든 필드를 상속받습니다.
class EmployeeCreate(EmployeeBase):
    pass

# API 키를 업데이트할 때 요청 본문으로 받을 데이터 스키마
class APIKeysUpdate(BaseModel):
    notion_api: Optional[str] = None
    slack_api: Optional[str] = None

# API 응답으로 클라이언트에게 반환될 직원 정보 스키마
class Employee(EmployeeBase):
    id: int
    company_id: Optional[int] = None
    job_dept_id: Optional[int] = None
    job_rank_id: Optional[int] = None
    # 실제 부서명과 직급명을 포함
    dept_name: Optional[str] = None
    rank_name: Optional[str] = None
    # API 키 존재 여부 (보안을 위해 실제 값은 반환하지 않음)
    has_notion_api: bool = False
    has_slack_api: bool = False

    # 이 설정은 SQLAlchemy 모델 객체를 Pydantic 스키마로 변환할 수 있게 해줍니다.
    class Config:
        from_attributes = True # 이전 버전의 orm_mode = True 와 동일
