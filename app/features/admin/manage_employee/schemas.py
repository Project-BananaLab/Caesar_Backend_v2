# schemas.py
# 직원 관리 API의 요청/응답 스키마를 정의합니다.

from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime


class EmployeeBase(BaseModel):
    """직원 기본 정보 스키마"""
    full_name: Optional[str] = None
    email: EmailStr
    job_dept_id: Optional[int] = None
    job_rank_id: Optional[int] = None


class EmployeeResponse(BaseModel):
    """직원 정보 응답 스키마"""
    id: int
    full_name: Optional[str] = None
    email: Optional[str] = None  # None 값 허용
    job_dept_id: Optional[int] = None
    job_rank_id: Optional[int] = None
    dept_name: Optional[str] = None  # 부서명
    rank_name: Optional[str] = None  # 직급명
    company_id: int
    google_user_id: Optional[str] = None  # None 값 허용
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EmployeeUpdateRequest(BaseModel):
    """직원 정보 수정 요청 스키마"""
    job_dept_id: Optional[int] = None
    job_rank_id: Optional[int] = None


class EmployeeListResponse(BaseModel):
    """직원 목록 응답 스키마"""
    employees: List[EmployeeResponse]
    total_count: int


class DeptResponse(BaseModel):
    """부서 정보 응답 스키마"""
    id: int
    dept_name: str

    class Config:
        from_attributes = True


class RankResponse(BaseModel):
    """직급 정보 응답 스키마"""
    id: int
    rank_name: str

    class Config:
        from_attributes = True
