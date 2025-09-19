# app/features/settings/test.py
"""
설정 스키마 모듈
"""
from typing import Optional
from pydantic import BaseModel


class JobDept(BaseModel):
    """부서 정보 스키마"""
    id: int
    dept_name: str  # varchar(255)


class JobRank(BaseModel):
    """직급 정보 스키마"""
    id: int
    rank_name: str  # varchar(255)


class Employee(BaseModel):
    """직원 정보 스키마"""
    id: int
    company_id: int
    job_dept_id: int
    job_rank_id: int
    user_name: str  # varchar(50)
    google_uid: str  # varchar(100)
    google_calendar_json: bytes  # blob
    google_drive_json: bytes  # blob
    notion_API: bytes  # blob
    slack_API: bytes  # blob