# app/features/settings/test.py
"""
설정 스키마 모듈
"""
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel


class JobDept(BaseModel):
    """부서 정보 스키마"""
    id: int
    dept_name: str  # varchar(255)


class JobRank(BaseModel):
    """직급 정보 스키마"""
    id: int
    rank_name: str  # varchar(255)


class EmployeeCreate(BaseModel):
    """직원 생성/수정용 스키마 (입력)"""
    company_id: int
    job_dept_id: int
    job_rank_id: int
    user_name: str  # varchar(50)
    google_uid: str  # varchar(100)
    google_calendar_json: Optional[Union[Dict[str, Any], str]] = None  # JSON 객체 또는 문자열
    google_drive_json: Optional[Union[Dict[str, Any], str]] = None  # JSON 객체 또는 문자열
    notion_api: Optional[str] = None  # API 키 문자열
    slack_api: Optional[str] = None  # API 키 문자열


class Employee(BaseModel):
    """직원 정보 스키마 (출력)"""
    id: int
    company_id: int
    job_dept_id: int
    job_rank_id: int
    user_name: str  # varchar(50)
    google_uid: str  # varchar(100)
    google_calendar_json: Optional[Union[Dict[str, Any], str]] = None  # 복호화된 JSON 객체
    google_drive_json: Optional[Union[Dict[str, Any], str]] = None  # 복호화된 JSON 객체
    notion_api: Optional[str] = None  # 복호화된 API 키
    slack_api: Optional[str] = None  # 복호화된 API 키