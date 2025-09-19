"""
Admin 스키마 모듈
"""
from pydantic import BaseModel


class Company(BaseModel):
    """회사 정보 스키마"""
    id: int
    code: str  # varchar(255)
    co_notion_API: bytes  # BLOB
    co_name: str  # varchar(255) 
    co_id: str  # varchar(255)


class Docs(BaseModel):
    """문서 정보 스키마"""
    id: int
    company_id: int
    employee_id: int
    docs_by: str  # varchar(255)
    file_name: str  # varchar(255)
    file_url: str  # varchar(255)
    doc_type: str  # varchar(50)
