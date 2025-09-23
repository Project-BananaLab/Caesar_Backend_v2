# app/features/login/company/models.py
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, LargeBinary
from app.utils.db import Base

class Company(Base):
    __tablename__ = "company"  # 실제 테이블명이 이게 맞다면 그대로 사용

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str | None] = mapped_column("code", String(255))
    # Postgres는 기본 소문자 컬럼이므로 실제 이름을 명시
    co_notion_API: Mapped[bytes | None] = mapped_column("co_notion_api", LargeBinary)
    co_name: Mapped[str | None] = mapped_column("co_name", String(255))
    co_id: Mapped[str | None] = mapped_column("co_id", String(255), unique=True)
