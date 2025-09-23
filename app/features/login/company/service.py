# app/features/login/company/service.py
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from .models import Company
from .schemas import CompanyLoginOut
from .security import create_access_token

def login_by_coid(db: Session, co_id_input: str) -> CompanyLoginOut:
    co_id = (co_id_input or "").strip()
    if not co_id:
        raise ValueError("coId는 필수입니다.")

    # citext 미사용 가정: LOWER 비교로 대소문자 무시
    stmt = select(Company).where(func.lower(Company.co_id) == func.lower(co_id)).limit(1)
    company = db.execute(stmt).scalars().first()
    if not company:
        raise LookupError("존재하지 않는 회사 계정입니다.")

    role = "admin"  # 요구사항: co_id만 맞으면 관리자
    token = create_access_token(company.id, company.co_id, role)

    # 여기선 snake_case 필드로 생성 (schemas가 camelCase로 직렬화)
    return CompanyLoginOut(
        company_id=company.id,
        co_id=company.co_id,
        co_name=company.co_name,
        role=role,
        access_token=token,
    )
