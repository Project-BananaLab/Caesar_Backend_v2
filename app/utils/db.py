# app/utils/db.py
from app.utils.env_loader import env_tokens

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.core.config import settings

# SQLAlchemy 엔진 (pre-ping으로 끊어진 연결 자동 감지)
engine = create_engine(settings.DB_URL, pool_pre_ping=True)

# 세션 팩토리                                               # 선택: 커밋 후 객체 즉시 재사용할 때 편함
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


# Base 클래스 (모든 모델이 상속)
class Base(DeclarativeBase):
    # 기본 스키마를 caesar로 설정
    metadata = MetaData(schema="caesar")


# FastAPI 의존성: 요청당 세션 제공
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 사용자별 OAuth 토큰 저장소
user_tokens = {
    "user_123": {
        "google": env_tokens["google"],
        "slack": env_tokens["slack"],
        "notion": env_tokens["notion"],
    }
}


def get_user_tokens(user_id: str) -> dict:
    """사용자 ID로 토큰 정보 조회"""
    return user_tokens.get(user_id, {})


def save_user_tokens(user_id: str, service: str, tokens: dict):
    """사용자 토큰 저장"""
    if user_id not in user_tokens:
        user_tokens[user_id] = {}
    user_tokens[user_id][service] = tokens


def get_service_token(user_id: str, service: str) -> dict:
    """특정 서비스의 토큰 조회"""
    return user_tokens.get(user_id, {}).get(service, {})


def get_user_api_tokens_from_db(google_user_id: str) -> dict:
    """
    데이터베이스에서 사용자별 API 토큰을 조회합니다.
    google_user_id로 조회하여 notion_api와 slack_api를 반환합니다.
    """
    from app.features.employee_google.crud import get_employee_by_google_id
    from app.utils.crypto_utils import decrypt_data
    
    print(f"🔍 DB에서 토큰 조회 시작 - Google User ID: {google_user_id}")
    
    db = SessionLocal()
    try:
        employee = get_employee_by_google_id(db, google_user_id)
        if not employee:
            print(f"❌ 사용자를 찾을 수 없습니다 - Google User ID: {google_user_id}")
            return {}
        
        print(f"✅ 사용자 찾음 - ID: {employee.id}, 이름: {employee.full_name}, 이메일: {employee.email}")    
        print(f"🔍 Notion API 필드 존재 여부: {employee.notion_api is not None}")
        print(f"🔍 Slack API 필드 존재 여부: {employee.slack_api is not None}")
            
        tokens = {}
        
        # Notion API 토큰 복호화
        if employee.notion_api:
            try:
                print(f"🔓 Notion 토큰 복호화 시도 중...")
                notion_token = decrypt_data(employee.notion_api, "string")
                tokens["notion"] = {"token": notion_token}
                print(f"✅ Notion 토큰 복호화 성공 - 토큰 길이: {len(notion_token)}")
            except Exception as e:
                print(f"❌ Notion 토큰 복호화 실패: {e}")
        else:
            print("❌ Notion API 토큰이 데이터베이스에 저장되어 있지 않습니다.")
        
        # Slack API 토큰 복호화  
        if employee.slack_api:
            try:
                print(f"🔓 Slack 토큰 복호화 시도 중...")
                slack_token = decrypt_data(employee.slack_api, "string")
                tokens["slack"] = {"user_token": slack_token}
                print(f"✅ Slack 토큰 복호화 성공 - 토큰 길이: {len(slack_token)}")
            except Exception as e:
                print(f"❌ Slack 토큰 복호화 실패: {e}")
        else:
            print("❌ Slack API 토큰이 데이터베이스에 저장되어 있지 않습니다.")
        
        print(f"🎯 최종 반환 토큰: {list(tokens.keys())}")
        return tokens
        
    finally:
        db.close()


def get_service_token_enhanced(user_id: str, service: str) -> dict:
    """
    특정 서비스의 토큰 조회 (DB에서 사용자별 토큰 우선 조회)
    1. 먼저 DB에서 google_user_id로 사용자별 토큰 조회
    2. 없으면 기존 메모리 저장소에서 조회
    3. 그것도 없으면 env_tokens에서 기본값 조회
    """
    print(f"🔍 get_service_token_enhanced 호출 - User ID: {user_id}, Service: {service}")
    
    # 1. DB에서 사용자별 토큰 조회
    db_tokens = get_user_api_tokens_from_db(user_id)
    print(f"🔍 DB에서 조회된 토큰들: {list(db_tokens.keys())}")
    
    if service in db_tokens:
        print(f"✅ DB에서 {service} 토큰 찾음")
        return db_tokens[service]
    else:
        print(f"❌ DB에서 {service} 토큰을 찾을 수 없음")
    
    # # 2. 기존 메모리 저장소에서 조회 (잘 되면 삭제 해도 무방)
    # from app.utils.env_loader import env_tokens
    # user_service_token = user_tokens.get(user_id, {}).get(service)
    # if user_service_token:
    #     print(f"✅ 메모리에서 {service} 토큰 찾음")
    #     return user_service_token
    # else:
    #     print(f"❌ 메모리에서 {service} 토큰을 찾을 수 없음")
        
    # # 3. env_tokens에서 기본값 조회
    # env_token = env_tokens.get(service, {})
    # if env_token:
    #     print(f"✅ 환경변수에서 {service} 토큰 찾음")
    #     return env_token
    # else:
    #     print(f"❌ 환경변수에서 {service} 토큰을 찾을 수 없음")
        
    # print(f"❌ 모든 소스에서 {service} 토큰을 찾을 수 없음")
    # return {}

from app.features.login.company.models import Company
from app.utils.crypto_utils import decrypt_data

def get_notion_token_by_company(company_id: int) -> str:
    """회사 ID로 Notion API 토큰 가져오기"""
    db = SessionLocal()
    try:
        company = db.query(Company).filter(Company.id == company_id).first()
        if company and company.co_notion_API:
            return decrypt_data(company.co_notion_API, return_type="string")
    finally:
        db.close()