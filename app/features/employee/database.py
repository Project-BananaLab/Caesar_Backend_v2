# database.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

# PostgreSQL 연결 정보 (사용자 환경에 맞게 수정)
# 형식: "postgresql://사용자이름:비밀번호@호스트:포트/데이터베이스이름"
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# 데이터베이스 엔진 생성
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 데이터베이스 세션 생성을 위한 SessionLocal 클래스
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemy 모델의 기본 클래스
Base = declarative_base()

# Dependency: API 요청마다 독립적인 DB 세션을 생성하고, 끝나면 닫는 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()