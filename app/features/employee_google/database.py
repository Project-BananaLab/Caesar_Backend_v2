# database.py
# 데이터베이스 연결 설정 및 세션 관리를 담당합니다.

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# .env 파일에 저장된 데이터베이스 URL을 가져옵니다.
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# 데이터베이스 엔진을 생성합니다. 이 엔진은 SQLAlchemy가 데이터베이스와 통신하는 시작점입니다.
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 데이터베이스 세션을 생성하기 위한 SessionLocal 클래스를 정의합니다.
# 이 세션은 실제 데이터베이스 작업을 수행하는 단위가 됩니다.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemy 모델을 만들기 위한 기본 클래스(Base)를 생성합니다.
# 모든 모델 클래스는 이 Base 클래스를 상속받아야 합니다.
Base = declarative_base()

# API 요청마다 데이터베이스 세션을 생성하고, 요청이 끝나면 세션을 닫는 의존성 함수입니다.
# 이 함수를 통해 API 엔드포인트에서 데이터베이스 세션을 안전하게 사용할 수 있습니다.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
