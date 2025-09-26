# models.py
# SQLAlchemy ORM 모델을 정의하여 데이터베이스 테이블과 파이썬 클래스를 매핑합니다. 

from sqlalchemy import Column, Integer, String, LargeBinary, ForeignKey
from sqlalchemy.orm import relationship
from app.utils.db import Base


# 'job_dept' 테이블에 매핑될 JobDept 클래스를 정의합니다.
class JobDept(Base):
    __tablename__ = "job_dept"

    id = Column(Integer, primary_key=True, index=True)
    dept_name = Column(String(100), nullable=False)


# 'job_rank' 테이블에 매핑될 JobRank 클래스를 정의합니다.
class JobRank(Base):
    __tablename__ = "job_rank"

    id = Column(Integer, primary_key=True, index=True)
    rank_name = Column(String(100), nullable=False)


# 'employee' 테이블에 매핑될 Employee 클래스를 정의합니다.
class Employee(Base):
    # __tablename__은 이 모델이 사용할 데이터베이스 테이블의 이름을 지정합니다.
    __tablename__ = "employee"

    # 각 컬럼을 정의합니다.
    id = Column(Integer, primary_key=True, index=True)  # 기본 키
    company_id = Column(Integer, ForeignKey("company.id"), nullable=False)
    job_dept_id = Column(Integer, ForeignKey("job_dept.id"), nullable=True)
    job_rank_id = Column(Integer, ForeignKey("job_rank.id"), nullable=True)
    # 암호화된 API 키는 바이너리 데이터이므로 LargeBinary 타입을 사용합니다.
    notion_api = Column(LargeBinary, nullable=True)
    slack_api = Column(LargeBinary, nullable=True)
    # 구글 사용자 정보
    google_user_id = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(100))

    # 관계 설정 - 부서와 직급 정보를 쉽게 조회할 수 있습니다.
    job_dept = relationship("JobDept", foreign_keys=[job_dept_id])
    job_rank = relationship("JobRank", foreign_keys=[job_rank_id])
