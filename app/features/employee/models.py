# models.py

from sqlalchemy import Column, Integer, String, LargeBinary
from .database import Base

class Employee(Base):
    __tablename__ = "employee"  # 데이터베이스 테이블 이름

    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String(50), index=True)  # Match existing column name
    user_email = Column(String(255), unique=True, index=True, nullable=False)  # Match existing column name
    hashed_password = Column(String(255), nullable=False)
    
    # Additional columns from existing schema
    company_id = Column(Integer, nullable=True)
    job_dept_id = Column(Integer, nullable=True)
    job_rank_id = Column(Integer, nullable=True)

    # Match existing column names for integrations (these are already LargeBinary/BYTEA)
    google_calendar_json = Column(LargeBinary, nullable=True)
    google_drive_json = Column(LargeBinary, nullable=True)
    notion_api = Column(LargeBinary, nullable=True)
    slack_api = Column(LargeBinary, nullable=True)
    
    # Properties to maintain compatibility with schemas
    @property
    def name(self):
        return self.user_name
        
    @property
    def email(self):
        return self.user_email