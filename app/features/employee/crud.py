# crud.py

from sqlalchemy.orm import Session
from . import models, schemas, security

def get_user_by_email(db: Session, email: str):
    """이메일로 사용자를 조회합니다."""
    return db.query(models.Employee).filter(models.Employee.user_email == email).first()

def create_user(db: Session, user: schemas.UserCreate):
    """새로운 사용자를 생성합니다."""
    # 비밀번호를 해시하여 저장
    hashed_password = security.get_password_hash(user.password)
    db_user = models.Employee(
        user_email=user.email,
        user_name=user.name,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user_integrations(db: Session, user_id: int, data: schemas.IntegrationsUpdate):
    """사용자의 외부 서비스 연동 정보를 암호화하여 업데이트합니다."""
    db_user = db.query(models.Employee).filter(models.Employee.id == user_id).first()
    if not db_user:
        return None

    update_data = data.model_dump(exclude_unset=True) # 입력된 값만 가져옴

    # 필드명 매핑: 스키마 -> 모델 (existing columns don't have encrypted_ prefix)
    field_mapping = {
        'google_calendar_json': 'google_calendar_json',
        'google_drive_json': 'google_drive_json', 
        'notion_api': 'notion_api',
        'slack_api': 'slack_api'
    }

    for key, value in update_data.items():
        if value and key in field_mapping:
            # 각 필드의 데이터를 암호화하여 저장
            encrypted_value = security.encrypt_data(value)
            setattr(db_user, field_mapping[key], encrypted_value)

    db.commit()
    db.refresh(db_user)
    return db_user

# 복호화 예시 함수 (실제 사용 사례에 맞게 응용)
def get_decrypted_notion_api(db: Session, user_id: int) -> str | None:
    """사용자의 Notion API 키를 복호화하여 반환합니다."""
    user = db.query(models.Employee).filter(models.Employee.id == user_id).first()
    if user and user.notion_api:
        return security.decrypt_data(user.notion_api)
    return None