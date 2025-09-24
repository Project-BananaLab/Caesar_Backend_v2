# crud.py
# CRUD (Create, Read, Update, Delete) 작업을 수행하는 함수들을 정의합니다.
# 데이터베이스 로직을 API 엔드포인트와 분리하여 코드를 더 깔끔하게 관리할 수 있습니다.

from sqlalchemy.orm import Session
from . import models, schemas, utils

def get_employee_by_google_id(db: Session, google_user_id: str):
    """
    Google User ID를 사용하여 직원을 조회합니다.
    부서명과 직급명을 포함하여 조회합니다.
    :param db: 데이터베이스 세션
    :param google_user_id: 조회할 구글 사용자 ID
    :return: Employee 모델 객체 또는 None (부서명, 직급명 포함)
    """
    # LEFT JOIN을 사용해서 부서와 직급 정보를 함께 조회합니다.
    employee = db.query(models.Employee)\
        .outerjoin(models.JobDept, models.Employee.job_dept_id == models.JobDept.id)\
        .outerjoin(models.JobRank, models.Employee.job_rank_id == models.JobRank.id)\
        .filter(models.Employee.google_user_id == google_user_id)\
        .first()
    
    if employee:
        # 부서명과 직급명을 동적으로 추가
        employee.dept_name = employee.job_dept.dept_name if employee.job_dept else None
        employee.rank_name = employee.job_rank.rank_name if employee.job_rank else None
        
        # API 키 존재 여부를 동적으로 추가 (보안을 위해 실제 값은 반환하지 않음)
        employee.has_notion_api = employee.notion_api is not None and len(employee.notion_api) > 0
        employee.has_slack_api = employee.slack_api is not None and len(employee.slack_api) > 0
    
    return employee

def create_employee(db: Session, employee: schemas.EmployeeCreate):
    """
    새로운 직원을 데이터베이스에 생성합니다.
    :param db: 데이터베이스 세션
    :param employee: 생성할 직원 정보 (Pydantic 스키마)
    :return: 생성된 Employee 모델 객체
    """
    # Pydantic 스키마를 SQLAlchemy 모델 객체로 변환합니다.
    db_employee = models.Employee(
        google_user_id=employee.google_user_id,
        email=employee.email,
        full_name=employee.full_name
    )
    db.add(db_employee) # 세션에 객체 추가
    db.commit()        # 데이터베이스에 변경 사항 저장
    db.refresh(db_employee) # 생성된 객체(예: auto-increment id)를 다시 로드
    return db_employee

def update_employee_api_keys(db: Session, google_user_id: str, api_keys: schemas.APIKeysUpdate):
    """
    직원의 Notion 및 Slack API 키를 암호화하여 업데이트합니다.
    :param db: 데이터베이스 세션
    :param google_user_id: 업데이트할 직원의 구글 사용자 ID
    :param api_keys: 업데이트할 API 키 정보 (Pydantic 스키마)
    :return: 업데이트된 Employee 모델 객체 또는 None
    """
    # 먼저 해당 직원을 조회합니다.
    db_employee = get_employee_by_google_id(db, google_user_id)
    if db_employee:
        # Notion API 키 업데이트 (None이 아닌 경우에만)
        if api_keys.notion_api is not None:
            if api_keys.notion_api.strip() == "":
                # 빈 문자열이면 None으로 저장 (삭제)
                db_employee.notion_api = None
            else:
                # 값이 있으면 암호화하여 저장
                db_employee.notion_api = utils.encrypt_data(api_keys.notion_api)
        
        # Slack API 키 업데이트 (None이 아닌 경우에만)
        if api_keys.slack_api is not None:
            if api_keys.slack_api.strip() == "":
                # 빈 문자열이면 None으로 저장 (삭제)
                db_employee.slack_api = None
            else:
                # 값이 있으면 암호화하여 저장
                db_employee.slack_api = utils.encrypt_data(api_keys.slack_api)
        
        db.commit() # 변경 사항 저장
        db.refresh(db_employee) # 객체 새로고침
    return db_employee
