# app/features/user/router/employee_router.py
"""
직원 정보 관리 API 라우터
간단하고 이해하기 쉽게 작성된 PostgreSQL 연결 및 CRUD 작업
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from ..schemas.employee_schemas import Employee, EmployeeCreate, JobDept, JobRank
from app.utils.crypto_utils import encrypt_data, decrypt_data

# APIRouter 생성
router = APIRouter(
    prefix="/employee",
    tags=["직원 관리"],
    responses={404: {"description": "Not found"}},
)

# 데이터베이스 연결 설정
def get_database_url():
    """
    환경변수에서 데이터베이스 URL을 가져옵니다.
    """
    return os.getenv("POSTGRESQL_DATABASE_URL")

def get_db_connection():
    """
    PostgreSQL 데이터베이스 연결을 생성합니다.
    """
    try:
        conn = psycopg2.connect(get_database_url(), cursor_factory=RealDictCursor)
        conn.cursor().execute("SET search_path TO caesar;")
        return conn
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=500, 
            detail=f"데이터베이스 연결 실패: {str(e)}"
        )

# 1. 모든 직원 조회
@router.get("/", response_model=List[Employee])
async def get_all_employees():
    """
    모든 직원 정보를 조회합니다.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, company_id, job_dept_id, job_rank_id, 
                   user_name, google_uid, google_calendar_json, 
                   google_drive_json, notion_api, slack_api
            FROM employee
            ORDER BY id
        """)
        
        employees = cursor.fetchall()
        
        # 암호화된 필드들을 복호화
        decrypted_employees = []
        for employee in employees:
            employee_dict = dict(employee)
            
            # google_calendar_json 복호화
            if employee_dict.get('google_calendar_json'):
                employee_dict['google_calendar_json'] = decrypt_data(employee_dict['google_calendar_json'])
            
            # google_drive_json 복호화
            if employee_dict.get('google_drive_json'):
                employee_dict['google_drive_json'] = decrypt_data(employee_dict['google_drive_json'])
            
            # notion_api 복호화
            if employee_dict.get('notion_api'):
                employee_dict['notion_api'] = decrypt_data(employee_dict['notion_api'])
            
            # slack_api 복호화
            if employee_dict.get('slack_api'):
                employee_dict['slack_api'] = decrypt_data(employee_dict['slack_api'])
            
            decrypted_employees.append(employee_dict)
        
        return decrypted_employees
        
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=500, 
            detail=f"직원 조회 실패: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

# 2. 특정 직원 조회
@router.get("/{employee_id}", response_model=Employee)
async def get_employee(employee_id: int):
    """
    ID로 특정 직원 정보를 조회합니다.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, company_id, job_dept_id, job_rank_id, 
                   user_name, google_uid, google_calendar_json, 
                   google_drive_json, notion_api, slack_api
            FROM employee 
            WHERE id = %s
        """, (employee_id,))
        
        employee = cursor.fetchone()
        
        if not employee:
            raise HTTPException(
                status_code=404, 
                detail=f"ID {employee_id}인 직원을 찾을 수 없습니다."
            )
        
        # 암호화된 필드들을 복호화해서 반환
        employee_dict = dict(employee)
        
        # google_calendar_json 복호화
        if employee_dict.get('google_calendar_json'):
            employee_dict['google_calendar_json'] = decrypt_data(employee_dict['google_calendar_json'])
        
        # google_drive_json 복호화
        if employee_dict.get('google_drive_json'):
            employee_dict['google_drive_json'] = decrypt_data(employee_dict['google_drive_json'])
        
        # notion_api 복호화
        if employee_dict.get('notion_api'):
            employee_dict['notion_api'] = decrypt_data(employee_dict['notion_api'])
        
        # slack_api 복호화
        if employee_dict.get('slack_api'):
            employee_dict['slack_api'] = decrypt_data(employee_dict['slack_api'])
        
        return employee_dict
        
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=500, 
            detail=f"직원 조회 실패: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

# 3. 새 직원 추가
@router.post("/", response_model=Employee)
async def create_employee(employee_data: EmployeeCreate):
    """
    새로운 직원 정보를 추가합니다.
    """
    conn = get_db_connection()
    try:
        encrypted_calendar = encrypt_data(employee_data.google_calendar_json) if employee_data.google_calendar_json else None
        encrypted_drive = encrypt_data(employee_data.google_drive_json) if employee_data.google_drive_json else None
        encrypted_notion = encrypt_data(employee_data.notion_api) if employee_data.notion_api else None
        encrypted_slack = encrypt_data(employee_data.slack_api) if employee_data.slack_api else None

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO employee (company_id, job_dept_id, job_rank_id, 
                                 user_name, google_uid, google_calendar_json, 
                                 google_drive_json, notion_api, slack_api)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, company_id, job_dept_id, job_rank_id, 
                      user_name, google_uid, google_calendar_json, 
                      google_drive_json, notion_api, slack_api
        """, (
            employee_data.company_id,
            employee_data.job_dept_id,
            employee_data.job_rank_id,
            employee_data.user_name,
            employee_data.google_uid,
            encrypted_calendar,
            encrypted_drive,
            encrypted_notion,
            encrypted_slack,
        ))
        
        new_employee = cursor.fetchone()
        conn.commit()
        
        # 암호화된 필드들을 복호화해서 반환
        employee_dict = dict(new_employee)
        
        # google_calendar_json 복호화
        if employee_dict.get('google_calendar_json'):
            try:
                employee_dict['google_calendar_json'] = decrypt_data(employee_dict['google_calendar_json'])
            except Exception as e:
                print(f"Error decrypting google_calendar_json: {e}, type: {type(employee_dict['google_calendar_json'])}")
                raise HTTPException(status_code=500, detail=f"복호화 실패 (google_calendar_json): {str(e)}")
        
        # google_drive_json 복호화
        if employee_dict.get('google_drive_json'):
            try:
                employee_dict['google_drive_json'] = decrypt_data(employee_dict['google_drive_json'])
            except Exception as e:
                print(f"Error decrypting google_drive_json: {e}, type: {type(employee_dict['google_drive_json'])}")
                raise HTTPException(status_code=500, detail=f"복호화 실패 (google_drive_json): {str(e)}")
        
        # notion_api 복호화
        if employee_dict.get('notion_api'):
            try:
                employee_dict['notion_api'] = decrypt_data(employee_dict['notion_api'])
            except Exception as e:
                print(f"Error decrypting notion_api: {e}, type: {type(employee_dict['notion_api'])}")
                raise HTTPException(status_code=500, detail=f"복호화 실패 (notion_api): {str(e)}")
        
        # slack_api 복호화
        if employee_dict.get('slack_api'):
            try:
                employee_dict['slack_api'] = decrypt_data(employee_dict['slack_api'])
            except Exception as e:
                print(f"Error decrypting slack_api: {e}, type: {type(employee_dict['slack_api'])}")
                raise HTTPException(status_code=500, detail=f"복호화 실패 (slack_api): {str(e)}")
        
        return employee_dict
        
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"직원 추가 실패: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

# 4. 직원 정보 수정
@router.put("/{employee_id}", response_model=Employee)
async def update_employee(employee_id: int, employee_data: EmployeeCreate):
    """
    기존 직원 정보를 수정합니다.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # 먼저 직원이 존재하는지 확인
        cursor.execute("SELECT id FROM employee WHERE id = %s", (employee_id,))
        if not cursor.fetchone():
            raise HTTPException(
                status_code=404, 
                detail=f"ID {employee_id}인 직원을 찾을 수 없습니다."
            )
        
        encrypted_calendar = encrypt_data(employee_data.google_calendar_json) if employee_data.google_calendar_json else None
        encrypted_drive = encrypt_data(employee_data.google_drive_json) if employee_data.google_drive_json else None
        encrypted_notion = encrypt_data(employee_data.notion_api) if employee_data.notion_api else None
        encrypted_slack = encrypt_data(employee_data.slack_api) if employee_data.slack_api else None

        # 직원 정보 업데이트
        cursor.execute("""
            UPDATE employee 
            SET company_id = %s, job_dept_id = %s, job_rank_id = %s,
                user_name = %s, google_uid = %s, google_calendar_json = %s,
                google_drive_json = %s, notion_api = %s, slack_api = %s
            WHERE id = %s
            RETURNING id, company_id, job_dept_id, job_rank_id, 
                      user_name, google_uid, google_calendar_json, 
                      google_drive_json, notion_api, slack_api
        """, (
            employee_data.company_id,
            employee_data.job_dept_id,
            employee_data.job_rank_id,
            employee_data.user_name,
            employee_data.google_uid,
            encrypted_calendar,
            encrypted_drive,
            encrypted_notion,
            encrypted_slack,
            employee_id
        ))
        
        updated_employee = cursor.fetchone()
        conn.commit()
        
        # 암호화된 필드들을 복호화해서 반환
        employee_dict = dict(updated_employee)
        
        # google_calendar_json 복호화
        if employee_dict.get('google_calendar_json'):
            employee_dict['google_calendar_json'] = decrypt_data(employee_dict['google_calendar_json'])
        
        # google_drive_json 복호화
        if employee_dict.get('google_drive_json'):
            employee_dict['google_drive_json'] = decrypt_data(employee_dict['google_drive_json'])
        
        # notion_api 복호화
        if employee_dict.get('notion_api'):
            employee_dict['notion_api'] = decrypt_data(employee_dict['notion_api'])
        
        # slack_api 복호화
        if employee_dict.get('slack_api'):
            employee_dict['slack_api'] = decrypt_data(employee_dict['slack_api'])
        
        return employee_dict
        
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"직원 수정 실패: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

# 5. 직원 삭제
@router.delete("/{employee_id}")
async def delete_employee(employee_id: int):
    """
    직원 정보를 삭제합니다.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # 먼저 직원이 존재하는지 확인
        cursor.execute("SELECT id FROM employee WHERE id = %s", (employee_id,))
        if not cursor.fetchone():
            raise HTTPException(
                status_code=404, 
                detail=f"ID {employee_id}인 직원을 찾을 수 없습니다."
            )
        
        # 직원 삭제
        cursor.execute("DELETE FROM employee WHERE id = %s", (employee_id,))
        conn.commit()
        
        return {"message": f"ID {employee_id}인 직원이 성공적으로 삭제되었습니다."}
        
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"직원 삭제 실패: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

# 6. 부서별 직원 조회
@router.get("/department/{dept_id}", response_model=List[Employee])
async def get_employees_by_department(dept_id: int):
    """
    특정 부서의 모든 직원을 조회합니다.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, company_id, job_dept_id, job_rank_id, 
                   user_name, google_uid, google_calendar_json, 
                   google_drive_json, notion_api, slack_api
            FROM employee
            WHERE job_dept_id = %s
            ORDER BY id
        """, (dept_id,))
        
        employees = cursor.fetchall()
        
        # 암호화된 필드들을 복호화
        decrypted_employees = []
        for employee in employees:
            employee_dict = dict(employee)
            
            # google_calendar_json 복호화
            if employee_dict.get('google_calendar_json'):
                employee_dict['google_calendar_json'] = decrypt_data(employee_dict['google_calendar_json'])
            
            # google_drive_json 복호화
            if employee_dict.get('google_drive_json'):
                employee_dict['google_drive_json'] = decrypt_data(employee_dict['google_drive_json'])
            
            # notion_api 복호화
            if employee_dict.get('notion_api'):
                employee_dict['notion_api'] = decrypt_data(employee_dict['notion_api'])
            
            # slack_api 복호화
            if employee_dict.get('slack_api'):
                employee_dict['slack_api'] = decrypt_data(employee_dict['slack_api'])
            
            decrypted_employees.append(employee_dict)
        
        return decrypted_employees
        
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=500, 
            detail=f"부서별 직원 조회 실패: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

# 7. 직급별 직원 조회
@router.get("/rank/{rank_id}", response_model=List[Employee])
async def get_employees_by_rank(rank_id: int):
    """
    특정 직급의 모든 직원을 조회합니다.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, company_id, job_dept_id, job_rank_id, 
                   user_name, google_uid, google_calendar_json, 
                   google_drive_json, notion_api, slack_api
            FROM employee
            WHERE job_rank_id = %s
            ORDER BY id
        """, (rank_id,))
        
        employees = cursor.fetchall()
        
        # 암호화된 필드들을 복호화
        decrypted_employees = []
        for employee in employees:
            employee_dict = dict(employee)
            
            # google_calendar_json 복호화
            if employee_dict.get('google_calendar_json'):
                employee_dict['google_calendar_json'] = decrypt_data(employee_dict['google_calendar_json'])
            
            # google_drive_json 복호화
            if employee_dict.get('google_drive_json'):
                employee_dict['google_drive_json'] = decrypt_data(employee_dict['google_drive_json'])
            
            # notion_api 복호화
            if employee_dict.get('notion_api'):
                employee_dict['notion_api'] = decrypt_data(employee_dict['notion_api'])
            
            # slack_api 복호화
            if employee_dict.get('slack_api'):
                employee_dict['slack_api'] = decrypt_data(employee_dict['slack_api'])
            
            decrypted_employees.append(employee_dict)
        
        return decrypted_employees
        
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=500, 
            detail=f"직급별 직원 조회 실패: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

# 8. 부서 정보 조회
@router.get("/departments/", response_model=List[JobDept])
async def get_all_departments():
    """
    모든 부서 정보를 조회합니다.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, dept_name 
            FROM job_dept
            ORDER BY id
        """)
        
        departments = cursor.fetchall()
        return [dict(dept) for dept in departments]
        
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=500, 
            detail=f"부서 조회 실패: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

# 9. 직급 정보 조회
@router.get("/ranks/", response_model=List[JobRank])
async def get_all_ranks():
    """
    모든 직급 정보를 조회합니다.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, rank_name 
            FROM job_rank
            ORDER BY id
        """)
        
        ranks = cursor.fetchall()
        return [dict(rank) for rank in ranks]
        
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=500, 
            detail=f"직급 조회 실패: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

# 10. 데이터베이스 연결 상태 확인
@router.get("/health/database")
async def check_database_connection():
    """
    데이터베이스 연결 상태를 확인합니다.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "message": "데이터베이스 연결이 정상입니다.",
            "database_url": get_database_url().replace(
                get_database_url().split("@")[0].split("//")[1], 
                "****"
            )  # 비밀번호 숨기기
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"데이터베이스 연결 실패: {str(e)}"
        )
