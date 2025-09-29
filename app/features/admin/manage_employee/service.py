# service.py
# 직원 관리 비즈니스 로직을 처리하는 서비스 계층입니다.

from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Optional
from app.features.login.employee_google.models import Employee, JobDept, JobRank
from app.features.admin.manage_employee.schemas import EmployeeUpdateRequest


class EmployeeManagementService:
    """직원 관리 서비스 클래스"""

    @staticmethod
    def get_employees_by_company(db: Session, company_id: int) -> List[Employee]:
        """
        특정 회사의 모든 직원을 조회합니다.
        
        Args:
            db: 데이터베이스 세션
            company_id: 회사 ID
            
        Returns:
            해당 회사의 직원 목록
        """
        return (
            db.query(Employee)
            .filter(Employee.company_id == company_id)
            .order_by(Employee.full_name)
            .all()
        )

    @staticmethod
    def get_employee_by_id(db: Session, employee_id: int, company_id: int) -> Optional[Employee]:
        """
        특정 회사의 직원을 ID로 조회합니다.
        
        Args:
            db: 데이터베이스 세션
            employee_id: 직원 ID
            company_id: 회사 ID (보안을 위해 회사 소속 확인)
            
        Returns:
            직원 정보 또는 None
        """
        return (
            db.query(Employee)
            .filter(
                and_(
                    Employee.id == employee_id,
                    Employee.company_id == company_id
                )
            )
            .first()
        )

    @staticmethod
    def update_employee(
        db: Session, 
        employee_id: int, 
        company_id: int, 
        update_data: EmployeeUpdateRequest
    ) -> Optional[Employee]:
        """
        직원 정보를 수정합니다.
        
        Args:
            db: 데이터베이스 세션
            employee_id: 직원 ID
            company_id: 회사 ID
            update_data: 수정할 데이터
            
        Returns:
            수정된 직원 정보 또는 None
        """
        employee = EmployeeManagementService.get_employee_by_id(db, employee_id, company_id)
        if not employee:
            return None
        
        # 부서 ID 업데이트
        if update_data.job_dept_id is not None:
            employee.job_dept_id = update_data.job_dept_id
        
        # 직급 ID 업데이트
        if update_data.job_rank_id is not None:
            employee.job_rank_id = update_data.job_rank_id
        
        db.commit()
        db.refresh(employee)
        return employee

    @staticmethod
    def delete_employee(db: Session, employee_id: int, company_id: int) -> bool:
        """
        직원을 삭제합니다.
        
        Args:
            db: 데이터베이스 세션
            employee_id: 직원 ID
            company_id: 회사 ID
            
        Returns:
            삭제 성공 여부
        """
        employee = EmployeeManagementService.get_employee_by_id(db, employee_id, company_id)
        if not employee:
            return False
        
        db.delete(employee)
        db.commit()
        return True

    @staticmethod
    def get_all_departments(db: Session) -> List[JobDept]:
        """
        모든 부서 목록을 조회합니다.
        
        Args:
            db: 데이터베이스 세션
            
        Returns:
            부서 목록
        """
        return db.query(JobDept).order_by(JobDept.dept_name).all()

    @staticmethod
    def get_all_ranks(db: Session) -> List[JobRank]:
        """
        모든 직급 목록을 조회합니다.
        
        Args:
            db: 데이터베이스 세션
            
        Returns:
            직급 목록
        """
        return db.query(JobRank).order_by(JobRank.rank_name).all()

    @staticmethod
    def get_employee_with_details(db: Session, company_id: int) -> List[dict]:
        """
        회사 직원들의 상세 정보를 부서명, 직급명과 함께 조회합니다.
        
        Args:
            db: 데이터베이스 세션
            company_id: 회사 ID
            
        Returns:
            직원 상세 정보 목록 (부서명, 직급명 포함)
        """
        # Employee와 JobDept, JobRank를 조인하여 조회
        result = (
            db.query(Employee, JobDept.dept_name, JobRank.rank_name)
            .outerjoin(JobDept, Employee.job_dept_id == JobDept.id)
            .outerjoin(JobRank, Employee.job_rank_id == JobRank.id)
            .filter(Employee.company_id == company_id)
            .order_by(Employee.full_name)
            .all()
        )
        
        # 결과를 딕셔너리 형태로 변환 (None 값 안전 처리)
        employees = []
        for employee, dept_name, rank_name in result:
            employee_dict = {
                "id": employee.id,
                "full_name": employee.full_name or "",  # None 처리
                "email": employee.email or "",  # None 처리
                "job_dept_id": employee.job_dept_id,
                "job_rank_id": employee.job_rank_id,
                "dept_name": dept_name,
                "rank_name": rank_name,
                "company_id": employee.company_id,
                "google_user_id": employee.google_user_id or "",  # None 처리
            }
            employees.append(employee_dict)
        
        return employees
