# app/features/admin/manage_employee/service.py
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
        관련된 데이터도 함께 정리합니다:
        - channels: 해당 직원이 소유한 채널들
        - docs: 해당 직원의 개인 문서들 (is_private=True)
        
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
        
        try:
            print(f"🗑️ 직원 연쇄 삭제 시작: {employee.full_name} (ID: {employee_id})")
            
            # 1. 해당 직원이 소유한 channels 삭제 (chats도 cascade로 함께 삭제됨)
            from app.features.channel.models.channel_models import Channel
            channels = db.query(Channel).filter(Channel.employee_id == employee_id).all()
            for channel in channels:
                db.delete(channel)
            print(f"🗑️ {len(channels)}개 채널 삭제됨")
            
            # 2. 해당 직원의 모든 문서 삭제 (개인문서 + 회사문서 중 해당 직원이 업로드한 것)
            from app.features.admin.models.docs import Doc
            from app.features.admin.services.file_ingest_service import delete_doc_everywhere
            
            # 해당 직원이 업로드한 모든 문서 (개인 + 회사 공개)
            all_employee_docs = db.query(Doc).filter(
                Doc.employee_id == employee_id
            ).all()
            
            for doc in all_employee_docs:
                try:
                    # 각 문서를 DB/S3/VectorDB에서 완전 삭제
                    delete_doc_everywhere(db, doc_id=doc.id)
                except Exception as e:
                    print(f"❌ 문서 삭제 실패 (doc_id={doc.id}): {e}")
            
            print(f"🗑️ {len(all_employee_docs)}개 문서 삭제됨")
            
            # 3. 기타 employee_id를 참조하는 데이터 정리
            # TODO: 향후 추가되는 테이블들도 여기서 정리
            
            # 4. 직원 삭제
            db.delete(employee)
            db.commit()
            
            print(f"✅ 직원 연쇄 삭제 완료: {employee.full_name} (ID: {employee_id})")
            print(f"   - 채널: {len(channels)}개")
            print(f"   - 문서: {len(all_employee_docs)}개")
            return True
            
        except Exception as e:
            db.rollback()
            print(f"❌ 직원 삭제 중 오류 발생: {e}")
            raise e

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
