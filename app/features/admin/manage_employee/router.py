# router.py
# 직원 관리 API 엔드포인트를 정의합니다.

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.utils.db import get_db
from app.features.admin.manage_employee.service import EmployeeManagementService
from app.features.admin.manage_employee.schemas import (
    EmployeeResponse,
    EmployeeUpdateRequest,
    EmployeeListResponse,
    DeptResponse,
    RankResponse
)
from app.features.auth.company_auth import get_current_company_admin

router = APIRouter(prefix="/api/admin/employees", tags=["직원 관리"])

@router.get("/list", response_model=List[EmployeeResponse])
async def list_employees(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_company_admin)
):
    """
    현재 로그인한 관리자 회사의 모든 직원 목록을 조회합니다.
    
    Returns:
        직원 목록과 상세 정보 (부서명, 직급명 포함)
    """
    try:
        # 현재 사용자의 회사 ID 가져오기
        company_id = current_user.get("company_id")
        if not company_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="회사 정보를 찾을 수 없습니다."
            )
        
        # 회사 직원들의 상세 정보 조회
        employees_data = EmployeeManagementService.get_employee_with_details(db, company_id)
        
        # 응답 형태로 변환 (안전한 데이터 처리)
        employees = []
        for emp_data in employees_data:
            employee = EmployeeResponse(
                id=emp_data["id"],
                full_name=emp_data.get("full_name") or "",
                email=emp_data.get("email") or "",
                job_dept_id=emp_data.get("job_dept_id"),
                job_rank_id=emp_data.get("job_rank_id"),
                dept_name=emp_data.get("dept_name"),
                rank_name=emp_data.get("rank_name"),
                company_id=emp_data["company_id"],
                google_user_id=emp_data.get("google_user_id") or ""
            )
            employees.append(employee)
        
        return employees
        
    except Exception as e:
        print(f"❌ 직원 목록 조회 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="직원 목록을 조회하는 중 오류가 발생했습니다."
        )


@router.get("/{employee_id}", response_model=EmployeeResponse)
async def get_employee(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_company_admin)
):
    """
    특정 직원의 상세 정보를 조회합니다.
    
    Args:
        employee_id: 조회할 직원의 ID
        
    Returns:
        직원 상세 정보
    """
    try:
        company_id = current_user.get("company_id")
        if not company_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="회사 정보를 찾을 수 없습니다."
            )
        
        employee = EmployeeManagementService.get_employee_by_id(db, employee_id, company_id)
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="직원을 찾을 수 없습니다."
            )
        
        # 부서명과 직급명 조회
        dept_name = employee.job_dept.dept_name if employee.job_dept else None
        rank_name = employee.job_rank.rank_name if employee.job_rank else None
        
        return EmployeeResponse(
            id=employee.id,
            full_name=employee.full_name or "",
            email=employee.email or "",
            job_dept_id=employee.job_dept_id,
            job_rank_id=employee.job_rank_id,
            dept_name=dept_name,
            rank_name=rank_name,
            company_id=employee.company_id,
            google_user_id=employee.google_user_id or ""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 직원 조회 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="직원 정보를 조회하는 중 오류가 발생했습니다."
        )


@router.put("/{employee_id}", response_model=EmployeeResponse)
async def update_employee(
    employee_id: int,
    update_data: EmployeeUpdateRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_company_admin)
):
    """
    직원 정보를 수정합니다.
    
    Args:
        employee_id: 수정할 직원의 ID
        update_data: 수정할 정보 (부서, 직급)
        
    Returns:
        수정된 직원 정보
    """
    try:
        company_id = current_user.get("company_id")
        if not company_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="회사 정보를 찾을 수 없습니다."
            )
        
        updated_employee = EmployeeManagementService.update_employee(
            db, employee_id, company_id, update_data
        )
        
        if not updated_employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="직원을 찾을 수 없습니다."
            )
        
        # 부서명과 직급명 조회
        dept_name = updated_employee.job_dept.dept_name if updated_employee.job_dept else None
        rank_name = updated_employee.job_rank.rank_name if updated_employee.job_rank else None
        
        return EmployeeResponse(
            id=updated_employee.id,
            full_name=updated_employee.full_name or "",
            email=updated_employee.email or "",
            job_dept_id=updated_employee.job_dept_id,
            job_rank_id=updated_employee.job_rank_id,
            dept_name=dept_name,
            rank_name=rank_name,
            company_id=updated_employee.company_id,
            google_user_id=updated_employee.google_user_id or ""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 직원 정보 수정 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="직원 정보를 수정하는 중 오류가 발생했습니다."
        )


@router.delete("/{employee_id}")
async def delete_employee(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_company_admin)
):
    """
    직원을 삭제합니다.
    
    Args:
        employee_id: 삭제할 직원의 ID
        
    Returns:
        삭제 결과 메시지
    """
    try:
        company_id = current_user.get("company_id")
        if not company_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="회사 정보를 찾을 수 없습니다."
            )
        
        success = EmployeeManagementService.delete_employee(db, employee_id, company_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="직원을 찾을 수 없습니다."
            )
        
        return {"message": "직원이 성공적으로 삭제되었습니다."}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 직원 삭제 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="직원을 삭제하는 중 오류가 발생했습니다."
        )


@router.get("/departments/list", response_model=List[DeptResponse])
async def list_departments(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_company_admin)
):
    """
    모든 부서 목록을 조회합니다.
    
    Returns:
        부서 목록
    """
    try:
        departments = EmployeeManagementService.get_all_departments(db)
        return [DeptResponse(id=dept.id, dept_name=dept.dept_name) for dept in departments]
        
    except Exception as e:
        print(f"❌ 부서 목록 조회 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="부서 목록을 조회하는 중 오류가 발생했습니다."
        )


@router.get("/ranks/list", response_model=List[RankResponse])
async def list_ranks(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_company_admin)
):
    """
    모든 직급 목록을 조회합니다.
    
    Returns:
        직급 목록
    """
    try:
        ranks = EmployeeManagementService.get_all_ranks(db)
        return [RankResponse(id=rank.id, rank_name=rank.rank_name) for rank in ranks]
        
    except Exception as e:
        print(f"❌ 직급 목록 조회 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="직급 목록을 조회하는 중 오류가 발생했습니다."
        )
