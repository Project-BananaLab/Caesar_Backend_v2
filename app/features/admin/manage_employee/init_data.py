# init_data.py
# 부서와 직급 초기 데이터를 데이터베이스에 삽입하는 스크립트입니다.

from sqlalchemy.orm import Session
from app.utils.db import SessionLocal
from app.features.login.employee_google.models import JobDept, JobRank


def init_departments_and_ranks():
    """
    부서와 직급 초기 데이터를 데이터베이스에 삽입합니다.
    이미 존재하는 데이터는 건너뜁니다.
    """
    db = SessionLocal()
    try:
        # 부서 데이터
        departments = [
            "경영지원", "인사", "재무회계", "법무", "총무", "영업",
            "마케팅", "제품기획", "개발(백엔드)", "개발(프론트엔드)", 
            "데이터", "인프라", "품질(QA)", "고객지원(CS)", "디자인", "운영"
        ]
        
        # 직급 데이터
        ranks = [
            "사원", "주임", "대리", "과장", "차장", "부장",
            "이사", "상무", "전무", "부사장", "사장", "대표이사"
        ]
        
        # 부서 데이터 삽입
        print("🏢 부서 데이터 초기화 중...")
        for dept_name in departments:
            existing_dept = db.query(JobDept).filter(JobDept.dept_name == dept_name).first()
            if not existing_dept:
                new_dept = JobDept(dept_name=dept_name)
                db.add(new_dept)
                print(f"  ✅ 부서 추가: {dept_name}")
            else:
                print(f"  ⚠️ 부서 이미 존재: {dept_name}")
        
        # 직급 데이터 삽입
        print("\n👔 직급 데이터 초기화 중...")
        for rank_name in ranks:
            existing_rank = db.query(JobRank).filter(JobRank.rank_name == rank_name).first()
            if not existing_rank:
                new_rank = JobRank(rank_name=rank_name)
                db.add(new_rank)
                print(f"  ✅ 직급 추가: {rank_name}")
            else:
                print(f"  ⚠️ 직급 이미 존재: {rank_name}")
        
        # 변경사항 커밋
        db.commit()
        print("\n✅ 부서 및 직급 초기 데이터 설정 완료!")
        
    except Exception as e:
        print(f"❌ 초기 데이터 설정 실패: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("부서 및 직급 초기 데이터를 설정합니다...")
    init_departments_and_ranks()
