# app/features/login/company/routers.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.utils.db import get_db
from app.utils.crypto_utils import encrypt_data
from app.features.auth.company_auth import get_current_company_admin
from .schemas import CompanyLoginIn, CompanyLoginOut, NotionApiUpdateIn
from .service import login_by_coid
from .models import Company
import threading
import time

# 프론트 요청 경로와 통일
router = APIRouter(prefix="/api/company", tags=["auth-company"])

# 추출 진행률을 저장할 전역 딕셔너리 (실제 운영환경에서는 Redis 등 사용 권장)
extraction_progress = {}
extraction_lock = threading.Lock()

@router.post(
    "/login",
    response_model=CompanyLoginOut,
    response_model_by_alias=True,  # 응답을 alias(camelCase)로 직렬화
)
def company_login(payload: CompanyLoginIn, db: Session = Depends(get_db)):
    try:
        # 요청은 camelCase(coId)로 들어오지만, 코드에선 payload.co_id 로 접근
        return login_by_coid(db, payload.co_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.get("/notion-api")
def get_notion_api(
    db: Session = Depends(get_db),
    current_company: dict = Depends(get_current_company_admin)
):
    """회사 계정의 Notion API 키 존재 여부 확인"""
    try:
        company_id = current_company["company_id"]
        company = db.query(Company).filter(Company.id == company_id).first()
        
        if not company:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="회사 정보를 찾을 수 없습니다.")
        
        has_api_key = company.co_notion_API is not None
        return {"hasApiKey": has_api_key}
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"API 키 확인 실패: {str(e)}")

@router.post("/notion-api")
def update_notion_api(
    payload: NotionApiUpdateIn, 
    db: Session = Depends(get_db),
    current_company: dict = Depends(get_current_company_admin)
):
    """회사 계정의 Notion API 키를 암호화하여 저장"""
    try:
        company_id = current_company["company_id"]
        
        # 회사 정보 조회
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="회사 정보를 찾을 수 없습니다.")
        
        # API 키 암호화
        encrypted_api_key = encrypt_data(payload.notion_api_key)
        
        # 회사 테이블에 암호화된 API 키 저장
        company.co_notion_API = encrypted_api_key
        db.commit()
        
        return {"message": "Notion API 키가 성공적으로 저장되었습니다."}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"API 키 저장 실패: {str(e)}")

@router.delete("/notion-api")
def delete_notion_api(
    db: Session = Depends(get_db),
    current_company: dict = Depends(get_current_company_admin)
):
    """회사 계정의 Notion API 키 삭제"""
    try:
        company_id = current_company["company_id"]
        
        # 회사 정보 조회
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="회사 정보를 찾을 수 없습니다.")
        
        # API 키 삭제
        company.co_notion_API = None
        db.commit()
        
        return {"message": "Notion API 키가 성공적으로 삭제되었습니다."}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"API 키 삭제 실패: {str(e)}")

@router.get("/notion-extract/progress")
def get_extraction_progress(
    current_company: dict = Depends(get_current_company_admin)
):
    """회사의 Notion 데이터 추출 진행률 조회"""
    company_id = current_company["company_id"]
    
    with extraction_lock:
        progress_data = extraction_progress.get(company_id, {
            "status": "idle",  # idle, in_progress, completed, failed
            "progress": 0,
            "message": "대기 중",
            "start_time": None,
            "end_time": None
        })
    
    return progress_data

def update_extraction_progress(company_id: int, progress: int, message: str, status: str = "in_progress"):
    """추출 진행률 업데이트"""
    with extraction_lock:
        if company_id not in extraction_progress:
            extraction_progress[company_id] = {
                "status": "idle",
                "progress": 0,
                "message": "대기 중",
                "start_time": None,
                "end_time": None
            }
        
        extraction_progress[company_id].update({
            "status": status,
            "progress": progress,
            "message": message
        })
        
        if status == "in_progress" and extraction_progress[company_id]["start_time"] is None:
            extraction_progress[company_id]["start_time"] = time.time()
        elif status in ["completed", "failed"]:
            extraction_progress[company_id]["end_time"] = time.time()

def run_notion_extraction(company_id: int):
    """백그라운드에서 실행될 추출 함수"""
    from app.utils.crypto_utils import decrypt_data
    from app.rag.notion_rag_tool.notion_embedding import run_notion_embedding
    
    try:
        # 회사의 암호화된 Notion API 키 가져오기
        from app.utils.db import get_db
        
        db = next(get_db())
        try:
            company = db.query(Company).filter(Company.id == company_id).first()
            if not company or not company.co_notion_API:
                update_extraction_progress(company_id, 0, "Notion API 키가 설정되지 않았습니다.", "failed")
                return
            
            # API 키 복호화
            decrypted_api_key = decrypt_data(company.co_notion_API, return_type="string")
            # 회사 코드 가져오기
            company_code = company.code
            
        finally:
            db.close()
        
        # 진행률 콜백 함수 정의
        def progress_callback(progress: int, message: str):
            update_extraction_progress(company_id, progress, message, "in_progress")
        
        # Notion 임베딩 함수 실행
        result = run_notion_embedding(
            notion_api_key=decrypted_api_key,
            company_id=company_id,
            company_code=company_code,
            progress_callback=progress_callback
        )
        
        if result["success"]:
            update_extraction_progress(company_id, 100, result["message"], "completed")
        else:
            update_extraction_progress(company_id, 0, result["message"], "failed")
            
    except Exception as e:
        update_extraction_progress(company_id, 0, f"추출 실행 중 오류: {str(e)}", "failed")

@router.post("/notion-extract")
def extract_notion_data(
    db: Session = Depends(get_db),
    current_company: dict = Depends(get_current_company_admin)
):
    """회사의 Notion 데이터를 추출하여 임베딩 처리"""
    try:
        company_id = current_company["company_id"]
        
        # 이미 진행 중인 추출이 있는지 확인
        with extraction_lock:
            current_progress = extraction_progress.get(company_id, {})
            if current_progress.get("status") == "in_progress":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT, 
                    detail="이미 추출이 진행 중입니다."
                )
        
        # 회사에 Notion API 키가 있는지 확인
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company or not company.co_notion_API:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Notion API 키가 설정되지 않았습니다.")
        
        # 백그라운드 스레드에서 추출 실행
        extraction_thread = threading.Thread(target=run_notion_extraction, args=(company_id,))
        extraction_thread.daemon = True
        extraction_thread.start()
        
        return {"message": "Notion 데이터 추출이 시작되었습니다."}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"추출 실행 실패: {str(e)}")
