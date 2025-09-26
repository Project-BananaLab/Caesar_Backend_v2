import os
from dotenv import load_dotenv
from notion_client import Client
from .get_text_from_notion import process_all_content_recursively
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from typing import Callable, Optional

# .env 파일에서 환경 변수 로드
load_dotenv()

def run_notion_embedding(
    notion_api_key: str, 
    company_id: int,
    company_code: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    start_page_id: str = '264120560ff680198c0fefbbe17bfc2c'
) -> dict:
    """
    Notion 데이터를 추출하고 임베딩하여 벡터 데이터베이스에 저장하는 함수
    
    Args:
        notion_api_key (str): Notion API 키
        company_id (int): 회사 ID
        company_code (str): 회사 코드 (컬렉션명에 사용)
        progress_callback (Callable): 진행률 업데이트 콜백 함수 (progress: int, message: str)
        start_page_id (str): 시작 페이지 ID
        
    Returns:
        dict: 실행 결과 (success: bool, message: str, error: str)
    """
    try:
        # 진행률 업데이트 함수
        def update_progress(progress: int, message: str):
            if progress_callback:
                progress_callback(progress, message)
            print(f"[{progress}%] {message}")

        # 환경 변수 확인
        required_env_vars = ["OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            error_msg = f"필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}"
            update_progress(0, error_msg)
            return {"success": False, "message": error_msg, "error": "missing_env_vars"}

        update_progress(5, "환경 설정 확인 완료")

        # Notion 클라이언트 초기화
        update_progress(10, "Notion API 연결 중...")
        notion = Client(auth=notion_api_key)
        update_progress(15, "Notion API 연결 완료")

        # get_text_from_notion 모듈의 전역 notion 클라이언트 업데이트
        from . import get_text_from_notion
        get_text_from_notion.notion = notion
        
        # Notion 페이지에서 텍스트 추출
        update_progress(25, "Notion 페이지에서 텍스트를 추출 중...")
        text_content = process_all_content_recursively(start_page_id)
        update_progress(45, "텍스트 추출 완료")

        # 텍스트를 청크로 분할
        update_progress(50, "텍스트를 청크로 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
        )
        texts = text_splitter.split_text(text_content)
        update_progress(60, f"{len(texts)}개의 청크로 분할 완료")

        # 임베딩 모델 초기화
        update_progress(65, "임베딩 모델 초기화 중...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # ChromaDB 컬렉션 이름 설정 (회사별로 구분)
        # company_code가 제공되면 code 사용, 없으면 company_id 사용 (하위 호환성)
        collection_name = f"{company_code}"

        # Chroma Cloud 클라이언트 초기화
        update_progress(70, "ChromaDB 연결 중...")
        client = chromadb.CloudClient(
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
            api_key=os.getenv("CHROMA_API_KEY")
        )

        # 기존 컬렉션이 있다면 삭제 (업데이트를 위해)
        try:
            existing_collections = client.list_collections()
            for collection in existing_collections:
                if collection.name == collection_name:
                    client.delete_collection(collection_name)
                    update_progress(75, "기존 데이터 정리 완료")
                    break
        except Exception as e:
            print(f"기존 컬렉션 확인/삭제 중 오류 (무시): {e}")

        # ChromaDB에 데이터 저장
        update_progress(80, "임베딩 생성 및 벡터 데이터베이스에 저장 중...")
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            collection_name=collection_name,
            client=client
        )
        
        update_progress(100, "임베딩 및 저장이 완료되었습니다!")
        
        return {
            "success": True,
            "message": f"총 {len(texts)}개의 텍스트 청크가 성공적으로 처리되었습니다.",
            "error": None,
            "chunks_count": len(texts),
            "collection_name": collection_name
        }
        
    except Exception as e:
        error_msg = f"임베딩 처리 중 오류 발생: {str(e)}"
        if progress_callback:
            progress_callback(0, error_msg)
        return {
            "success": False,
            "message": error_msg,
            "error": str(e)
        }

# 스크립트로 직접 실행될 때의 처리 (기존 호환성 유지)
if __name__ == "__main__":
    # 환경 변수에서 회사 ID와 토큰을 가져와서 실행
    company_id = int(os.getenv("CURRENT_COMPANY_ID", "1"))
    notion_token = os.getenv("NOTION_TOKEN2")
    
    if not notion_token:
        print("오류: NOTION_TOKEN2 환경 변수가 설정되지 않았습니다.")
        exit(1)
    
    result = run_notion_embedding(notion_token, company_id)
    
    if result["success"]:
        print(f"성공: {result['message']}")
    else:
        print(f"실패: {result['message']}")
        exit(1)