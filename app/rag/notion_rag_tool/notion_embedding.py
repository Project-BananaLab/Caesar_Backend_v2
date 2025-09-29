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
    start_page_id: str = None
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
        
        # START_PAGE_ID 동적 설정
        if start_page_id is None:
            update_progress(20, "워크스페이스 루트 페이지들 검색 중...")
            # 모든 루트 페이지를 가져옴
            root_pages = get_text_from_notion.get_workspace_root_pages(notion)
            if root_pages:
                update_progress(22, f"{len(root_pages)}개의 루트 페이지 발견")
                
                # 모든 루트 페이지에서 텍스트 추출
                update_progress(25, "모든 루트 페이지에서 텍스트를 추출 중...")
                all_text_content = ""
                
                for i, page in enumerate(root_pages, 1):
                    try:
                        progress = 25 + (15 * i // len(root_pages))  # 25%~40% 구간에서 진행률 표시
                        update_progress(progress, f"페이지 {i}/{len(root_pages)} 처리 중: {page['title']}")
                        
                        page_content = process_all_content_recursively(page["id"], notion_client=notion)
                        if page_content and page_content.strip():
                            all_text_content += f"\n\n=== 📄 {page['title']} ===\n{page_content}"
                        else:
                            print(f"⚠️ 페이지 '{page['title']}'에서 추출된 내용이 없습니다.")
                            
                    except Exception as e:
                        print(f"⚠️ 페이지 '{page['title']}' 처리 중 오류: {e}")
                        continue
                
                text_content = all_text_content
                update_progress(40, f"모든 페이지 텍스트 추출 완료")
            else:
                # 루트 페이지가 없으면 모든 사용 가능한 페이지 검색
                update_progress(22, "루트 페이지 없음 - 모든 사용 가능한 페이지 검색 중...")
                available_pages = get_text_from_notion.get_available_start_pages(notion)
                if available_pages:
                    start_page_id = available_pages[0]["id"]
                    update_progress(25, f"첫 번째 사용 가능한 페이지 사용: {available_pages[0]['title']}")
                    text_content = process_all_content_recursively(start_page_id, notion_client=notion)
                else:
                    error_msg = "처리할 수 있는 페이지를 찾을 수 없습니다."
                    update_progress(22, error_msg)
                    return {"success": False, "message": error_msg, "error": "no_pages_found"}
        else:
            update_progress(20, f"지정된 시작 페이지 사용: {start_page_id[:8]}...")
            # 지정된 단일 페이지만 처리
            update_progress(25, "지정된 페이지에서 텍스트를 추출 중...")
            text_content = process_all_content_recursively(start_page_id, notion_client=notion)
        
        # 디버깅 정보
        content_length = len(text_content) if text_content else 0
        update_progress(45, f"총 텍스트 추출 완료 (길이: {content_length:,}자)")
        
        # 추출된 텍스트의 앞부분을 로그로 출력 (디버깅용)
        if text_content:
            preview = text_content[:200].replace('\n', ' ')
            print(f"[DEBUG] 추출된 텍스트 미리보기: {preview}...")
            
            # 페이지별 통계 출력
            page_sections = text_content.split("===")
            if len(page_sections) > 1:
                print(f"[INFO] 총 {len(page_sections) - 1}개 페이지 처리됨")

        # 추출된 텍스트 검증
        if not text_content or not text_content.strip():
            error_msg = "추출된 텍스트가 비어있습니다. 워크스페이스에 내용이 있는 페이지가 있는지 확인해주세요."
            update_progress(45, error_msg)
            return {"success": False, "message": error_msg, "error": "empty_content"}

        # 텍스트를 청크로 분할
        update_progress(50, "텍스트를 청크로 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
        )
        texts = text_splitter.split_text(text_content)
        
        # 분할된 텍스트 검증
        if not texts or len(texts) == 0:
            error_msg = "텍스트 분할 결과가 비어있습니다. 페이지 내용을 확인해주세요."
            update_progress(50, error_msg)
            return {"success": False, "message": error_msg, "error": "empty_chunks"}
        
        # 빈 청크 제거
        texts = [text.strip() for text in texts if text.strip()]
        
        if not texts:
            error_msg = "유효한 텍스트 청크가 없습니다. 페이지에 텍스트 내용이 있는지 확인해주세요."
            update_progress(50, error_msg)
            return {"success": False, "message": error_msg, "error": "no_valid_chunks"}
        
        update_progress(60, f"{len(texts)}개의 유효한 청크로 분할 완료")

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

        # 기존 Notion 데이터만 선택적으로 삭제 (다른 문서는 보존)
        try:
            existing_collections = client.list_collections()
            collection_exists = any(col.name == collection_name for col in existing_collections)
            
            if collection_exists:
                collection = client.get_collection(collection_name)
                # Notion 관련 문서만 필터링해서 삭제
                # metadata에 source="notion" 또는 start_page_id가 있는 문서들을 찾아서 삭제
                try:
                    # 컬렉션의 모든 문서 조회
                    results = collection.get()
                    notion_ids = []
                    
                    # Notion 관련 문서 ID 수집
                    if results['metadatas']:
                        for i, metadata in enumerate(results['metadatas']):
                            if (metadata and 
                                (metadata.get('source') == 'notion' or 
                                 'start_page_id' in metadata or
                                 'notion' in str(metadata).lower())):
                                notion_ids.append(results['ids'][i])
                    
                    # Notion 관련 문서만 삭제
                    if notion_ids:
                        collection.delete(ids=notion_ids)
                        update_progress(75, f"기존 Notion 데이터 {len(notion_ids)}개 정리 완료")
                    else:
                        update_progress(75, "삭제할 기존 Notion 데이터 없음")
                        
                except Exception as delete_error:
                    print(f"기존 Notion 데이터 삭제 중 오류 (무시): {delete_error}")
                    update_progress(75, "기존 데이터 정리 건너뛰기")
        except Exception as e:
            print(f"기존 컬렉션 확인 중 오류 (무시): {e}")

        # ChromaDB에 데이터 저장
        update_progress(80, "임베딩 생성 및 벡터 데이터베이스에 저장 중...")
        
        # Notion 문서임을 식별할 수 있는 메타데이터 생성
        metadatas = []
        
        # 처리된 페이지 수 계산
        processed_pages_count = len(text_content.split("=== 📄")) - 1 if "=== 📄" in text_content else 1
        
        for i, text in enumerate(texts):
            metadatas.append({
                "source": "notion",
                "start_page_id": start_page_id if start_page_id else "multiple_root_pages",
                "chunk_index": i,
                "company_id": str(company_id),
                "processed_pages_count": processed_pages_count
            })
        
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name=collection_name,
            client=client
        )
        
        update_progress(100, "임베딩 및 저장이 완료되었습니다!")
        
        return {
            "success": True,
            "message": f"총 {processed_pages_count}개 페이지에서 {len(texts)}개의 텍스트 청크가 성공적으로 처리되었습니다.",
            "error": None,
            "chunks_count": len(texts),
            "processed_pages_count": processed_pages_count,
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