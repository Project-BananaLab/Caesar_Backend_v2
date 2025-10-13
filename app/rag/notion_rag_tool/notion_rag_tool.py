"""
Notion RAG Tool for Agent Core

이 파일은 Notion RAG 기능을 LangChain tool로 변환하여
Agent Core에서 사용할 수 있도록 만든 도구입니다.

Agent Core 연결 방법:
1. user_id를 통해 company_id를 조회: get_company_id_by_user_id(user_id)
2. company_id로 도구 생성: create_notion_rag_tool(company_id)
3. create_react_agent에서 tools 파라미터로 전달

사용 예시:
    from app.rag.notion_rag_tool.notion_rag_tool import get_company_id_by_user_id, create_notion_rag_tool

    company_id = get_company_id_by_user_id(user_id)
    if company_id:
        notion_tool = create_notion_rag_tool(company_id)
        tools = [notion_tool]
        graph = create_react_agent(model, tools=tools)
"""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import chromadb

# .env 파일에서 환경 변수 로드
load_dotenv()


class NotionRAGService:
    """Notion RAG 서비스 클래스"""

    def __init__(self, company_code: str = None):
        # company_code가 제공되면 사용, 없으면 기본값
        self.collection_name = company_code if company_code else "notion-collection"
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None
        self._setup()

    @classmethod
    def create_for_company(cls, company_id: int):
        """회사 ID로 NotionRAGService 인스턴스 생성"""
        from app.utils.db import get_db
        from app.features.login.company.models import Company

        db = next(get_db())
        try:
            company = db.query(Company).filter(Company.id == company_id).first()
            company_code = company.code if company else None
            return cls(company_code=company_code)
        finally:
            db.close()

    def _setup(self):
        """RAG 시스템 설정"""
        try:
            # 임베딩 모델 초기화
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

            # Chroma Cloud 클라이언트 초기화
            client = chromadb.CloudClient(
                tenant=os.getenv("CHROMA_TENANT"),
                database=os.getenv("CHROMA_DATABASE"),
                api_key=os.getenv("CHROMA_API_KEY"),
            )

            # ChromaDB에서 벡터 저장소 로드
            self.vectorstore = Chroma(
                client=client,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )

            # 리트리버 생성 (source="notion" 메타데이터 필터 적용)
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"filter": {"source": "notion"}}
            )

            # gpt-4o-mini 모델 초기화
            self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        except Exception as e:
            print(f"Notion RAG 서비스 초기화 실패: {e}")
            self.rag_chain = None

    def search(self, query: str) -> str:
        """Notion 문서에서 질문에 대한 답변 검색"""

        if self.vectorstore is None:
            return (
                "Notion RAG 시스템이 초기화되지 않았습니다. 환경 변수를 확인해주세요."
            )

        try:
            # vectorstore에서 직접 검색하여 source="notion" 필터 적용
            documents = self.vectorstore.similarity_search(
                query=query,
                k=5,  # 상위 5개 문서 검색
                filter={"source": "notion"},  # notion 소스만 필터링
            )

            # 검색 결과가 없는 경우
            if not documents:
                return "관련된 Notion 문서를 찾을 수 없습니다."

            # 검색된 문서들의 내용을 결합하여 문자열로 반환
            result_content = []
            print(f"✅ {len(documents)}개의 Notion 문서를 찾았습니다.")

            for i, doc in enumerate(documents):  # 모든 검색된 문서 사용
                content = doc.page_content.strip()
                metadata = doc.metadata or {}

                # 메타데이터 정보 로그 출력 (디버깅용)
                print(
                    f"  [문서 {i+1}] source: {metadata.get('source', 'unknown')}, "
                    f"company_id: {metadata.get('company_id', 'unknown')}"
                )

                if content:
                    # 메타데이터에서 페이지 정보 추출
                    page_title = metadata.get("page_title", "Notion 문서")
                    chunk_index = metadata.get("chunk_index", 0)
                    page_id = metadata.get("start_page_id", "")

                    # 페이지 ID가 있으면 링크 생성
                    notion_url = ""
                    if (
                        page_id
                        and page_id != "multiple_root_pages"
                        and not page_id.startswith("page_")
                    ):
                        # 실제 노션 페이지 ID인 경우 (UUID 형태)
                        if len(page_id) >= 32:  # UUID 길이 체크
                            page_id_clean = page_id.replace("-", "")
                            notion_url = f"https://www.notion.so/{page_id_clean}"
                        else:
                            notion_url = "https://www.notion.so"
                    elif page_id.startswith("page_"):
                        # 페이지 제목 기반 ID인 경우, 기본 노션 워크스페이스 링크 사용
                        notion_url = "https://www.notion.so"
                    elif page_id == "multiple_root_pages":
                        # multiple_root_pages인 경우, 기본 노션 워크스페이스 링크 사용
                        notion_url = "https://www.notion.so"

                    # 검색 결과에 상세한 페이지 정보 포함
                    if notion_url:
                        result_content.append(
                            f"[문서 {i+1}] 📄 {page_title}\n"
                            f"📍 위치: 청크 {chunk_index + 1}\n"
                            f"🔗 링크: [{page_title} 페이지 링크]({notion_url})\n"
                            f"📋 관련 내용:\n{content}\n"
                            f"💡 더 자세한 내용은 위 링크에서 확인하세요."
                        )
                    else:
                        result_content.append(
                            f"[문서 {i+1}] 📄 {page_title}\n"
                            f"📍 위치: 청크 {chunk_index + 1}\n"
                            f"📋 관련 내용:\n{content}"
                        )

            if result_content:
                return "\n\n".join(result_content)
            else:
                return "검색된 문서에서 유효한 내용을 찾을 수 없습니다."

        except Exception as e:
            return f"검색 중 오류가 발생했습니다: {str(e)}"


def get_company_id_by_user_id(user_id: str) -> int:
    """사용자 ID로 회사 ID 조회"""
    from app.utils.db import get_db
    from app.features.login.employee_google.models import Employee

    db = next(get_db())
    try:
        # user_id(google_user_id)로 Employee 조회
        employee = db.query(Employee).filter(Employee.google_user_id == user_id).first()
        return employee.company_id if employee else None
    finally:
        db.close()


def create_notion_rag_tool(company_id: int):
    """회사별 Notion RAG 도구 생성 함수"""

    # 회사별 NotionRAGService 인스턴스 생성
    service = NotionRAGService.create_for_company(company_id)

    @tool
    def notion_rag_search(query: str) -> str:
        """
        Search information from Notion workspace pages and databases.

        ⚠️ This tool ONLY searches content from Notion platform pages, databases, and blocks.

        Use when:
        - Questions about Notion page content
        - Notion database information queries
        - Searching documents or notes written in Notion

        Args:
            query (str): Question or keyword to search in Notion

        Returns:
            str: Answer based on Notion workspace documents
        """
        return service.search(query)

    return notion_rag_search


# 기본 전역 서비스 (하위 호환성을 위해 유지)
_notion_rag_service = NotionRAGService()


@tool
def notion_rag_search(query: str) -> str:
    """
    Default Notion RAG search (for backward compatibility)

    Args:
        query (str): Question or keyword to search

    Returns:
        str: Answer based on Notion documents
    """
    return _notion_rag_service.search(query)


# # Agent Core에서 사용할 수 있도록 tools 리스트로 제공
# tools = [notion_rag_search]

# agent = create_react_agent(model, tools=tools) # Agent Core에서 사용할 수 있도록 제공
