"""
Notion RAG Tool for Agent Core

이 파일은 Notion RAG 기능을 LangChain tool로 변환하여
Agent Core에서 사용할 수 있도록 만든 도구입니다.

Agent Core 연결 방법:
1. 이 파일을 agent_core/utils.py나 적절한 위치에 import
2. tools 리스트에 notion_rag_search 추가
3. create_react_agent에서 tools 파라미터로 전달

사용 예시:
    from tmp.notion_RAG.notion_rag_tool import notion_rag_search

    tools = [notion_rag_search]
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

NOTION_TOKEN = os.getenv("NOTION_TOKEN2")
START_PAGE_ID = (
    "264120560ff680198c0fefbbe17bfc2c"  # 시작 페이지 ID. 나중에 Frontend에서 받아올 것
)


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

    @classmethod
    def create_for_user(cls, user_id: str):
        """사용자 ID(google_user_id)로 NotionRAGService 인스턴스 생성"""
        from app.utils.db import get_db
        from app.features.login.employee_google.models import Employee
        from app.features.login.company.models import Company
        
        db = next(get_db())
        try:
            # user_id로 Employee 조회
            employee = db.query(Employee).filter(Employee.google_user_id == user_id).first()
            if not employee:
                return cls()  # 기본 컬렉션 사용
            
            # Employee의 company_id로 Company 조회
            company = db.query(Company).filter(Company.id == employee.company_id).first()
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

            # 리트리버 생성
            self.retriever = self.vectorstore.as_retriever()

            # gpt-4o-mini 모델 초기화
            self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        except Exception as e:
            print(f"Notion RAG 서비스 초기화 실패: {e}")
            self.rag_chain = None

    def search(self, query: str) -> str:
        """Notion 문서에서 질문에 대한 답변 검색"""

        return self.retriever.invoke(query)
        # if self.rag_chain is None:
        #     return "Notion RAG 시스템이 초기화되지 않았습니다. 환경 변수를 확인해주세요."

        # try:
        #     return self.rag_chain.invoke(query)
        #     # return self.retriever.invoke(query)
        # except Exception as e:
        #     return f"검색 중 오류가 발생했습니다: {str(e)}"


def create_notion_rag_tool_for_user(user_id: str):
    """사용자별 Notion RAG 도구 생성 함수"""
    
    # 사용자별 NotionRAGService 인스턴스 생성
    service = NotionRAGService.create_for_user(user_id)
    
    @tool
    def notion_rag_search(query: str) -> str:
        """
        Notion 문서에서 정보를 검색하고 질문에 답변합니다.

        이 도구는 사전에 임베딩된 Notion 문서들을 검색하여
        사용자의 질문과 관련된 정보를 찾아 답변을 생성합니다.

        Args:
            query (str): 검색하고자 하는 질문이나 키워드

        Returns:
            str: Notion 문서를 기반으로 한 답변
        """
        return service.search(query)
    
    return notion_rag_search

def create_notion_rag_tool(company_id: int):
    """회사별 Notion RAG 도구 생성 함수 (하위 호환성)"""
    
    # 회사별 NotionRAGService 인스턴스 생성
    service = NotionRAGService.create_for_company(company_id)
    
    @tool
    def notion_rag_search(query: str) -> str:
        """
        Notion 문서에서 정보를 검색하고 질문에 답변합니다.

        이 도구는 사전에 임베딩된 Notion 문서들을 검색하여
        사용자의 질문과 관련된 정보를 찾아 답변을 생성합니다.

        Args:
            query (str): 검색하고자 하는 질문이나 키워드

        Returns:
            str: Notion 문서를 기반으로 한 답변
        """
        return service.search(query)
    
    return notion_rag_search


# 기본 전역 서비스 (하위 호환성을 위해 유지)
_notion_rag_service = NotionRAGService()

@tool  
def notion_rag_search(query: str) -> str:
    """
    기본 Notion RAG 검색 (하위 호환성)
    
    Args:
        query (str): 검색하고자 하는 질문이나 키워드

    Returns:
        str: Notion 문서를 기반으로 한 답변
    """
    return _notion_rag_service.search(query)


# # Agent Core에서 사용할 수 있도록 tools 리스트로 제공
# tools = [notion_rag_search]

# agent = create_react_agent(model, tools=tools) # Agent Core에서 사용할 수 있도록 제공
