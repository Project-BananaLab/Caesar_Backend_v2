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

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NotionRAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.collection_name = "notion-collection"
            self.embeddings = None
            self.vectorstore = None
            self.retriever = None
            self.llm = None
            self.rag_chain = None
            self._setup()
            NotionRAGService._initialized = True

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


# 전역 서비스 인스턴스
_notion_rag_service = NotionRAGService()


@tool
def notion_rag_search(query: str) -> str:
    """
    Notion 문서에서 정보를 검색하고 질문에 답변합니다.

    이 도구는 사전에 임베딩된 Notion 문서들을 검색하여
    사용자의 질문과 관련된 정보를 찾아 gpt-4o-mini모델을 통해 답변을 생성합니다.

    Args:
        query (str): 검색하고자 하는 질문이나 키워드

    Returns:
        str: Notion 문서를 기반으로 한 답변
    """
    return _notion_rag_service.search(query)


# # Agent Core에서 사용할 수 있도록 tools 리스트로 제공
# tools = [notion_rag_search]

# agent = create_react_agent(model, tools=tools) # Agent Core에서 사용할 수 있도록 제공
