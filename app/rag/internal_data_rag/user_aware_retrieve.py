# app/rag/internal_data_rag/user_aware_retrieve.py
# -*- coding: utf-8 -*-
"""
사용자별 권한을 고려한 RAG 문서 검색 서비스
- 회사 공개 문서: 같은 회사 직원 모두 접근 가능
- 개인 문서: 본인만 접근 가능
- 다른 직원의 개인 문서: 접근 불가
"""

import os
from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.documents import Document

from app.utils.db import get_db
from app.features.login.employee_google import crud as employee_crud
from app.features.admin.services.file_ingest_service import _get_company_code
from app.rag.internal_data_rag.internal_retrieve import (
    _stable_similarity,
    _truncate_context_blocks,
    MAX_CONTEXT_CHARS,
    CHROMA_PATH,
    EMBED_MODEL,
    CHAT_MODEL,
)


class UserAwareRAGService:
    """사용자별 권한을 고려한 RAG 검색 서비스"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        self.parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 사내 문서를 기반으로 정확히 답하는 어시스턴트입니다. "
                    "주어진 컨텍스트에서만 정보를 추출하여 답변하고, "
                    "추측하지 말고 모르는 내용은 '모른다'고 명확히 말하세요. "
                    "가능한 한 출처 문서명과 함께 답변하세요.",
                ),
                (
                    "user",
                    "질문: {question}\n\n" "참고 컨텍스트(여러 청크):\n{context}",
                ),
            ]
        )

    def get_user_context(self, user_id: str) -> Optional[dict]:
        """
        user_id(google_user_id)로부터 사용자 정보 조회
        Returns: {"employee_id": int, "company_id": int, "company_code": str} or None
        """
        try:
            db = next(get_db())
            employee = employee_crud.get_employee_by_google_id(
                db, google_user_id=user_id
            )
            if not employee:
                print(f"❌ 사용자를 찾을 수 없습니다: {user_id}")
                return None

            company_code = _get_company_code(db, employee.company_id)
            return {
                "employee_id": employee.id,
                "company_id": employee.company_id,
                "company_code": company_code,
            }
        except Exception as e:
            print(f"❌ 사용자 컨텍스트 조회 실패: {e}")
            return None
        finally:
            db.close()

    def retrieve_documents_with_permission(
        self, query: str, user_id: str, top_k: int = 5
    ) -> List[Tuple[str, dict]]:
        """
        사용자별 권한을 고려한 문서 검색
        - 회사 공개 문서(is_private=False): 같은 회사 직원 모두 접근 가능
        - 개인 문서(is_private=True): 본인(user_id)만 접근 가능
        """
        user_context = self.get_user_context(user_id)
        if not user_context:
            print(f"❌ 사용자 정보 없음 - 검색 중단: {user_id}")
            return []

        company_code = user_context["company_code"]
        employee_id = user_context["employee_id"]

        try:
            print(
                f"🔍 권한별 문서 검색: '{query}' (회사: {company_code}, 사용자: {employee_id})"
            )

            # 회사별 컬렉션에서 검색 (Chroma Cloud 사용)
            import chromadb

            # Chroma Cloud 클라이언트 초기화
            client = chromadb.CloudClient(
                tenant=os.getenv("CHROMA_TENANT"),
                database=os.getenv("CHROMA_DATABASE"),
                api_key=os.getenv("CHROMA_API_KEY"),
            )

            vectorstore = Chroma(
                client=client,
                collection_name=company_code,
                embedding_function=self.embeddings,
            )

            # 메타데이터 필터로 권한 있는 문서만 검색
            # Chroma 필터 문법: {"$and": [조건1, 조건2]} 또는 {"$or": [조건1, 조건2]}

            # 1. 회사 공개 문서 검색 (company_id 일치 AND is_private=False)
            company_filter = {
                "$and": [
                    {"company_id": {"$eq": user_context["company_id"]}},
                    {"is_private": {"$eq": False}},
                ]
            }

            # 2. 개인 문서 검색 (company_id 일치 AND is_private=True AND user_id 일치)
            personal_filter = {
                "$and": [
                    {"company_id": {"$eq": user_context["company_id"]}},
                    {"is_private": {"$eq": True}},
                    {"user_id": {"$eq": employee_id}},
                ]
            }

            print(f"🔍 회사 공개 문서 검색 필터: {company_filter}")
            print(f"🔍 개인 문서 검색 필터: {personal_filter}")

            # 회사 공개 문서 검색
            company_results = vectorstore.similarity_search_with_score(
                query, k=top_k, filter=company_filter
            )
            print(f"📊 회사 공개 문서: {len(company_results)}개 발견")

            # 개인 문서 검색
            personal_results = vectorstore.similarity_search_with_score(
                query, k=top_k, filter=personal_filter
            )
            print(f"📊 개인 문서: {len(personal_results)}개 발견")

            # 두 결과 합치고 유사도 순으로 정렬
            all_results = company_results + personal_results
            all_results.sort(key=lambda x: x[1])  # distance 기준 오름차순 정렬

            # top_k 개수만큼만 선택
            results = all_results[:top_k]

            if not results:
                print("❌ 관련 문서를 찾지 못했습니다.")
                return []

            # 메타데이터 필터로 이미 권한 검증된 결과 처리
            contexts = []
            print(f"✅ 권한 필터링된 {len(results)}개의 관련 문서를 찾았습니다.")

            for i, (doc, distance) in enumerate(results, start=1):
                similarity = _stable_similarity(distance)
                meta = dict(doc.metadata or {})
                meta["similarity_score"] = similarity

                is_private = meta.get("is_private", False)
                doc_type = "개인 문서" if is_private else "회사 문서"
                preview = (
                    (doc.page_content[:80] + "...")
                    if len(doc.page_content) > 80
                    else doc.page_content
                )

                print(
                    f"  [Rank {i}] 유사도={similarity:.4f}, {doc_type}, source={meta.get('source')}"
                )
                print(f"          내용: {preview}")

                contexts.append((doc.page_content, meta))

            return contexts

        except Exception as e:
            print(f"❌ 권한별 문서 검색 중 오류 발생: {e}")
            return []

    def generate_answer(
        self, query: str, contexts: List[Tuple[str, dict]], model: str = None
    ) -> str:
        """프롬프트에 컨텍스트를 주입해 LLM 호출"""
        if not contexts:
            return "관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요."

        try:
            model_label = model or CHAT_MODEL
            print(
                f"⚙️ 답변 생성 중... ({len(contexts)}개 컨텍스트, 모델: {model_label})"
            )

            # 컨텍스트 트렁케이션 (유사도 순)
            context_text = _truncate_context_blocks(
                contexts, max_chars=MAX_CONTEXT_CHARS
            )

            # LCEL 체인 실행
            used_llm = (
                self.llm if model is None else ChatOpenAI(model=model, temperature=0)
            )
            chain = self.prompt | used_llm | self.parser
            answer = chain.invoke({"question": query, "context": context_text})

            print("✅ 답변 생성 완료")
            return answer
        except Exception as e:
            print(f"❌ 답변 생성 중 오류 발생: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {e}"

    def query_rag_with_permission(
        self,
        query: str,
        user_id: str,
        top_k: int = 4,
        model: str = None,
        show_sources: bool = True,
    ) -> str:
        """사용자별 권한을 고려한 RAG 질의 처리"""
        print(f"\n🔍 권한별 RAG 질의: {query} (사용자: {user_id})")

        contexts = self.retrieve_documents_with_permission(query, user_id, top_k)
        if not contexts:
            return "죄송합니다. 접근 가능한 관련 문서를 찾을 수 없습니다."

        answer = self.generate_answer(query, contexts, model)

        if show_sources:
            sources = []
            for _, meta in contexts:
                is_private = meta.get("is_private", False)
                doc_type = "개인 문서" if is_private else "회사 문서"
                src = f"- {meta.get('source', '알 수 없음')} ({doc_type}, 청크 {meta.get('chunk_idx', 'N/A')})"
                if src not in sources:
                    sources.append(src)
            return f"{answer}\n\n📋 참고한 문서:\n" + "\n".join(sources)

        return answer


# 전역 서비스 인스턴스
_user_aware_rag_service = UserAwareRAGService()


# ========================= LangChain 도구 정의 =========================


@tool
def user_aware_rag_search(query: str, user_id: str) -> str:
    """
    사용자별 권한을 고려한 내부 문서 검색 도구
    - 회사 공개 문서: 같은 회사 직원 모두 접근 가능
    - 개인 문서: 본인만 접근 가능

    Args:
        query (str): 검색할 질문
        user_id (str): 사용자 ID (Google User ID)

    Returns:
        str: 검색 결과 및 답변
    """
    try:
        return _user_aware_rag_service.query_rag_with_permission(
            query=query, user_id=user_id, top_k=4, show_sources=True
        )
    except Exception as e:
        print(f"❌ 사용자별 RAG 검색 실패: {e}")
        return f"문서 검색 중 오류가 발생했습니다: {e}"


# ========================= 사용자별 도구 팩토리 =========================


def create_user_aware_rag_tools(user_id: str) -> list:
    """
    특정 사용자를 위한 RAG 도구 생성 (user_id 바인딩)

    Args:
        user_id (str): 사용자 ID (Google User ID)

    Returns:
        list: 사용자별로 바인딩된 RAG 도구 목록
    """
    print(f"🔧 사용자별 RAG 도구 생성 중: {user_id}")

    @tool
    def internal_rag_search(query: str) -> str:
        """

      


        Search and answer questions from uploaded files (PDF, DOCX, XLSX, etc.).
        
        🎯 This tool ONLY searches content from uploaded files:
        - PDF documents (regulations, manuals, reports, etc.)
        - Word documents (policies, guidelines, contracts, etc.)  
        - Excel files (data, forms, status reports, etc.)
        - Other uploaded document files
        
        Search scope:
        - Company public documents: Company policies, manuals, announcements, forms, etc.
        - Personal uploaded documents: Files personally uploaded by the user
        
        Use when:
        - File-based questions like "production status writing methods", "employee regulations", "manuals"
        - Specific content or procedure inquiries from uploaded documents
        - Questions about forms, templates, formats
        
        Args:
            query (str): Question or keyword to search in uploaded documents
            

        Returns:
            str: Answer based on uploaded documents
        """
        print(f"🔍 internal_rag_search 호출됨: query='{query}', user_id='{user_id}'")
        try:
            result = _user_aware_rag_service.query_rag_with_permission(
                query=query,
                user_id=user_id,  # 팩토리에서 전달받은 user_id 바인딩
                top_k=4,
                show_sources=True,
            )
            return result
        except Exception as e:
            print(f"❌ RAG 검색 실패: {e}")
            return f"내부 문서 검색 중 오류가 발생했습니다: {e}"

    print(f"✅ RAG 도구 생성 완료: {user_id}")
    return [internal_rag_search]


# 기존 범용 도구 (매개변수 2개 - 직접 호출용)
user_aware_rag_tools = [user_aware_rag_search]
