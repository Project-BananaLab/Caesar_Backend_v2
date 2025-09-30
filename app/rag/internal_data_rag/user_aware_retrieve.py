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
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "당신은 사내 문서를 기반으로 정확히 답하는 어시스턴트입니다. "
                "주어진 컨텍스트에서만 정보를 추출하여 답변하고, "
                "추측하지 말고 모르는 내용은 '모른다'고 명확히 말하세요. "
                "가능한 한 출처 문서명과 함께 답변하세요.",
            ),
            (
                "user",
                "질문: {question}\n\n"
                "참고 컨텍스트(여러 청크):\n{context}",
            ),
        ])

    def get_user_context(self, user_id: str) -> Optional[dict]:
        """
        user_id(google_user_id)로부터 사용자 정보 조회
        Returns: {"employee_id": int, "company_id": int, "company_code": str} or None
        """
        try:
            db = next(get_db())
            employee = employee_crud.get_employee_by_google_id(db, google_user_id=user_id)
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
        self, 
        query: str, 
        user_id: str, 
        top_k: int = 5
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
            print(f"🔍 권한별 문서 검색: '{query}' (회사: {company_code}, 사용자: {employee_id})")
            
            # 회사별 컬렉션에서 검색
            vectorstore = Chroma(
                collection_name=company_code,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PATH,
            )
            
            # 더 많은 결과를 가져와서 권한 필터링 후 top_k 만큼 반환
            results = vectorstore.similarity_search_with_score(query, k=top_k * 2)
            
            if not results:
                print("❌ 관련 문서를 찾지 못했습니다.")
                return []

            # 권한 필터링
            filtered_contexts = []
            for doc, distance in results:
                meta = dict(doc.metadata or {})
                is_private = meta.get("is_private", False)
                doc_user_id = meta.get("user_id")  # 문서를 업로드한 사용자 ID (employee_id)
                doc_company_id = meta.get("company_id")  # 문서가 속한 회사 ID
                
                # 권한 체크
                # 1. 회사 검증: 문서의 company_id와 현재 사용자의 company_id가 같아야 함
                if doc_company_id != user_context["company_id"]:
                    print(f"🔒 다른 회사 문서 접근 차단: doc_company={doc_company_id}, user_company={user_context['company_id']}")
                    continue
                
                # 2. 개인 문서 검증: is_private=True인 경우 본인만 접근 가능
                if is_private:
                    if doc_user_id != employee_id:
                        print(f"🔒 개인 문서 접근 차단: doc_user_id={doc_user_id}, current_user={employee_id}")
                        continue
                # else: 회사 공개 문서(is_private=False) - 같은 회사면 접근 가능
                
                similarity = _stable_similarity(distance)
                meta["similarity_score"] = similarity
                
                preview = (doc.page_content[:80] + "...") if len(doc.page_content) > 80 else doc.page_content
                print(f"  ✅ [허용] 유사도={similarity:.4f}, private={is_private}, source={meta.get('source')}")
                print(f"          내용: {preview}")
                
                filtered_contexts.append((doc.page_content, meta))
                
                # top_k 개수만큼만 반환
                if len(filtered_contexts) >= top_k:
                    break
            
            print(f"✅ 권한 필터링 후 {len(filtered_contexts)}개 문서 반환")
            return filtered_contexts
            
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
            print(f"⚙️ 답변 생성 중... ({len(contexts)}개 컨텍스트, 모델: {model_label})")

            # 컨텍스트 트렁케이션 (유사도 순)
            context_text = _truncate_context_blocks(contexts, max_chars=MAX_CONTEXT_CHARS)

            # LCEL 체인 실행
            used_llm = self.llm if model is None else ChatOpenAI(model=model, temperature=0)
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
        show_sources: bool = True
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
            query=query, 
            user_id=user_id, 
            top_k=4, 
            show_sources=True
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
    @tool
    def internal_rag_search(query: str) -> str:
        """
        사용자별 권한을 고려한 내부 문서 검색
        - 회사 공개 문서: 같은 회사 직원 모두 접근 가능  
        - 개인 문서: 본인만 접근 가능
        """
        return _user_aware_rag_service.query_rag_with_permission(
            query=query, 
            user_id=user_id,  # 팩토리에서 전달받은 user_id 바인딩
            top_k=4,
            show_sources=True
        )
    
    return [internal_rag_search]


# 기존 범용 도구 (매개변수 2개 - 직접 호출용)
user_aware_rag_tools = [user_aware_rag_search]
