# -*- coding: utf-8 -*-
"""
RAG 문서 검색 & 답변 서비스 (LangChain + Chroma)
- 권장사항 반영 버전

변경 핵심
1) 컨텍스트 길이 제한(MAX_CONTEXT_CHARS) 추가로 프롬프트 초과 방지
2) 안정 유사도 변환: similarity = 1 / (1 + distance) (코사인 거리 가정)
3) 헬스체크 시 private 속성 접근 제거 → 실제 검색 시도로 체크
4) Document 임포트 경로 최신화 (langchain_core.documents)
5) CHROMA_PATH 절대 경로화(로그 표시) + 환경변수 통일
6) 모델 오버라이드 지원(메서드 인자 model)
7) 컨텍스트 트렁케이션 시 유사도 순 정렬 후 누적 바이트 컷
8) 로그 메시지 개선
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv

# LangChain - LLM/Embeddings/VectorStore/Prompt/Parser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.documents import Document  # ✅ 최신 경로

# ─────────────────────────────────────────────────────────
# 환경 변수 로드
# ─────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────
# 환경 변수
# ─────────────────────────────────────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
CHROMA_PATH = os.path.abspath(CHROMA_PATH)  # ✅ 절대 경로화 (로그 가시성)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "inside_data")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")  # ✅ 운영 기본값 권장
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))  # ✅ 프롬프트 길이 제한

# ─────────────────────────────────────────────────────────
# LangChain 컴포넌트 초기화
# ─────────────────────────────────────────────────────────
# 임베딩 함수 (OpenAI Embeddings 래퍼)
_embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

# 벡터스토어 (Chroma 래퍼)
# - persist_directory: CHROMA_PATH
# - collection_name: COLLECTION_NAME
_vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=CHROMA_PATH,
)

# LLM (ChatOpenAI 래퍼)
_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

# 프롬프트 (시스템+사용자)
_prompt = ChatPromptTemplate.from_messages(
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
            "질문: {question}\n\n"
            "참고 컨텍스트(여러 청크):\n{context}",
        ),
    ]
)

# 출력 파서
_parser = StrOutputParser()


# ─────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────

def _stable_similarity(distance: float) -> float:
    """코사인 거리 가정 시 안정적 유사도 변환: 1 / (1 + d)."""
    try:
        d = float(distance)
    except Exception:
        d = 0.0
    if d < 0:
        d = 0.0
    return 1.0 / (1.0 + d)


def _truncate_context_blocks(blocks: List[Tuple[str, dict]], max_chars: int) -> str:
    """컨텍스트 블록을 유사도 순으로 정렬 후, max_chars 까지 누적하여 문자열 구성.
    blocks: [(doc, meta)] with meta["similarity_score"] 존재 가정
    """
    # 유사도 높은 순 정렬
    sorted_blocks = sorted(
        blocks,
        key=lambda x: float(x[1].get("similarity_score", 0.0)),
        reverse=True,
    )

    acc: List[str] = []
    total = 0
    sep = "\n\n---\n\n"

    for doc, meta in sorted_blocks:
        header = f"[출처: {meta.get('source', '알 수 없음')} / 청크: {meta.get('chunk_idx', 'N/A')}]\n"
        block = header + doc
        add_len = len(block) + (len(sep) if acc else 0)
        if total + add_len > max_chars:
            break
        if acc:
            acc.append(sep)
            total += len(sep)
        acc.append(block)
        total += len(block)

    return "".join(acc)


class RetrieveService:
    """문서 검색 및 답변 생성을 담당하는 서비스 (LangChain 버전, 권장사항 반영)"""

    def __init__(self):
        self.vectorstore = _vectorstore
        self.llm = _llm
        self.prompt = _prompt
        self.parser = _parser

    # ========================= 문서 검색 =========================

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, dict]]:
        """
        LangChain의 vectorstore 래퍼로 유사도 검색 수행.
        similarity_search_with_score를 사용해 점수(distance)를 함께 받습니다.
        (Chroma의 score는 보통 'cosine distance'로, 값이 작을수록 유사.)
        """
        try:
            print(f"🔍 문서 검색: '{query}' (상위 {top_k}개)")

            results: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(
                query, k=top_k
            )
            if not results:
                print("❌ 관련 문서를 찾지 못했습니다.")
                return []

            contexts: List[Tuple[str, dict]] = []
            print(f"✅ {len(results)}개의 관련 문서를 찾았습니다.")
            for i, (doc, distance) in enumerate(results, start=1):
                similarity = _stable_similarity(distance)  # ✅ 안정 유사도 변환
                meta = dict(doc.metadata or {})
                meta["similarity_score"] = similarity  # 보고/정렬용
                preview = (doc.page_content[:80] + "...") if len(doc.page_content) > 80 else doc.page_content
                print(
                    f"  [Rank {i}] 유사도={similarity:.4f}, "
                    f"source={meta.get('source')}, chunk={meta.get('chunk_idx')}"
                )
                print(f"          내용: {preview}")
                contexts.append((doc.page_content, meta))

            return contexts
        except Exception as e:
            print(f"❌ 문서 검색 중 오류 발생: {e}")
            return []

    # ========================= 답변 생성 =========================

    def generate_answer(
        self, query: str, contexts: List[Tuple[str, dict]], model: str | None = None
    ) -> str:
        """프롬프트에 컨텍스트를 주입해 LLM 호출(LCEL 체인 사용)."""
        if not contexts:
            return "관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요."

        try:
            model_label = model or getattr(self.llm, "model", getattr(self.llm, "model_name", "unknown"))
            print(f"⚙️ 답변 생성 중... ({len(contexts)}개 컨텍스트, 모델: {model_label})")

            # ✅ 컨텍스트 트렁케이션 (유사도 순)
            context_text = _truncate_context_blocks(contexts, max_chars=MAX_CONTEXT_CHARS)

            # LCEL: prompt → llm → parser
            used_llm = self.llm if model is None else ChatOpenAI(model=model, temperature=0)
            chain = self.prompt | used_llm | self.parser
            answer: str = chain.invoke({"question": query, "context": context_text})

            print("✅ 답변 생성 완료")
            return answer
        except Exception as e:
            print(f"❌ 답변 생성 중 오류 발생: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {e}"

    # ========================= 통합 RAG 처리 =========================

    def query_rag(
        self, query: str, top_k: int = 4, model: str | None = None, show_sources: bool = True
    ) -> str:
        """질의 → 검색 → 생성까지 통합 실행"""
        print(f"\n🔍 질의: {query}")

        contexts = self.retrieve_documents(query, top_k)
        if not contexts:
            return "죄송합니다. 해당 질문과 관련된 내부 문서를 찾을 수 없습니다."

        answer = self.generate_answer(query, contexts, model)

        if show_sources:
            sources = []
            for _, meta in contexts:
                src = f"- {meta.get('source', '알 수 없음')} (청크 {meta.get('chunk_idx', 'N/A')})"
                if src not in sources:
                    sources.append(src)
            return f"{answer}\n\n📋 참고한 문서:\n" + "\n".join(sources)

        return answer

    # ========================= 대화형 인터페이스 =========================

    def interactive_mode(self):
        print("\n🎯 대화형 RAG 검색 시작! (종료: 빈 줄)")
        print("-" * 60)
        while True:
            try:
                q = input("\n> ").strip()
                if not q:
                    print("👋 종료합니다.")
                    break
                print("\n=== 답변 ===")
                print(self.query_rag(q))
            except KeyboardInterrupt:
                print("\n\n👋 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")


# ========================= LangChain 도구 정의 =========================
_retrieve_service = RetrieveService()

@tool
def rag_search_tool(query: str) -> str:
    """
    내부 문서에서 정보를 검색하고 답변을 생성하는 RAG 도구입니다.
    사내 문서, 정책, 절차, 가이드라인 등에 대한 질문에 답변합니다.
    """
    print(f"\n📚 RAG 도구 실행: '{query}'")
    return _retrieve_service.query_rag(query, top_k=3)


rag_tools = [rag_search_tool]


# ========================= 편의 함수들 =========================

def retrieve_documents(query: str, top_k: int = 3) -> List[Tuple[str, dict]]:
    return RetrieveService().retrieve_documents(query, top_k)


def generate_answer(query: str, contexts: List[Tuple[str, dict]], model: str | None = None) -> str:
    return RetrieveService().generate_answer(query, contexts, model)


def query_rag(query: str, top_k: int = 4, model: str | None = None) -> str:
    return RetrieveService().query_rag(query, top_k, model)


# ========================= CLI =========================

def _healthcheck_vectorstore() -> bool:
    """✅ private 속성 접근 없이 헬스체크: 더미 검색 시도"""
    try:
        _ = _vectorstore.similarity_search("__healthcheck__", k=1)
        return True
    except Exception as e:
        print(f"❌ Chroma 헬스체크 실패: {e}")
        return False


def main():
    print("=" * 80)
    print("🔍 문서 검색 및 답변 생성 서비스 (LangChain 권장사항 반영)")
    print("=" * 80)

    print(f"📁 CHROMA_PATH: {CHROMA_PATH}")
    print(f"🗄️ COLLECTION_NAME: {COLLECTION_NAME}")
    print(f"🔤 EMBED_MODEL: {EMBED_MODEL}")
    print(f"🧠 CHAT_MODEL: {CHAT_MODEL}")
    print(f"🧻 MAX_CONTEXT_CHARS: {MAX_CONTEXT_CHARS}")

    if not _healthcheck_vectorstore():
        print("먼저 ingest 파이프라인으로 문서를 적재하세요.")
        return

    RetrieveService().interactive_mode()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🧪 테스트 모드: 간단 질의 3개 실행")
        for q in ["기록물 관리", "관리기준표가 뭐야?", "야간 및 휴일근로 관련 규정 알려줘"]:
            print("\nQ:", q)
            print(query_rag(q))
            print("-" * 60)
    else:
        main()
