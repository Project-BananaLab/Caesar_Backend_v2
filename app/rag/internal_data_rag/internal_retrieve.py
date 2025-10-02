# app/rag/internal_data_rag/internal_retrieve.py
# -*- coding: utf-8 -*-
# RAG 검색 & 답변 (시트/페이지 메타 표시 보강)
import os
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.documents import Document

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
CHROMA_PATH = os.path.abspath(CHROMA_PATH)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "inside_data1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

# Chroma Cloud 설정
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")

_embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

# Chroma Cloud 연결
import chromadb

# Chroma Cloud 클라이언트 (CloudClient 방식)
chroma_client = chromadb.CloudClient(
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
    api_key=CHROMA_API_KEY,
)

_vectorstore = Chroma(
    client=chroma_client,
    embedding_function=_embeddings,
    collection_name=COLLECTION_NAME,
)
print(f"✅ Chroma Cloud 연결: tenant={CHROMA_TENANT}, database={CHROMA_DATABASE}")

# 로컬 Chroma 연결 (주석 처리)
# _vectorstore = Chroma(
#     collection_name=COLLECTION_NAME,
#     embedding_function=_embeddings,
#     persist_directory=CHROMA_PATH,
# )
# print(f"✅ 로컬 Chroma 연결: {CHROMA_PATH}")
_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "당신은 사내 문서를 기반으로 정확히 답하는 어시스턴트입니다. "
         "주어진 컨텍스트에서만 정보를 추출하여 답변하고, "
         "모르는 내용은 모른다고 말하세요. 가능하다면 출처(파일/시트/페이지)를 함께 표시하세요."),
        ("user", "질문: {question}\n\n참고 컨텍스트(여러 청크):\n{context}"),
    ]
)
_parser = StrOutputParser()

def _stable_similarity(distance: float) -> float:
    try:
        d = float(distance)
    except Exception:
        d = 0.0
    if d < 0:
        d = 0.0
    return 1.0 / (1.0 + d)

def _truncate_context_blocks(blocks: List[Tuple[str, dict]], max_chars: int) -> str:
    sorted_blocks = sorted(blocks, key=lambda x: float(x[1].get("similarity_score", 0.0)), reverse=True)
    acc: List[str] = []
    total = 0
    sep = "\n\n---\n\n"
    for doc, meta in sorted_blocks:
        src = meta.get('source', '알 수 없음')
        sheet = meta.get('sheet')
        page = meta.get('page')
        chunk = meta.get('chunk_idx', 'N/A')
        src_tag = f"[출처: {src}"
        if sheet: src_tag += f" / 시트: {sheet}"
        if page: src_tag += f" / 페이지: {page}"
        src_tag += f" / 청크: {chunk}]"
        block = f"{src_tag}\n{doc}"
        add_len = len(block) + (len(sep) if acc else 0)
        if total + add_len > max_chars:
            break
        if acc:
            acc.append(sep); total += len(sep)
        acc.append(block); total += len(block)
    return "".join(acc)

class RetrieveService:
    def __init__(self):
        self.vectorstore = _vectorstore
        self.llm = _llm
        self.prompt = _prompt
        self.parser = _parser

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, dict]]:
        try:
            print(f"🔍 문서 검색: '{query}' (상위 {top_k}개)")
            results: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(query, k=top_k)
            if not results:
                print("❌ 관련 문서를 찾지 못했습니다.")
                return []
            contexts: List[Tuple[str, dict]] = []
            print(f"✅ {len(results)}개의 관련 문서를 찾았습니다.")
            for i, (doc, distance) in enumerate(results, start=1):
                sim = _stable_similarity(distance)
                meta = dict(doc.metadata or {}); meta["similarity_score"] = sim
                preview = (doc.page_content[:80] + "...") if len(doc.page_content) > 80 else doc.page_content
                print(f"  [Rank {i}] 유사도={sim:.4f}, source={meta.get('source')}, "
                      f"sheet={meta.get('sheet')}, page={meta.get('page')}, chunk={meta.get('chunk_idx')}")
                print(f"          내용: {preview}")
                contexts.append((doc.page_content, meta))
            return contexts
        except Exception as e:
            print(f"❌ 문서 검색 중 오류 발생: {e}")
            return []

    def generate_answer(self, query: str, contexts: List[Tuple[str, dict]], model: str | None = None) -> str:
        if not contexts:
            return "관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요."
        try:
            model_label = model or getattr(self.llm, "model", getattr(self.llm, "model_name", "unknown"))
            print(f"⚙️ 답변 생성 중... ({len(contexts)}개 컨텍스트, 모델: {model_label})")
            context_text = _truncate_context_blocks(contexts, max_chars=MAX_CONTEXT_CHARS)
            used_llm = self.llm if model is None else ChatOpenAI(model=model, temperature=0)
            chain = self.prompt | used_llm | self.parser
            answer: str = chain.invoke({"question": query, "context": context_text})
            print("✅ 답변 생성 완료")
            return answer
        except Exception as e:
            print(f"❌ 답변 생성 중 오류 발생: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {e}"

    def query_rag(self, query: str, top_k: int = 4, model: str | None = None, show_sources: bool = True) -> str:
        print(f"\n🔍 질의: {query}")
        contexts = self.retrieve_documents(query, top_k)
        if not contexts:
            return "죄송합니다. 해당 질문과 관련된 내부 문서를 찾을 수 없습니다."
        answer = self.generate_answer(query, contexts, model)
        if show_sources:
            sources = []
            for _, meta in contexts:
                src = meta.get('source', '알 수 없음')
                sheet = meta.get('sheet')
                page = meta.get('page')
                chunk = meta.get('chunk_idx', 'N/A')
                tag = f"- {src}"
                if sheet: tag += f" / 시트 {sheet}"
                if page: tag += f" / 페이지 {page}"
                tag += f" (청크 {chunk})"
                if tag not in sources:
                    sources.append(tag)
            return f"{answer}\n\n📋 참고한 문서:\n" + "\n".join(sources)
        return answer

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

_retrieve_service = RetrieveService()

@tool
def rag_search_tool(query: str) -> str:
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
    print(f"\n📚 RAG 도구 실행: '{query}'")
    return _retrieve_service.query_rag(query, top_k=3)

rag_tools = [rag_search_tool]

def retrieve_documents(query: str, top_k: int = 3) -> List[Tuple[str, dict]]:
    return RetrieveService().retrieve_documents(query, top_k)

def generate_answer(query: str, contexts: List[Tuple[str, dict]], model: str | None = None) -> str:
    return RetrieveService().generate_answer(query, contexts, model)

def query_rag(query: str, top_k: int = 4, model: str | None = None) -> str:
    return RetrieveService().query_rag(query, top_k, model)

def _healthcheck_vectorstore() -> bool:
    try:
        _ = _vectorstore.similarity_search("__healthcheck__", k=1)
        return True
    except Exception as e:
        print(f"❌ Chroma 헬스체크 실패: {e}")
        return False

def main():
    print("=" * 80)
    print("🔍 문서 검색 및 답변 생성 서비스")
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
