# app/rag/internal_data_rag/user_aware_retrieve.py
# -*- coding: utf-8 -*-
"""
사용자별 권한을 고려한 RAG 문서 검색 서비스
- 회사 공개 문서: 같은 회사 직원 모두 접근 가능
- 개인 문서: 본인만 접근 가능
- 다른 직원의 개인 문서: 접근 불가

- TF-IDF 희소 검색 + 권한 사후 필터 + RRF 결합
- 희소 인덱스가 로컬에 없으면 S3에서 자동 다운로드 후 캐시
- 질의 확장(동의어/플랫폼 정규화 + PRF) → dense/sparse에 동일 적용

추가:
- 유사도 임계값 컷 + 문서/유형 쿼터링
- Cross-encoder Rerank
- LLM Multi-Query
- 출처/페이지 dedup + 문장 중복 제거
"""

import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# 희소 검색(있으면 사용)
try:
    from sklearn.metrics.pairwise import cosine_similarity
    import joblib
except Exception:
    cosine_similarity = None
    joblib = None

# Cross-encoder (선택)
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

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

# ─────────────────────────────────────────
# 환경/옵션 (internal_retrieve.py와 동일하게)
# ─────────────────────────────────────────
SPARSE_INDEX_PATH = os.getenv("SPARSE_INDEX_PATH", os.path.join(CHROMA_PATH, "sparse"))

# 임계/쿼터/리랭크 (internal_retrieve.py와 동일)
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.25"))
DOC_QUOTA = int(os.getenv("DOC_QUOTA", "15"))
KIND_QUOTA = os.getenv("KIND_QUOTA", "text:12,image_ocr:8,image_vlm:4")
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MultiBERT-L-12")
# CE 입력 최적화
CE_MAX_CHARS = int(os.getenv("CE_MAX_CHARS", "1800"))
CE_BATCH_SIZE = int(os.getenv("CE_BATCH_SIZE", "16"))

# LLM Multi-Query (보수화)
ENABLE_LLM_MULTIQUERY = os.getenv("ENABLE_LLM_MULTIQUERY", "true").lower() == "true"
N_QUERY_REWRITES = int(os.getenv("N_QUERY_REWRITES", "1"))  # 기본 1로 완화
SHORT_QUERY_CHAR = int(os.getenv("SHORT_QUERY_CHAR", "40"))

# 디듑 파라미터
MAX_PER_PAGE = int(os.getenv("MAX_PER_PAGE", "2"))  # 페이지당 최대 유지 개수

# S3(옵션) — 희소 인덱스 동기화 (이름 통일)
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION")
AWS_S3_PREFIX = os.getenv("AWS_S3_PREFIX", "rag/sparse")

# 질의 동의어 사전(외부 파일 경로 - 선택)
QUERY_SYNONYM_PATH = os.getenv("QUERY_SYNONYM_PATH")  # e.g. ./config/synonyms.json

# ─────────────────────────────────────────
# NEW: 질의 확장 유틸(동의어/정규화 + PRF)
# ─────────────────────────────────────────
def _load_synonym_dict() -> dict:
    try:
        if QUERY_SYNONYM_PATH and Path(QUERY_SYNONYM_PATH).exists():
            with open(QUERY_SYNONYM_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "유튜브": ["인터넷 개인방송", "동영상 공유 서비스", "개인방송", "영상 업로드"],
        "브이로그": ["인터넷 개인방송", "개인방송", "영상 제작", "동영상 업로드"],
        "틱톡": ["인터넷 개인방송", "동영상 공유 서비스", "소셜미디어"],
        "인스타": ["소셜미디어", "SNS", "콘텐츠 업로드"],
        "라이브": ["실시간 방송", "라이브 스트리밍", "인터넷 개인방송"],
        "수익": ["광고수익", "수익 창출", "협찬", "간접광고", "PPL"],
        "협찬": ["간접광고", "PPL", "대가성 제공"],
        "겸업": ["겸업허가", "겸업의 허가", "부업"],
        "보안": ["직무상 비밀", "영업비밀", "정보보안"],
        "명예훼손": ["명예 침해", "권리 침해"],
        # 인사/경력 관련 동의어 추가
        "이력사": ["이력서", "경력", "경험", "커리어", "프로필", "개인정보"],
        "이력서": ["이력사", "경력서", "경력", "커리어", "프로필", "개인정보"],
        "경력": ["이력서", "이력사", "경험", "커리어", "프로필"],
        "경험": ["이력서", "이력사", "경력", "커리어"],
    }

_SYNONYM_DICT = _load_synonym_dict()
_NORMALIZE_MAP = {
    "youtube": "유튜브",
    "yt": "유튜브",
    "vlog": "브이로그",
    "shorts": "쇼츠",
    "tiktok": "틱톡",
    "instagram": "인스타",
}

def _normalize_terms(q: str) -> str:
    low = q.lower()
    for a, b in _NORMALIZE_MAP.items():
        low = re.sub(rf"\b{re.escape(a)}\b", b, low)
    return low

_STOP = set(["및","또는","그리고","관련","관련된","여부","것","수","등","해당","하는","경우","대한","으로","에서","하다"])
def _extract_keywords(text: str, top_k: int = 6) -> list:
    toks = re.split(r"[^가-힣A-Za-z0-9]+", text)
    cnt: Dict[str, int] = {}
    for t in toks:
        if len(t) < 2 or t in _STOP:
            continue
        cnt[t] = cnt.get(t, 0) + 1
    return [w for w,_ in sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:top_k]]

def _expand_query_base(q: str) -> str:
    qn = _normalize_terms(q)
    aug: List[str] = []
    for k, vs in _SYNONYM_DICT.items():
        if k in qn:
            aug.extend(vs)
    if any(x in qn for x in ["유튜브","브이로그","틱톡","인스타","라이브","쇼츠"]):
        aug.extend(["인터넷 개인방송", "동영상 공유 서비스", "겸업허가", "제82조", "인터넷 개인방송 활동 제한"])
    aug = list(dict.fromkeys([v for v in aug if v not in q]))  # 중복 제거
    return q if not aug else f"{q} " + " ".join(aug)

def _augment_with_prf(vectorstore, q: str, k_init: int = 4, kw_top: int = 6) -> str:
    try:
        hits = vectorstore.similarity_search_with_score(q, k=k_init)
        if not hits:
            return q
        blob = "\n".join([d.page_content for d,_ in hits])
        kws = _extract_keywords(blob, top_k=kw_top)
        kws = [w for w in kws if w not in q]
        return f"{q} " + " ".join(kws) if kws else q
    except Exception:
        return q

def expand_query(vectorstore, query: str) -> str:
    q1 = _expand_query_base(query)
    q2 = _augment_with_prf(vectorstore, q1)
    if q2 != query:
        print(f"🧩 쿼리 확장: '{query}' → '{q2}'")
    return q2

# ─────────────────────────────────────────
# 🔎 정확 문자열 부스팅 (추가)
# ─────────────────────────────────────────
def _boost_exact_match(query: str, pairs: List[Tuple[Document, float]], factor: float = 0.85) -> List[Tuple[Document, float]]:
    """
    쿼리(정확 문자열)가 청크 본문에 포함되면 거리를 factor 배로 줄여(=유사도↑) 상위로 끌어올림.
    - 대소문자 무시 매칭도 포함
    """
    q = (query or "").strip()
    if not q or not pairs:
        return pairs
    q_low = q.lower()
    boosted: List[Tuple[Document, float]] = []
    for doc, dist in pairs:
        txt = doc.page_content or ""
        if (q in txt) or (q_low in txt.lower()):
            dist *= factor
        boosted.append((doc, dist))
    return boosted

# ─────────────────────────────────────────
# S3 다운로드 유틸 — 희소 인덱스 미존재 시 원격에서 확보
# ─────────────────────────────────────────
def _s3_configured() -> bool:
    return all(os.getenv(k) for k in ["S3_BUCKET", "S3_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])

def _download_sparse_from_s3(local_docs_fp: str, local_tfidf_fp: str, collection_name: str):
    if not _s3_configured():
        return False
    try:
        import boto3
        s3 = boto3.client(
            "s3",
            region_name=os.getenv("S3_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        prefix = os.getenv("AWS_S3_PREFIX", "rag/sparse")
        key_docs = f"{prefix}/{collection_name}_docs.joblib"
        key_tfidf = f"{prefix}/{collection_name}_tfidf.joblib"
        bucket = os.getenv("S3_BUCKET")
        s3.download_file(bucket, key_docs, local_docs_fp)
        s3.download_file(bucket, key_tfidf, local_tfidf_fp)
        print(f"☁️  S3에서 희소 인덱스 다운로드 완료 → {local_docs_fp}, {local_tfidf_fp}")
        return True
    except Exception as e:
        print(f"⚠️ S3 다운로드 실패(희소 검색 비활성): {e}")
        return False

# ─────────────────────────────────────────
# 희소 검색기(TF-IDF)
# ─────────────────────────────────────────
class _SparseSearcher:
    def __init__(self, base_dir: str, collection_name: str):
        os.makedirs(base_dir, exist_ok=True)
        self.collection_name = collection_name
        self.data_fp = os.path.join(base_dir, f"{collection_name}_docs.joblib")
        self.index_fp = os.path.join(base_dir, f"{collection_name}_tfidf.joblib")
        self.enabled = joblib is not None and cosine_similarity is not None
        self.loaded = False
        self.ids = []; self.docs = []; self.metas = []
        self.vectorizer = None; self.mat = None

    def _ensure_local(self):
        if os.path.exists(self.data_fp) and os.path.exists(self.index_fp):
            return
        _download_sparse_from_s3(self.data_fp, self.index_fp, self.collection_name)

    def _lazy_load(self):
        if self.loaded or not self.enabled:
            return
        self._ensure_local()
        if not (os.path.exists(self.data_fp) and os.path.exists(self.index_fp)):
            self.enabled = False
            print("ℹ️ TF-IDF 미활성화(로컬 .joblib 없음) → 벡터 검색만 사용.")
            return
        self.ids, self.docs, self.metas = joblib.load(self.data_fp)
        self.vectorizer, self.mat = joblib.load(self.index_fp)
        self.loaded = True
        print(f"✅ TF-IDF 활성화: {self.collection_name} (docs={len(self.ids)})")

    def search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        if not self.enabled:
            return []
        self._lazy_load()
        if not self.loaded:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.mat).ravel()
        idxs = sims.argsort()[::-1][:k]
        out = []
        for i in idxs:
            out.append((Document(page_content=self.docs[i], metadata=self.metas[i]), float(1 - sims[i])))
        return out

# ─────────────────────────────────────────
# RRF 결합 + 도우미 (internal_retrieve.py와 동일)
# ─────────────────────────────────────────
def _rrf_fusion(dense: List[Tuple[Document, float]], sparse: List[Tuple[Document, float]], k_rrf: int = 60, top_k: int = 6):
    """
    여러 순위 리스트(dense/sparse)를 RRF로 결합.
    - meta['rrf_score'] : 원시 RRF 누적값
    - meta['rrf_sim']   : 0~1 정규화 유사도
    - 반환 distance     : 1 - rrf_sim (작을수록 가깝다)
    """
    pools: Dict[str, Dict[str, Any]] = {}
    for rank, (doc, _) in enumerate(dense, start=1):
        key = (doc.metadata or {}).get("chunk_hash") or f"d:{rank}:{hash(doc.page_content)%10_000_000}"
        pools.setdefault(key, {"doc": doc, "rrf": 0.0})
        pools[key]["rrf"] += 1.0 / (k_rrf + rank)
    for rank, (doc, _) in enumerate(sparse, start=1):
        key = (doc.metadata or {}).get("chunk_hash") or f"s:{rank}:{hash(doc.page_content)%10_000_000}"
        pools.setdefault(key, {"doc": doc, "rrf": 0.0})
        pools[key]["rrf"] += 1.0 / (k_rrf + rank)

    fused = sorted(pools.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]
    max_rrf = max((f["rrf"] for f in fused), default=1.0)

    out: List[Tuple[Document, float]] = []
    for f in fused:
        doc = f["doc"]
        rrf = f["rrf"]
        rrf_sim = rrf / max_rrf if max_rrf > 0 else 0.0
        dist = 1.0 - rrf_sim
        meta = dict(doc.metadata or {})
        meta["rrf_score"] = rrf
        meta["rrf_sim"] = rrf_sim
        doc.metadata = meta
        out.append((doc, dist))
    return out

def _parse_kind_quota(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tok in s.split(","):
        if ":" in tok:
            k, v = tok.split(":", 1)
            try:
                out[k.strip()] = int(v)
            except:
                pass
    return out

_KIND_QUOTA = _parse_kind_quota(KIND_QUOTA)
_cross_encoder = None
def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None and CrossEncoder and ENABLE_RERANK:
        try:
            _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            print(f"✅ Cross-encoder 로드: {CROSS_ENCODER_MODEL}")
        except Exception as e:
            print(f"⚠️ Cross-encoder 로드 실패: {e}")
    return _cross_encoder

def _adaptive_threshold(sims: List[float], base: float) -> float:
    if not sims:
        return base
    xs = sorted(sims)
    q30 = xs[int(len(xs)*0.30)]
    thr = max(base, min(q30, 0.35))  # cap 0.35
    return thr

def _apply_threshold_and_quota(pairs: List[Tuple[Document, float]], top_k: int) -> List[Tuple[Document, float]]:
    """
    - rrf_sim(가능 시) 기반 임계값 컷, 없으면 distance→_stable_similarity 백업.
    - 분포 기반 적응형 임계값 사용(하한: SIM_THRESHOLD).
    - source/kind 쿼터 적용.
    - 컷 사유 카운트 로깅.
    """
    scored_meta = []
    for doc, dist in pairs:
        meta = doc.metadata or {}
        sim = meta.get("rrf_sim", _stable_similarity(dist))
        scored_meta.append((doc, dist, sim))

    sims = [s for _, _, s in scored_meta]
    thr = _adaptive_threshold(sims, SIM_THRESHOLD)

    scored = []
    dropped_reason_counts = {"threshold": 0, "doc_quota": 0, "kind_quota": 0}

    for doc, dist, sim in scored_meta:
        if sim >= thr:
            scored.append((doc, dist, sim))
        else:
            dropped_reason_counts["threshold"] += 1

    by_src: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    kept: List[Tuple[Document, float]] = []

    for doc, dist, sim in sorted(scored, key=lambda x: x[2], reverse=True):
        meta = doc.metadata or {}
        src = meta.get("source", "unknown")
        kind = meta.get("content_kind") or meta.get("content_type") or "text"

        if by_src.get(src, 0) >= DOC_QUOTA:
            dropped_reason_counts["doc_quota"] += 1
            continue
        if _KIND_QUOTA.get(kind, 10**9) <= by_kind.get(kind, 0):
            dropped_reason_counts["kind_quota"] += 1
            continue

        kept.append((doc, dist))
        by_src[src] = by_src.get(src, 0) + 1
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if len(kept) >= top_k:
            break

    print(f"임계/쿼터 적용 결과: thr={thr:.3f}, kept={len(kept)}, dropped={dropped_reason_counts}")
    return kept

def _dedup_by_source_page(pairs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """
    (source, page, sheet) 기준 보수적 디듑.
    - page가 None이면 디듑하지 않음(메타 부실 보호)
    - 페이지당 최대 MAX_PER_PAGE개까지 허용
    """
    counts: Dict[Tuple[Any, Any, Any], int] = {}
    out = []
    for doc, dist in pairs:
        m = doc.metadata or {}
        src = m.get("source")
        page = m.get("page")
        sheet = m.get("sheet")
        if page is None:
            out.append((doc, dist))
            continue
        key = (src, page, sheet)
        if counts.get(key, 0) >= MAX_PER_PAGE:
            continue
        counts[key] = counts.get(key, 0) + 1
        out.append((doc, dist))
    return out

# 블록 단위 중복 제거(동일 라인 3회째부터 제거)
def _dedup_sentences_blockwise(text: str) -> str:
    blocks = [b for b in re.split(r"\n\s*---\s*\n", text) if b.strip()]
    new_blocks = []
    for b in blocks:
        lines = [ln for ln in b.splitlines()]
        seen: Dict[str, int] = {}
        acc = []
        for ln in lines:
            key = ln.strip().lower()
            c = seen.get(key, 0) + 1
            seen[key] = c
            if c > 2:
                continue
            acc.append(ln)
        new_blocks.append("\n".join(acc))
    return "\n\n---\n\n".join(new_blocks)

# LLM Multi-Query 재작성 (보수화)
def _should_multiquery(q: str) -> bool:
    return ENABLE_LLM_MULTIQUERY and len(q) <= SHORT_QUERY_CHAR

def _llm_rewrites(llm: ChatOpenAI, query: str, n: int) -> List[str]:
    if not _should_multiquery(query) or n <= 0:
        return []
    try:
        prompt = (
            "다음 질문을 문서 검색에 강건한 형태로 {n}개 재작성하세요.\n"
            "- 핵심 키워드는 유지하되 표현을 다양화(동의어/전문용어)\n"
            "- 한국어 공식 문서 용어 선호\n"
            "- 한 줄에 하나씩 출력"
        ).format(n=n)
        txt = llm.invoke(prompt + "\n\n질문: " + query).content
        rewrites = [line.strip("-• ").strip() for line in txt.splitlines() if line.strip()]
        uniq, seen = [], set()
        for r in rewrites:
            k = r.lower()
            if k not in seen:
                seen.add(k); uniq.append(r)
        return uniq[:n]
    except Exception:
        return []

def _multi_query_dense(vectorstore, base_q: str, llm: ChatOpenAI, top_each: int) -> List[Tuple[Document, float]]:
    qs = [base_q]
    qs.extend(_llm_rewrites(llm, base_q, N_QUERY_REWRITES))
    qs = list(dict.fromkeys([q for q in qs if q]))
    pool: List[Tuple[Document, float]] = []
    for i, q in enumerate(qs):
        results = vectorstore.similarity_search_with_score(q, k=top_each)
        weight = 2.0 if i == 0 else 1.0  # 원본 쿼리 가중
        pool.extend([(doc, dist / weight) for doc, dist in results])
    return _rrf_fusion(pool, [], k_rrf=60, top_k=min(6, len(qs) * top_each))  # 소규모 RRF

# ─────────────────────────────────────────
# 권한 서비스
# ─────────────────────────────────────────
class UserAwareRAGService:
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
                ("user", "질문: {question}\n\n참고 컨텍스트(여러 청크):\n{context}"),
            ]
        )

    def get_user_context(self, user_id: str) -> Optional[dict]:
        try:
            db = next(get_db())
            employee = employee_crud.get_employee_by_google_id(db, google_user_id=user_id)
            if not employee:
                print(f"❌ 사용자를 찾을 수 없습니다: {user_id}")
                return None
            company_code = _get_company_code(db, employee.company_id)
            return {"employee_id": employee.id, "company_id": employee.company_id, "company_code": company_code}
        except Exception as e:
            print(f"❌ 사용자 컨텍스트 조회 실패: {e}")
            return None
        finally:
            db.close()

    def _allow_meta(self, meta: dict, company_id: int, employee_id: int) -> bool:
        if not isinstance(meta, dict):
            return False
        if meta.get("company_id") != company_id:
            return False
        if meta.get("is_private") is True:
            return meta.get("user_id") == employee_id
        return True

    def retrieve_documents_with_permission(
        self, query: str, user_id: str, top_k: int = 6
    ) -> List[Tuple[str, dict]]:
        user_context = self.get_user_context(user_id)
        if not user_context:
            print(f"❌ 사용자 정보 없음 - 검색 중단: {user_id}")
            return []

        company_code = user_context["company_code"]
        company_id = user_context["company_id"]
        employee_id = user_context["employee_id"]

        try:
            # 회사별 벡터스토어
            import chromadb
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

            # 쿼리 확장
            q_exp = expand_query(vectorstore, query)
            print(f"🔍 권한별 문서 검색(하이브리드 RRF): '{q_exp}' (회사: {company_code}, 사용자: {employee_id})")

            # 권한별 필터
            company_filter = {"$and": [{"company_id": {"$eq": company_id}}, {"is_private": {"$eq": False}}]}
            personal_filter = {"$and": [{"company_id": {"$eq": company_id}}, {"is_private": {"$eq": True}}, {"user_id": {"$eq": employee_id}}]}

            # dense: LLM multi-query 재작성(보수화) → 소규모 RRF 융합
            dense_public_pool = []
            dense_personal_pool = []
            
            # Multi-query 생성
            queries = [q_exp]
            queries.extend(_llm_rewrites(self.llm, q_exp, N_QUERY_REWRITES))
            queries = list(dict.fromkeys([q for q in queries if q]))
            
            for i, q in enumerate(queries):
                weight = 2.0 if i == 0 else 1.0  # 원본 쿼리 가중
                # 회사 공개 문서 검색
                public_results = vectorstore.similarity_search_with_score(q, k=max(3, top_k//2), filter=company_filter)
                dense_public_pool.extend([(doc, dist / weight) for doc, dist in public_results])
                # 개인 문서 검색
                personal_results = vectorstore.similarity_search_with_score(q, k=max(3, top_k//2), filter=personal_filter)
                dense_personal_pool.extend([(doc, dist / weight) for doc, dist in personal_results])
            
            dense_pool = dense_public_pool + dense_personal_pool

            # sparse: 희소 검색(TF-IDF) – 회사별 인덱스 후 권한 사후 필터
            sparse_searcher = _SparseSearcher(SPARSE_INDEX_PATH, company_code)
            sparse_all = sparse_searcher.search(q_exp, k=max(12, top_k * 2))
            sparse = [(d, dist) for (d, dist) in sparse_all if self._allow_meta(d.metadata or {}, company_id, employee_id)]

            # 1) RRF 결합 (후보군 충분 확보)
            fused = _rrf_fusion(dense_pool, sparse, k_rrf=60, top_k=max(top_k * 3, 20))

            # 1-1) 페이지/출처 중복 제거(보수적)
            fused = _dedup_by_source_page(fused)

            # 2) 임계 + 쿼터(적응형 컷)
            fused = _apply_threshold_and_quota(fused, top_k=max(top_k * 2, 20))

            # 2-1) 🔎 정확 문자열 부스팅
            fused = _boost_exact_match(query, fused)

            # 3) CE 리랭크 → 최종 top_k
            fused = self._rerank(query, fused, top_k)

            if not fused:
                print("❌ 관련 문서를 찾지 못했습니다.")
                return []

            contexts: List[Tuple[str, dict]] = []
            print(f"✅ 최종 상위 {len(fused)}개 결과")
            for i, (doc, distance) in enumerate(fused, start=1):
                meta = dict(doc.metadata or {})
                sim = meta.get("rrf_sim", _stable_similarity(distance))
                meta["similarity_score"] = sim
                is_private = meta.get("is_private", False)
                doc_type = "개인 문서" if is_private else "회사 문서"
                preview = (doc.page_content[:100] + "...") if len(doc.page_content) > 100 else doc.page_content
                print(
                    f"  [Rank {i}] sim={sim:.4f}, {doc_type}, src={meta.get('source')} "
                    f"sheet={meta.get('sheet')} page={meta.get('page')} "
                    f"chunk={meta.get('chunk_idx')} kind={meta.get('content_kind') or meta.get('content_type')}"
                )
                print(f"          내용: {preview}")
                contexts.append((doc.page_content, meta))

            return contexts

        except Exception as e:
            print(f"❌ 권한별 문서 검색 중 오류 발생: {e}")
            return []

    def _clip(self, txt: str, max_chars: int) -> str:
        return txt if len(txt) <= max_chars else txt[:max_chars]

    def _rerank(self, query: str, pairs: List[Tuple[Document, float]], top_k: int) -> List[Tuple[Document, float]]:
        # MultiBERT는 다국어 지원 → 한국어 포함 질의도 CE 적용
        ce = _get_cross_encoder()
        if not ce or not pairs:
            return pairs[:top_k]
        try:
            texts = [(query, self._clip(d.page_content, CE_MAX_CHARS)) for d, _ in pairs]
            # sentence-transformers CrossEncoder는 batch_size 인자를 지원
            scores = ce.predict(texts, batch_size=CE_BATCH_SIZE)
            ranked = sorted(zip(pairs, scores), key=lambda x: x[1], reverse=True)[:top_k]
            return [p for (p, _) in ranked]
        except Exception as e:
            print(f"⚠️ Cross-encoder 예측 실패: {e}")
            return pairs[:top_k]

    def generate_answer(self, query: str, contexts: List[Tuple[str, dict]], model: str = None) -> str:
        if not contexts:
            return "관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요."
        try:
            def _rank_key(item):
                _, meta = item
                kind = (meta.get("content_kind") or meta.get("content_type") or "text")
                pri = 2
                if kind in ("image_vlm",): pri = 0
                if kind in ("image_ocr", "text"): pri = 3
                return (pri, meta.get("similarity_score", 0.0))

            contexts = sorted(contexts, key=_rank_key, reverse=True)
            model_label = model or getattr(self.llm, "model", getattr(self.llm, "model_name", "unknown"))
            print(f"⚙️ 답변 생성 중... (컨텍스트 {len(contexts)}개, 모델: {model_label})")
            context_text = _truncate_context_blocks(contexts, max_chars=MAX_CONTEXT_CHARS)
            # 블록 단위 중복 제거(3회째부터 제거)
            context_text = _dedup_sentences_blockwise(context_text)
            used_llm = self.llm if model is None else ChatOpenAI(model=model, temperature=0)
            chain = self.prompt | used_llm | self.parser
            answer = chain.invoke({"question": query, "context": context_text})
            print("✅ 답변 생성 완료")
            return answer
        except Exception as e:
            print(f"❌ 답변 생성 중 오류 발생: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {e}"

    def query_rag_with_permission(self, query: str, user_id: str, top_k: int = 6, model: str = None, show_sources: bool = True) -> str:
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
    """사용자별 권한을 고려한 RAG 검색/답변 (회사 공개 + 본인 개인 문서)."""
    try:
        return _user_aware_rag_service.query_rag_with_permission(
            query=query, user_id=user_id, top_k=6, show_sources=True
        )
    except Exception as e:
        print(f"❌ 사용자별 RAG 검색 실패: {e}")
        return f"문서 검색 중 오류가 발생했습니다: {e}"

# ========================= 사용자별 도구 팩토리 =========================
def create_user_aware_rag_tools(user_id: str) -> list:
    """특정 사용자를 위한 RAG 도구 생성 (user_id 바인딩)."""
    print(f"🔧 사용자별 RAG 도구 생성 중: {user_id}")

    @tool
    def internal_rag_search(query: str) -> str:
        """업로드된 회사/개인 문서에서 검색/답변(하이브리드)."""
        print(f"🔍 internal_rag_search 호출됨: query='{query}', user_id='{user_id}'")
        try:
            result = _user_aware_rag_service.query_rag_with_permission(
                query=query, user_id=user_id, top_k=6, show_sources=True
            )
            return result
        except Exception as e:
            print(f"❌ RAG 검색 실패: {e}")
            return f"내부 문서 검색 중 오류가 발생했습니다: {e}"

    print(f"✅ RAG 도구 생성 완료: {user_id}")
    return [internal_rag_search]

# 기존 범용 도구
user_aware_rag_tools = [user_aware_rag_search]


# # app/rag/internal_data_rag/user_aware_retrieve.py
# # -*- coding: utf-8 -*-
# """
# 사용자별 권한을 고려한 RAG 문서 검색 서비스
# - 회사 공개 문서: 같은 회사 직원 모두 접근 가능
# - 개인 문서: 본인만 접근 가능
# - 다른 직원의 개인 문서: 접근 불가

# - TF-IDF 희소 검색 + 권한 사후 필터 + RRF 결합
# - 희소 인덱스가 로컬에 없으면 S3에서 자동 다운로드 후 캐시
# - 질의 확장(동의어/플랫폼 정규화 + PRF) → dense/sparse에 동일 적용

# 추가:
# - 유사도 임계값 컷 + 문서/유형 쿼터링
# - Cross-encoder Rerank
# - LLM Multi-Query
# - 출처/페이지 dedup + 문장 중복 제거
# """

# import os
# import re
# import json
# from pathlib import Path
# from typing import List, Tuple, Optional, Dict, Any
# from sqlalchemy.orm import Session
# from dotenv import load_dotenv

# # 희소 검색(있으면 사용)
# try:
#     from sklearn.metrics.pairwise import cosine_similarity
#     import joblib
# except Exception:
#     cosine_similarity = None
#     joblib = None

# # Cross-encoder (선택)
# try:
#     from sentence_transformers import CrossEncoder
# except Exception:
#     CrossEncoder = None

# load_dotenv()

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.tools import tool
# from langchain_core.documents import Document

# from app.utils.db import get_db
# from app.features.login.employee_google import crud as employee_crud
# from app.features.admin.services.file_ingest_service import _get_company_code
# from app.rag.internal_data_rag.internal_retrieve import (
#     _stable_similarity,
#     _truncate_context_blocks,
#     MAX_CONTEXT_CHARS,
#     CHROMA_PATH,
#     EMBED_MODEL,
#     CHAT_MODEL,
# )

# # ─────────────────────────────────────────
# # 환경/옵션 (internal_retrieve.py와 동일하게)
# # ─────────────────────────────────────────
# SPARSE_INDEX_PATH = os.getenv("SPARSE_INDEX_PATH", os.path.join(CHROMA_PATH, "sparse"))

# # 임계/쿼터/리랭크 (internal_retrieve.py와 동일)
# SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.25"))
# DOC_QUOTA = int(os.getenv("DOC_QUOTA", "15"))
# KIND_QUOTA = os.getenv("KIND_QUOTA", "text:12,image_ocr:8,image_vlm:4")
# ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
# CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MultiBERT-L-12")
# # CE 입력 최적화
# CE_MAX_CHARS = int(os.getenv("CE_MAX_CHARS", "1800"))
# CE_BATCH_SIZE = int(os.getenv("CE_BATCH_SIZE", "16"))

# # LLM Multi-Query (보수화)
# ENABLE_LLM_MULTIQUERY = os.getenv("ENABLE_LLM_MULTIQUERY", "true").lower() == "true"
# N_QUERY_REWRITES = int(os.getenv("N_QUERY_REWRITES", "1"))  # 기본 1로 완화
# SHORT_QUERY_CHAR = int(os.getenv("SHORT_QUERY_CHAR", "40"))

# # 디듑 파라미터
# MAX_PER_PAGE = int(os.getenv("MAX_PER_PAGE", "2"))  # 페이지당 최대 유지 개수

# # S3(옵션) — 희소 인덱스 동기화 (이름 통일)
# S3_BUCKET = os.getenv("S3_BUCKET")
# S3_REGION = os.getenv("S3_REGION")
# AWS_S3_PREFIX = os.getenv("AWS_S3_PREFIX", "rag/sparse")

# # 질의 동의어 사전(외부 파일 경로 - 선택)
# QUERY_SYNONYM_PATH = os.getenv("QUERY_SYNONYM_PATH")  # e.g. ./config/synonyms.json

# # ─────────────────────────────────────────
# # NEW: 질의 확장 유틸(동의어/정규화 + PRF)
# # ─────────────────────────────────────────
# def _load_synonym_dict() -> dict:
#     try:
#         if QUERY_SYNONYM_PATH and Path(QUERY_SYNONYM_PATH).exists():
#             with open(QUERY_SYNONYM_PATH, "r", encoding="utf-8") as f:
#                 return json.load(f)
#     except Exception:
#         pass
#     return {
#         "유튜브": ["인터넷 개인방송", "동영상 공유 서비스", "개인방송", "영상 업로드"],
#         "브이로그": ["인터넷 개인방송", "개인방송", "영상 제작", "동영상 업로드"],
#         "틱톡": ["인터넷 개인방송", "동영상 공유 서비스", "소셜미디어"],
#         "인스타": ["소셜미디어", "SNS", "콘텐츠 업로드"],
#         "라이브": ["실시간 방송", "라이브 스트리밍", "인터넷 개인방송"],
#         "수익": ["광고수익", "수익 창출", "협찬", "간접광고", "PPL"],
#         "협찬": ["간접광고", "PPL", "대가성 제공"],
#         "겸업": ["겸업허가", "겸업의 허가", "부업"],
#         "보안": ["직무상 비밀", "영업비밀", "정보보안"],
#         "명예훼손": ["명예 침해", "권리 침해"],
#         # 인사/경력 관련 동의어 추가
#         "이력사": ["이력서", "경력", "경험", "커리어", "프로필", "개인정보"],
#         "이력서": ["이력사", "경력서", "경력", "커리어", "프로필", "개인정보"],
#         "경력": ["이력서", "이력사", "경험", "커리어", "프로필"],
#         "경험": ["이력서", "이력사", "경력", "커리어"],
#     }

# _SYNONYM_DICT = _load_synonym_dict()
# _NORMALIZE_MAP = {
#     "youtube": "유튜브",
#     "yt": "유튜브",
#     "vlog": "브이로그",
#     "shorts": "쇼츠",
#     "tiktok": "틱톡",
#     "instagram": "인스타",
# }

# def _normalize_terms(q: str) -> str:
#     low = q.lower()
#     for a, b in _NORMALIZE_MAP.items():
#         low = re.sub(rf"\b{re.escape(a)}\b", b, low)
#     return low

# _STOP = set(["및","또는","그리고","관련","관련된","여부","것","수","등","해당","하는","경우","대한","으로","에서","하다"])
# def _extract_keywords(text: str, top_k: int = 6) -> list:
#     toks = re.split(r"[^가-힣A-Za-z0-9]+", text)
#     cnt: Dict[str, int] = {}
#     for t in toks:
#         if len(t) < 2 or t in _STOP:
#             continue
#         cnt[t] = cnt.get(t, 0) + 1
#     return [w for w,_ in sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:top_k]]

# def _expand_query_base(q: str) -> str:
#     qn = _normalize_terms(q)
#     aug: List[str] = []
#     for k, vs in _SYNONYM_DICT.items():
#         if k in qn:
#             aug.extend(vs)
#     if any(x in qn for x in ["유튜브","브이로그","틱톡","인스타","라이브","쇼츠"]):
#         aug.extend(["인터넷 개인방송", "동영상 공유 서비스", "겸업허가", "제82조", "인터넷 개인방송 활동 제한"])
#     aug = list(dict.fromkeys([v for v in aug if v not in q]))  # 중복 제거
#     return q if not aug else f"{q} " + " ".join(aug)

# def _augment_with_prf(vectorstore, q: str, k_init: int = 4, kw_top: int = 6) -> str:
#     try:
#         hits = vectorstore.similarity_search_with_score(q, k=k_init)
#         if not hits:
#             return q
#         blob = "\n".join([d.page_content for d,_ in hits])
#         kws = _extract_keywords(blob, top_k=kw_top)
#         kws = [w for w in kws if w not in q]
#         return f"{q} " + " ".join(kws) if kws else q
#     except Exception:
#         return q

# def expand_query(vectorstore, query: str) -> str:
#     q1 = _expand_query_base(query)
#     q2 = _augment_with_prf(vectorstore, q1)
#     if q2 != query:
#         print(f"🧩 쿼리 확장: '{query}' → '{q2}'")
#     return q2

# # ─────────────────────────────────────────
# # S3 다운로드 유틸 — 희소 인덱스 미존재 시 원격에서 확보
# # ─────────────────────────────────────────
# def _s3_configured() -> bool:
#     return all(os.getenv(k) for k in ["S3_BUCKET", "S3_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])

# def _download_sparse_from_s3(local_docs_fp: str, local_tfidf_fp: str, collection_name: str):
#     if not _s3_configured():
#         return False
#     try:
#         import boto3
#         s3 = boto3.client(
#             "s3",
#             region_name=os.getenv("S3_REGION"),
#             aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#             aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#         )
#         prefix = os.getenv("AWS_S3_PREFIX", "rag/sparse")
#         key_docs = f"{prefix}/{collection_name}_docs.joblib"
#         key_tfidf = f"{prefix}/{collection_name}_tfidf.joblib"
#         bucket = os.getenv("S3_BUCKET")
#         s3.download_file(bucket, key_docs, local_docs_fp)
#         s3.download_file(bucket, key_tfidf, local_tfidf_fp)
#         print(f"☁️  S3에서 희소 인덱스 다운로드 완료 → {local_docs_fp}, {local_tfidf_fp}")
#         return True
#     except Exception as e:
#         print(f"⚠️ S3 다운로드 실패(희소 검색 비활성): {e}")
#         return False

# # ─────────────────────────────────────────
# # 희소 검색기(TF-IDF)
# # ─────────────────────────────────────────
# class _SparseSearcher:
#     def __init__(self, base_dir: str, collection_name: str):
#         os.makedirs(base_dir, exist_ok=True)
#         self.collection_name = collection_name
#         self.data_fp = os.path.join(base_dir, f"{collection_name}_docs.joblib")
#         self.index_fp = os.path.join(base_dir, f"{collection_name}_tfidf.joblib")
#         self.enabled = joblib is not None and cosine_similarity is not None
#         self.loaded = False
#         self.ids = []; self.docs = []; self.metas = []
#         self.vectorizer = None; self.mat = None

#     def _ensure_local(self):
#         if os.path.exists(self.data_fp) and os.path.exists(self.index_fp):
#             return
#         _download_sparse_from_s3(self.data_fp, self.index_fp, self.collection_name)

#     def _lazy_load(self):
#         if self.loaded or not self.enabled:
#             return
#         self._ensure_local()
#         if not (os.path.exists(self.data_fp) and os.path.exists(self.index_fp)):
#             self.enabled = False
#             print("ℹ️ TF-IDF 미활성화(로컬 .joblib 없음) → 벡터 검색만 사용.")
#             return
#         self.ids, self.docs, self.metas = joblib.load(self.data_fp)
#         self.vectorizer, self.mat = joblib.load(self.index_fp)
#         self.loaded = True
#         print(f"✅ TF-IDF 활성화: {self.collection_name} (docs={len(self.ids)})")

#     def search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
#         if not self.enabled:
#             return []
#         self._lazy_load()
#         if not self.loaded:
#             return []
#         qv = self.vectorizer.transform([query])
#         sims = cosine_similarity(qv, self.mat).ravel()
#         idxs = sims.argsort()[::-1][:k]
#         out = []
#         for i in idxs:
#             out.append((Document(page_content=self.docs[i], metadata=self.metas[i]), float(1 - sims[i])))
#         return out

# # ─────────────────────────────────────────
# # RRF 결합 + 도우미 (internal_retrieve.py와 동일)
# # ─────────────────────────────────────────
# def _rrf_fusion(dense: List[Tuple[Document, float]], sparse: List[Tuple[Document, float]], k_rrf: int = 60, top_k: int = 6):
#     """
#     여러 순위 리스트(dense/sparse)를 RRF로 결합.
#     - meta['rrf_score'] : 원시 RRF 누적값
#     - meta['rrf_sim']   : 0~1 정규화 유사도
#     - 반환 distance     : 1 - rrf_sim (작을수록 가깝다)
#     """
#     pools: Dict[str, Dict[str, Any]] = {}
#     for rank, (doc, _) in enumerate(dense, start=1):
#         key = (doc.metadata or {}).get("chunk_hash") or f"d:{rank}:{hash(doc.page_content)%10_000_000}"
#         pools.setdefault(key, {"doc": doc, "rrf": 0.0})
#         pools[key]["rrf"] += 1.0 / (k_rrf + rank)
#     for rank, (doc, _) in enumerate(sparse, start=1):
#         key = (doc.metadata or {}).get("chunk_hash") or f"s:{rank}:{hash(doc.page_content)%10_000_000}"
#         pools.setdefault(key, {"doc": doc, "rrf": 0.0})
#         pools[key]["rrf"] += 1.0 / (k_rrf + rank)

#     fused = sorted(pools.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]
#     max_rrf = max((f["rrf"] for f in fused), default=1.0)

#     out: List[Tuple[Document, float]] = []
#     for f in fused:
#         doc = f["doc"]
#         rrf = f["rrf"]
#         rrf_sim = rrf / max_rrf if max_rrf > 0 else 0.0
#         dist = 1.0 - rrf_sim
#         meta = dict(doc.metadata or {})
#         meta["rrf_score"] = rrf
#         meta["rrf_sim"] = rrf_sim
#         doc.metadata = meta
#         out.append((doc, dist))
#     return out

# def _parse_kind_quota(s: str) -> Dict[str, int]:
#     out: Dict[str, int] = {}
#     for tok in s.split(","):
#         if ":" in tok:
#             k, v = tok.split(":", 1)
#             try:
#                 out[k.strip()] = int(v)
#             except:
#                 pass
#     return out

# _KIND_QUOTA = _parse_kind_quota(KIND_QUOTA)
# _cross_encoder = None
# def _get_cross_encoder():
#     global _cross_encoder
#     if _cross_encoder is None and CrossEncoder and ENABLE_RERANK:
#         try:
#             _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
#             print(f"✅ Cross-encoder 로드: {CROSS_ENCODER_MODEL}")
#         except Exception as e:
#             print(f"⚠️ Cross-encoder 로드 실패: {e}")
#     return _cross_encoder

# def _adaptive_threshold(sims: List[float], base: float) -> float:
#     if not sims:
#         return base
#     xs = sorted(sims)
#     q30 = xs[int(len(xs)*0.30)]
#     thr = max(base, min(q30, 0.35))  # cap 0.35
#     return thr

# def _apply_threshold_and_quota(pairs: List[Tuple[Document, float]], top_k: int) -> List[Tuple[Document, float]]:
#     """
#     - rrf_sim(가능 시) 기반 임계값 컷, 없으면 distance→_stable_similarity 백업.
#     - 분포 기반 적응형 임계값 사용(하한: SIM_THRESHOLD).
#     - source/kind 쿼터 적용.
#     - 컷 사유 카운트 로깅.
#     """
#     scored_meta = []
#     for doc, dist in pairs:
#         meta = doc.metadata or {}
#         sim = meta.get("rrf_sim", _stable_similarity(dist))
#         scored_meta.append((doc, dist, sim))

#     sims = [s for _, _, s in scored_meta]
#     thr = _adaptive_threshold(sims, SIM_THRESHOLD)

#     scored = []
#     dropped_reason_counts = {"threshold": 0, "doc_quota": 0, "kind_quota": 0}

#     for doc, dist, sim in scored_meta:
#         if sim >= thr:
#             scored.append((doc, dist, sim))
#         else:
#             dropped_reason_counts["threshold"] += 1

#     by_src: Dict[str, int] = {}
#     by_kind: Dict[str, int] = {}
#     kept: List[Tuple[Document, float]] = []

#     for doc, dist, sim in sorted(scored, key=lambda x: x[2], reverse=True):
#         meta = doc.metadata or {}
#         src = meta.get("source", "unknown")
#         kind = meta.get("content_kind") or meta.get("content_type") or "text"

#         if by_src.get(src, 0) >= DOC_QUOTA:
#             dropped_reason_counts["doc_quota"] += 1
#             continue
#         if _KIND_QUOTA.get(kind, 10**9) <= by_kind.get(kind, 0):
#             dropped_reason_counts["kind_quota"] += 1
#             continue

#         kept.append((doc, dist))
#         by_src[src] = by_src.get(src, 0) + 1
#         by_kind[kind] = by_kind.get(kind, 0) + 1
#         if len(kept) >= top_k:
#             break

#     print(f"임계/쿼터 적용 결과: thr={thr:.3f}, kept={len(kept)}, dropped={dropped_reason_counts}")
#     return kept

# def _dedup_by_source_page(pairs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
#     """
#     (source, page, sheet) 기준 보수적 디듑.
#     - page가 None이면 디듑하지 않음(메타 부실 보호)
#     - 페이지당 최대 MAX_PER_PAGE개까지 허용
#     """
#     counts: Dict[Tuple[Any, Any, Any], int] = {}
#     out = []
#     for doc, dist in pairs:
#         m = doc.metadata or {}
#         src = m.get("source")
#         page = m.get("page")
#         sheet = m.get("sheet")
#         if page is None:
#             out.append((doc, dist))
#             continue
#         key = (src, page, sheet)
#         if counts.get(key, 0) >= MAX_PER_PAGE:
#             continue
#         counts[key] = counts.get(key, 0) + 1
#         out.append((doc, dist))
#     return out

# # 블록 단위 중복 제거(동일 라인 3회째부터 제거)
# def _dedup_sentences_blockwise(text: str) -> str:
#     blocks = [b for b in re.split(r"\n\s*---\s*\n", text) if b.strip()]
#     new_blocks = []
#     for b in blocks:
#         lines = [ln for ln in b.splitlines()]
#         seen: Dict[str, int] = {}
#         acc = []
#         for ln in lines:
#             key = ln.strip().lower()
#             c = seen.get(key, 0) + 1
#             seen[key] = c
#             if c > 2:
#                 continue
#             acc.append(ln)
#         new_blocks.append("\n".join(acc))
#     return "\n\n---\n\n".join(new_blocks)

# # LLM Multi-Query 재작성 (보수화)
# def _should_multiquery(q: str) -> bool:
#     return ENABLE_LLM_MULTIQUERY and len(q) <= SHORT_QUERY_CHAR

# def _llm_rewrites(llm: ChatOpenAI, query: str, n: int) -> List[str]:
#     if not _should_multiquery(query) or n <= 0:
#         return []
#     try:
#         prompt = (
#             "다음 질문을 문서 검색에 강건한 형태로 {n}개 재작성하세요.\n"
#             "- 핵심 키워드는 유지하되 표현을 다양화(동의어/전문용어)\n"
#             "- 한국어 공식 문서 용어 선호\n"
#             "- 한 줄에 하나씩 출력"
#         ).format(n=n)
#         txt = llm.invoke(prompt + "\n\n질문: " + query).content
#         rewrites = [line.strip("-• ").strip() for line in txt.splitlines() if line.strip()]
#         uniq, seen = [], set()
#         for r in rewrites:
#             k = r.lower()
#             if k not in seen:
#                 seen.add(k); uniq.append(r)
#         return uniq[:n]
#     except Exception:
#         return []

# def _multi_query_dense(vectorstore, base_q: str, llm: ChatOpenAI, top_each: int) -> List[Tuple[Document, float]]:
#     qs = [base_q]
#     qs.extend(_llm_rewrites(llm, base_q, N_QUERY_REWRITES))
#     qs = list(dict.fromkeys([q for q in qs if q]))
#     pool: List[Tuple[Document, float]] = []
#     for i, q in enumerate(qs):
#         results = vectorstore.similarity_search_with_score(q, k=top_each)
#         weight = 2.0 if i == 0 else 1.0  # 원본 쿼리 가중
#         pool.extend([(doc, dist / weight) for doc, dist in results])
#     return _rrf_fusion(pool, [], k_rrf=60, top_k=min(6, len(qs) * top_each))  # 소규모 RRF

# # ─────────────────────────────────────────
# # 권한 서비스
# # ─────────────────────────────────────────
# class UserAwareRAGService:
#     def __init__(self):
#         self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
#         self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
#         self.parser = StrOutputParser()
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "당신은 사내 문서를 기반으로 정확히 답하는 어시스턴트입니다. "
#                     "주어진 컨텍스트에서만 정보를 추출하여 답변하고, "
#                     "추측하지 말고 모르는 내용은 '모른다'고 명확히 말하세요. "
#                     "가능한 한 출처 문서명과 함께 답변하세요.",
#                 ),
#                 ("user", "질문: {question}\n\n참고 컨텍스트(여러 청크):\n{context}"),
#             ]
#         )

#     def get_user_context(self, user_id: str) -> Optional[dict]:
#         try:
#             db = next(get_db())
#             employee = employee_crud.get_employee_by_google_id(db, google_user_id=user_id)
#             if not employee:
#                 print(f"❌ 사용자를 찾을 수 없습니다: {user_id}")
#                 return None
#             company_code = _get_company_code(db, employee.company_id)
#             return {"employee_id": employee.id, "company_id": employee.company_id, "company_code": company_code}
#         except Exception as e:
#             print(f"❌ 사용자 컨텍스트 조회 실패: {e}")
#             return None
#         finally:
#             db.close()

#     def _allow_meta(self, meta: dict, company_id: int, employee_id: int) -> bool:
#         if not isinstance(meta, dict):
#             return False
#         if meta.get("company_id") != company_id:
#             return False
#         if meta.get("is_private") is True:
#             return meta.get("user_id") == employee_id
#         return True

#     def retrieve_documents_with_permission(
#         self, query: str, user_id: str, top_k: int = 6
#     ) -> List[Tuple[str, dict]]:
#         user_context = self.get_user_context(user_id)
#         if not user_context:
#             print(f"❌ 사용자 정보 없음 - 검색 중단: {user_id}")
#             return []

#         company_code = user_context["company_code"]
#         company_id = user_context["company_id"]
#         employee_id = user_context["employee_id"]

#         try:
#             # 회사별 벡터스토어
#             import chromadb
#             client = chromadb.CloudClient(
#                 tenant=os.getenv("CHROMA_TENANT"),
#                 database=os.getenv("CHROMA_DATABASE"),
#                 api_key=os.getenv("CHROMA_API_KEY"),
#             )
#             vectorstore = Chroma(
#                 client=client,
#                 collection_name=company_code,
#                 embedding_function=self.embeddings,
#             )

#             # 쿼리 확장
#             q_exp = expand_query(vectorstore, query)
#             print(f"🔍 권한별 문서 검색(하이브리드 RRF): '{q_exp}' (회사: {company_code}, 사용자: {employee_id})")

#             # 권한별 필터
#             company_filter = {"$and": [{"company_id": {"$eq": company_id}}, {"is_private": {"$eq": False}}]}
#             personal_filter = {"$and": [{"company_id": {"$eq": company_id}}, {"is_private": {"$eq": True}}, {"user_id": {"$eq": employee_id}}]}

#             # dense: LLM multi-query 재작성(보수화) → 소규모 RRF 융합
#             # 권한별 필터로 검색
#             dense_public_pool = []
#             dense_personal_pool = []
            
#             # Multi-query 생성
#             queries = [q_exp]
#             queries.extend(_llm_rewrites(self.llm, q_exp, N_QUERY_REWRITES))
#             queries = list(dict.fromkeys([q for q in queries if q]))
            
#             for i, q in enumerate(queries):
#                 weight = 2.0 if i == 0 else 1.0  # 원본 쿼리 가중
#                 # 회사 공개 문서 검색
#                 public_results = vectorstore.similarity_search_with_score(q, k=max(3, top_k//2), filter=company_filter)
#                 dense_public_pool.extend([(doc, dist / weight) for doc, dist in public_results])
#                 # 개인 문서 검색
#                 personal_results = vectorstore.similarity_search_with_score(q, k=max(3, top_k//2), filter=personal_filter)
#                 dense_personal_pool.extend([(doc, dist / weight) for doc, dist in personal_results])
            
#             dense_pool = dense_public_pool + dense_personal_pool

#             # sparse: 희소 검색(TF-IDF) – 회사별 인덱스 후 권한 사후 필터
#             sparse_searcher = _SparseSearcher(SPARSE_INDEX_PATH, company_code)
#             sparse_all = sparse_searcher.search(q_exp, k=max(12, top_k * 2))
#             sparse = [(d, dist) for (d, dist) in sparse_all if self._allow_meta(d.metadata or {}, company_id, employee_id)]

#             # 1) RRF 결합 (후보군 충분 확보)
#             fused = _rrf_fusion(dense_pool, sparse, k_rrf=60, top_k=max(top_k * 3, 20))

#             # 1-1) 페이지/출처 중복 제거(보수적)
#             fused = _dedup_by_source_page(fused)

#             # 2) 임계 + 쿼터(적응형 컷)
#             fused = _apply_threshold_and_quota(fused, top_k=max(top_k * 2, 20))

#             # 3) CE 리랭크 → 최종 top_k
#             fused = self._rerank(query, fused, top_k)

#             if not fused:
#                 print("❌ 관련 문서를 찾지 못했습니다.")
#                 return []

#             contexts: List[Tuple[str, dict]] = []
#             print(f"✅ 최종 상위 {len(fused)}개 결과")
#             for i, (doc, distance) in enumerate(fused, start=1):
#                 meta = dict(doc.metadata or {})
#                 sim = meta.get("rrf_sim", _stable_similarity(distance))
#                 meta["similarity_score"] = sim
#                 is_private = meta.get("is_private", False)
#                 doc_type = "개인 문서" if is_private else "회사 문서"
#                 preview = (doc.page_content[:100] + "...") if len(doc.page_content) > 100 else doc.page_content
#                 print(
#                     f"  [Rank {i}] sim={sim:.4f}, {doc_type}, src={meta.get('source')} "
#                     f"sheet={meta.get('sheet')} page={meta.get('page')} "
#                     f"chunk={meta.get('chunk_idx')} kind={meta.get('content_kind') or meta.get('content_type')}"
#                 )
#                 print(f"          내용: {preview}")
#                 contexts.append((doc.page_content, meta))

#             return contexts

#         except Exception as e:
#             print(f"❌ 권한별 문서 검색 중 오류 발생: {e}")
#             return []

#     def _clip(self, txt: str, max_chars: int) -> str:
#         return txt if len(txt) <= max_chars else txt[:max_chars]

#     def _rerank(self, query: str, pairs: List[Tuple[Document, float]], top_k: int) -> List[Tuple[Document, float]]:
#         # MultiBERT는 다국어 지원 → 한국어 포함 질의도 CE 적용
#         ce = _get_cross_encoder()
#         if not ce or not pairs:
#             return pairs[:top_k]
#         try:
#             texts = [(query, self._clip(d.page_content, CE_MAX_CHARS)) for d, _ in pairs]
#             # sentence-transformers CrossEncoder는 batch_size 인자를 지원
#             scores = ce.predict(texts, batch_size=CE_BATCH_SIZE)
#             ranked = sorted(zip(pairs, scores), key=lambda x: x[1], reverse=True)[:top_k]
#             return [p for (p, _) in ranked]
#         except Exception as e:
#             print(f"⚠️ Cross-encoder 예측 실패: {e}")
#             return pairs[:top_k]

#     def generate_answer(self, query: str, contexts: List[Tuple[str, dict]], model: str = None) -> str:
#         if not contexts:
#             return "관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요."
#         try:
#             def _rank_key(item):
#                 _, meta = item
#                 kind = (meta.get("content_kind") or meta.get("content_type") or "text")
#                 pri = 2
#                 if kind in ("image_vlm",): pri = 0
#                 if kind in ("image_ocr", "text"): pri = 3
#                 return (pri, meta.get("similarity_score", 0.0))

#             contexts = sorted(contexts, key=_rank_key, reverse=True)
#             model_label = model or getattr(self.llm, "model", getattr(self.llm, "model_name", "unknown"))
#             print(f"⚙️ 답변 생성 중... (컨텍스트 {len(contexts)}개, 모델: {model_label})")
#             context_text = _truncate_context_blocks(contexts, max_chars=MAX_CONTEXT_CHARS)
#             # 블록 단위 중복 제거(3회째부터 제거)
#             context_text = _dedup_sentences_blockwise(context_text)
#             used_llm = self.llm if model is None else ChatOpenAI(model=model, temperature=0)
#             chain = self.prompt | used_llm | self.parser
#             answer = chain.invoke({"question": query, "context": context_text})
#             print("✅ 답변 생성 완료")
#             return answer
#         except Exception as e:
#             print(f"❌ 답변 생성 중 오류 발생: {e}")
#             return f"답변 생성 중 오류가 발생했습니다: {e}"

#     def query_rag_with_permission(self, query: str, user_id: str, top_k: int = 6, model: str = None, show_sources: bool = True) -> str:
#         print(f"\n🔍 권한별 RAG 질의: {query} (사용자: {user_id})")
#         contexts = self.retrieve_documents_with_permission(query, user_id, top_k)
#         if not contexts:
#             return "죄송합니다. 접근 가능한 관련 문서를 찾을 수 없습니다."
#         answer = self.generate_answer(query, contexts, model)
#         if show_sources:
#             sources = []
#             for _, meta in contexts:
#                 is_private = meta.get("is_private", False)
#                 doc_type = "개인 문서" if is_private else "회사 문서"
#                 src = f"- {meta.get('source', '알 수 없음')} ({doc_type}, 청크 {meta.get('chunk_idx', 'N/A')})"
#                 if src not in sources:
#                     sources.append(src)
#             return f"{answer}\n\n📋 참고한 문서:\n" + "\n".join(sources)
#         return answer

# # 전역 서비스 인스턴스
# _user_aware_rag_service = UserAwareRAGService()

# # ========================= LangChain 도구 정의 =========================
# @tool
# def user_aware_rag_search(query: str, user_id: str) -> str:
#     """사용자별 권한을 고려한 RAG 검색/답변 (회사 공개 + 본인 개인 문서)."""
#     try:
#         return _user_aware_rag_service.query_rag_with_permission(
#             query=query, user_id=user_id, top_k=6, show_sources=True
#         )
#     except Exception as e:
#         print(f"❌ 사용자별 RAG 검색 실패: {e}")
#         return f"문서 검색 중 오류가 발생했습니다: {e}"

# # ========================= 사용자별 도구 팩토리 =========================
# def create_user_aware_rag_tools(user_id: str) -> list:
#     """특정 사용자를 위한 RAG 도구 생성 (user_id 바인딩)."""
#     print(f"🔧 사용자별 RAG 도구 생성 중: {user_id}")

#     @tool
#     def internal_rag_search(query: str) -> str:
#         """업로드된 회사/개인 문서에서 검색/답변(하이브리드)."""
#         print(f"🔍 internal_rag_search 호출됨: query='{query}', user_id='{user_id}'")
#         try:
#             result = _user_aware_rag_service.query_rag_with_permission(
#                 query=query, user_id=user_id, top_k=6, show_sources=True
#             )
#             return result
#         except Exception as e:
#             print(f"❌ RAG 검색 실패: {e}")
#             return f"내부 문서 검색 중 오류가 발생했습니다: {e}"

#     print(f"✅ RAG 도구 생성 완료: {user_id}")
#     return [internal_rag_search]

# # 기존 범용 도구
# user_aware_rag_tools = [user_aware_rag_search]
