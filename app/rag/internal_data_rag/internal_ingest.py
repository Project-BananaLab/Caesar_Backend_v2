# app/rag/internal_data_rag/internal_ingest.py
# -*- coding: utf-8 -*-
# 문서 임베딩 및 ChromaDB 저장 서비스
# (MD 정규화 + 시트 단위 청킹 + 메타 확장 + OCR 1차 + 조건부 VLM 2차 + TF-IDF 희소 인덱스)
#
# 설계 요약
# - 문서 → Markdown 정규화(PDF/DOCX/XLSX 지원) + 페이지/시트 경계 보존
# - 고정 청킹(적응형 제거) : CHUNK_SIZE/CHUNK_OVERLAP
# - 이미지(OCR 먼저) → OCR 글자수 부족/사진·도형일 가능성 시 VLM 캡션 추가
# - Chroma Cloud 우선, 실패 시 Local → In-Memory 폴백
# - 희소 인덱스(TF-IDF) 동시 구축 → 검색에서 RRF로 결합(#2, #3 파일과 호환)
# - S3 업로드(옵션) → 검색 서버에서 자동 다운로드
#
# 추가: Semantic Chunking(옵션)
# (섹션 기반 + 시맨틱 + 가비지 필터/사후 병합 + OCR/VLM 병합 + TF-IDF)
# 원문 보존 우선: 불필요한 삭제/필터링 제거


import os, sys, re, io, zipfile, base64, hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import deque, defaultdict

import pdfplumber, docx, openpyxl
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# ── 선택 의존성
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
except Exception:
    TfidfVectorizer = None
    joblib = None
try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:
    SentenceTransformer = None
    st_util = None

load_dotenv()
client = OpenAI()

# ── 설정
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "inside_data4")
CHROMA_PATH = os.path.abspath(CHROMA_PATH)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# 시맨틱
SEM_SPLIT_ENABLE = os.getenv("SEM_SPLIT_ENABLE", "true").lower() == "true"
SEM_MODEL  = os.getenv("SEM_MODEL", "all-MiniLM-L6-v2")
SEM_MIN_SIM= float(os.getenv("SEM_MIN_SIM", "0.45"))
SEM_MAX_CHARS = int(os.getenv("SEM_MAX_CHARS", "500"))  # Chroma Cloud 300개 제한을 위해 500자로 강제 제한
MIN_CHARS_PER_CHUNK = int(os.getenv("MIN_CHARS_PER_CHUNK", "280"))  # ▶ 최소 글자
MIN_SENTS_PER_CHUNK = int(os.getenv("MIN_SENTS_PER_CHUNK", "2"))    # ▶ 최소 문장

# 섹션
SECTION_TITLE_MAX_LEN = int(os.getenv("SECTION_TITLE_MAX_LEN", "120"))
SECTION_HEADING_LEVELS = os.getenv("SECTION_HEADING_LEVELS", "1,2,3")

# 정규화
NORMALIZE_TO_MD = os.getenv("NORMALIZE_TO_MD", "true").lower() == "true"
PDF_ADD_PAGE_MARKERS = os.getenv("PDF_ADD_PAGE_MARKERS", "true").lower() == "true"
PDF_HEADING_HEURISTIC = os.getenv("PDF_HEADING_HEURISTIC", "true").lower() == "true"

# XLSX
XLSX_MAX_ROWS_PER_SHEET = int(os.getenv("XLSX_MAX_ROWS_PER_SHEET", "10000"))
XLSX_MAX_COLS_PER_SHEET = int(os.getenv("XLSX_MAX_COLS_PER_SHEET", "512"))
XLSX_SKIP_HIDDEN_SHEETS = os.getenv("XLSX_SKIP_HIDDEN_SHEETS", "true").lower() == "true"
XLSX_SAMPLE_TOP = int(os.getenv("XLSX_SAMPLE_TOP", "50"))
XLSX_SAMPLE_BOTTOM = int(os.getenv("XLSX_SAMPLE_BOTTOM", "50"))
XLSX_SCHEMA_SCAN_ROWS = int(os.getenv("XLSX_SCHEMA_SCAN_ROWS", "200"))

# OCR/VLM
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
ENABLE_VLM = os.getenv("ENABLE_VLM", "true").lower() == "true"
OCR_MIN_CHARS_TO_KEEP = int(os.getenv("OCR_MIN_CHARS_TO_KEEP", "30"))
OCR_COVERAGE_MIN_CHARS = int(os.getenv("OCR_COVERAGE_MIN_CHARS", "60"))
VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o-mini")
OCR_AVAILABLE = False
try:
    import pytesseract
    # .env에서 Tesseract 경로 설정
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
    print(f"✅ Tesseract OCR 사용 가능: {pytesseract.get_tesseract_version()}")
except Exception as e:
    if ENABLE_OCR:
        print(f"⚠️ Tesseract 미설치 또는 경로 오류: {e}")
        print("   ENABLE_OCR=false 권장 또는 TESSERACT_CMD 경로 확인")

SPARSE_INDEX_PATH = os.getenv("SPARSE_INDEX_PATH", os.path.join(CHROMA_PATH, "sparse"))

# (선택) tiktoken
try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None

SHEET_HEADER_RE = re.compile(r'^### \[Sheet\] (?P<name>.+)$', re.MULTILINE)
PAGE_MARK_RE = re.compile(r'^\[PAGE\s+(\d+)\]$', re.MULTILINE)

# ── 유틸
def _detect_office_kind(path: Path) -> Optional[str]:
    try:
        if not zipfile.is_zipfile(path): return None
        with zipfile.ZipFile(path) as z: names = set(z.namelist())
        if any(n.startswith("word/") for n in names): return "docx"
        if any(n.startswith("xl/") for n in names): return "xlsx"
        return None
    except Exception:
        return None

def _estimate_tokens(text: str) -> int:
    if _TIKTOKEN_ENC is not None:
        try: return len(_TIKTOKEN_ENC.encode(text))
        except Exception: pass
    return max(1, len(text)//4)

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def embed_texts_batched(texts: List[str]) -> List[List[float]]:
    if not texts: return []
    max_tok=int(os.getenv("EMBED_MAX_TOKENS_PER_REQUEST","280000"))
    max_it =int(os.getenv("EMBED_MAX_ITEMS_PER_REQUEST","256"))
    batches=[]; cur=[]; cur_tok=0
    for t in texts:
        tk=_estimate_tokens(t)
        if tk>max_tok:
            if cur: batches.append(cur); cur=[]; cur_tok=0
            batches.append([t]); continue
        if cur and (cur_tok+tk>max_tok or len(cur)>=max_it):
            batches.append(cur); cur=[]; cur_tok=0
        cur.append(t); cur_tok+=tk
    if cur: batches.append(cur)
    all_embeddings=[]
    for i,b in enumerate(batches,1):
        print(f"  🔎 임베딩 배치 {i}/{len(batches)} (items={len(b)})")
        r=client.embeddings.create(model="text-embedding-3-small", input=b)
        all_embeddings.extend([d.embedding for d in r.data])
    return all_embeddings

def _escape_md_cell(v: Any) -> str:
    s="" if v is None else str(v)
    return s.replace("|","\\|").replace("\n"," ").strip()

def _docx_heading_level(p: docx.text.paragraph.Paragraph) -> Optional[int]:
    try:
        name=(p.style and p.style.name) or ""
        m=re.search(r"Heading\s*([1-6])", str(name), re.IGNORECASE)
        if m: return int(m.group(1))
    except Exception: pass
    return None

def _is_pdf_heading(line: str) -> bool:
    if not line or len(line)<6 or len(line)>120: return False
    if re.match(r"^\d+(?:\.\d+)*\s+\S+", line): return True
    letters=[ch for ch in line if ch.isalpha()]
    if not letters: return False
    return (sum(1 for ch in letters if ch.isupper())/len(letters))>=0.6

# ── OCR 보조
def _extract_images_from_pdf(path: Path) -> List[Tuple[bytes, Dict[str, Any]]]:
    if fitz is None: return []
    imgs=[]
    try:
        doc=fitz.open(str(path))
        for pno in range(len(doc)):
            page=doc[pno]
            for img in page.get_images(full=True):
                xref=img[0]; base=doc.extract_image(xref)
                imgs.append((base["image"], {"page": str(pno+1)}))
        doc.close()
    except Exception: pass
    return imgs

def _extract_images_from_office_zip(path: Path) -> List[Tuple[bytes, Dict[str, Any]]]:
    imgs=[]
    try:
        if not zipfile.is_zipfile(path): return imgs
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.startswith(("word/media/","xl/media/")) and name.lower().endswith((".png",".jpg",".jpeg",".bmp",".gif",".tiff",".webp")):
                    with z.open(name) as f: imgs.append((f.read(), {"office_media": name}))
    except Exception: pass
    return imgs

def _ocr_image_bytes(img_bytes: bytes) -> str:
    if not (ENABLE_OCR and OCR_AVAILABLE and Image): return ""
    try:
        im=Image.open(io.BytesIO(img_bytes))
        return (pytesseract.image_to_string(im, lang=os.getenv("OCR_LANG","kor+eng")) or "").strip()
    except Exception as e:
        print(f"⚠️ OCR 오류: {e}"); return ""

def _vlm_caption_image(img_bytes: bytes) -> str:
    if not ENABLE_VLM: return ""
    try:
        b64=base64.b64encode(img_bytes).decode("utf-8")
        r=client.chat.completions.create(
            model=VLM_MODEL, temperature=0.2, max_tokens=160,
            messages=[
                {"role":"system","content":"짧고 검색 친화적인 한국어 캡션을 작성합니다."},
                {"role":"user","content":[
                    {"type":"text","text":"한 줄 캡션. 차트/도형이면 축과 핵심 메시지 요약."},
                    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}
                ]}
            ]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""

# ── TF-IDF
class SparseIndexer:
    def __init__(self, base_dir: str, collection: str):
        self.base_dir=Path(base_dir); self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cname=collection
        self.data_fp=self.base_dir/f"{self.cname}_docs.joblib"
        self.index_fp=self.base_dir/f"{self.cname}_tfidf.joblib"

    def _load_all(self):
        if joblib is None: return None
        if not self.data_fp.exists() or not self.index_fp.exists(): return None
        ids,docs,metas=joblib.load(self.data_fp)
        vec,mat=joblib.load(self.index_fp)
        return ids,docs,metas,vec,mat

    def upsert_and_save(self, ids_new: List[str], docs_new: List[str], metas_new: List[Dict[str,Any]]):
        if joblib is None or TfidfVectorizer is None:
            print("⚠️ TF-IDF 패키지 없음 — 스킵"); return
        prev=self._load_all()
        if prev:
            p_ids,p_docs,p_metas,_,_=prev
            mp={pid:(pdoc,pmeta) for pid,pdoc,pmeta in zip(p_ids,p_docs,p_metas)}
            for i,(nid,ndoc,nmeta) in enumerate(zip(ids_new,docs_new,metas_new)): mp[nid]=(ndoc,nmeta)
            ids=list(mp.keys()); docs=[mp[i][0] for i in ids]; metas=[mp[i][1] for i in ids]
        else:
            ids,docs,metas=ids_new,docs_new,metas_new
        vec=TfidfVectorizer(max_features=int(os.getenv("SPARSE_MAX_FEATURES","100000")), ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b", lowercase=True)
        mat=vec.fit_transform(docs)
        joblib.dump((ids,docs,metas), self.data_fp); joblib.dump((vec,mat), self.index_fp)
        print(f"✅ TF-IDF 저장: {self.index_fp} / 문서 {len(ids)}개")

# ── 시맨틱 스플리터
class SemanticTextSplitter:
    def __init__(self):
        self.enabled = SEM_SPLIT_ENABLE and SentenceTransformer is not None and st_util is not None
        self.model = SentenceTransformer(SEM_MODEL) if self.enabled else None
        if self.enabled:
            print(f"✅ Semantic splitter 사용: {SEM_MODEL}")
        elif SEM_SPLIT_ENABLE:
            print("⚠️ sentence-transformers 미설치 → 고정 청킹 사용")

    def _sent_tokenize(self, text: str) -> List[str]:
        # 문장 분리 + 최소 가비지만 제거
        sents = re.split(r"(?<=[\.!?])\s+|\n+", text.strip())
        out=[]
        for s in sents:
            s=s.strip()
            if not s: continue
            if s in ("---","—","–"): continue  # 구분선만 제거
            if re.fullmatch(r"\[PAGE\s*\d+\]", s, flags=re.I): continue  # 페이지 마커만 제거
            if re.fullmatch(r"\d+\.?", s): continue  # 단독 숫자만 제거
            if len(s)<=2: continue  # 너무 짧은 것만 제거
            out.append(s)
        return out

    def split(self, text: str) -> List[str]:
        if not self.enabled or not text.strip():
            return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n","\n"," ",""]).split_text(text)
        sents=self._sent_tokenize(text)
        if not sents: return []
        embs=self.model.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
        chunks=[]; cur=[sents[0]]; clen=len(sents[0]); ccnt=1
        for i in range(1,len(sents)):
            sim=float(st_util.cos_sim(embs[i-1], embs[i])); nxt=sents[i]
            boundary=(sim<SEM_MIN_SIM) or (clen+len(nxt)>SEM_MAX_CHARS)
            if boundary and (clen>=MIN_CHARS_PER_CHUNK and ccnt>=MIN_SENTS_PER_CHUNK):
                chunks.append("\n".join(cur)); cur=[nxt]; clen=len(nxt); ccnt=1
            else:
                cur.append(nxt); clen+=len(nxt); ccnt+=1
        if cur: chunks.append("\n".join(cur))
        if len(chunks)>=2 and len(chunks[-1])<MIN_CHARS_PER_CHUNK:
            chunks[-2]=(chunks[-2]+"\n"+chunks[-1]).strip(); chunks.pop()
        return chunks

# ── 섹션 유틸
def _allowed_docx_levels()->set:
    try:
        lvls={int(x.strip()) for x in SECTION_HEADING_LEVELS.split(",") if x.strip()}
        return {l for l in lvls if 1<=l<=6}
    except Exception:
        return {1,2,3}

def _normalize_section_body(body: str) -> str:
    lines=[ln.rstrip() for ln in body.splitlines()]
    acc=[]; buf=""
    for ln in lines:
        if not ln.strip():
            if buf: acc.append(buf.strip()); buf=""
            acc.append(""); continue
        line=ln.strip()
        if len(line)<20 and not re.search(r"[.!?]$", line):
            buf=(buf+" "+line).strip()
        else:
            if buf: acc.append((buf+" "+line).strip()); buf=""
            else: acc.append(line)
    if buf: acc.append(buf.strip())
    text="\n".join(acc)
    text=re.sub(r"\n{3,}", "\n\n", text)
    text=re.sub(r"^\s*[-–—]{3,}\s*$","",text, flags=re.M)
    text=re.sub(r"^\s*\[PAGE\s*\d+\]\s*$","",text, flags=re.M|re.I)
    return text.strip()

def _split_md_into_sections(md_text: str, source_type: str) -> List[Dict[str, Any]]:
    sections=[]; sid=0
    if source_type=="xlsx":
        matches=list(SHEET_HEADER_RE.finditer(md_text))
        if not matches:
            sid=1; sections.append({"section_id":sid,"title":"Sheet:unknown","path":"Sheet:unknown","body":_normalize_section_body(md_text),"page_range":"","sheet":"unknown"})
            return sections
        for i,m in enumerate(matches):
            title=m.group("name").strip()
            st=m.start(); ed=matches[i+1].start() if i+1<len(matches) else len(md_text)
            block=md_text[st:ed].strip(); sid+=1
            sections.append({"section_id":sid,"title":title[:SECTION_TITLE_MAX_LEN],"path":title,"body":_normalize_section_body(block),"page_range":"","sheet":title})
        return sections

    lines=md_text.splitlines()
    allowed=_allowed_docx_levels()
    cur_title=None; cur_path=[]; cur_buf=[]; cur_pages=[]
    def flush():
        nonlocal sid,cur_title,cur_path,cur_buf,cur_pages
        body="\n".join(cur_buf).strip()
        if not body and cur_title is None: return
        # 헤딩을 본문 앞에 포함 (검색 가능하도록)
        if cur_title:
            body = f"{cur_title}\n\n{body}".strip()
        pr=""
        if cur_pages:
            mn,mx=min(cur_pages),max(cur_pages); pr=f"{mn}-{mx}" if mn!=mx else f"{mn}"
        sections.append({"section_id":sid+1,"title":(cur_title or "")[:SECTION_TITLE_MAX_LEN],"path":">".join(cur_path)[:256] if cur_path else (cur_title or ""), "body":_normalize_section_body(body), "page_range":pr, "sheet":""})
        sid+=1; cur_buf.clear(); cur_pages.clear()

    for raw in lines:
        line=raw.strip()
        m=re.match(r"^\[PAGE\s+(\d+)\]$", line)
        if m:
            try: cur_pages.append(int(m.group(1)))
            except Exception: pass
            continue
        if not line:
            cur_buf.append(""); continue
        is_heading=False
        if line.startswith("#"):
            h=len(line)-len(line.lstrip("#")); rest=line[h:]
            # 마크다운 헤딩: '# 제목' (띄어쓰기 필수), 해시태그: '#키워드' (띄어쓰기 없음)
            if rest and rest[0]==' ':
                title=rest.strip()
                if 1<=h<=6 and h in allowed and title: is_heading=True; lvl=h
        if not is_heading and source_type=="pdf" and _is_pdf_heading(line):
            is_heading=True; lvl=2; title=line
        if is_heading:
            if cur_title is not None or any(x.strip() for x in cur_buf): flush()
            cur_title=title; cur_path=cur_path[:max(0,lvl-1)]; cur_path.append(title)
        else:
            cur_buf.append(line)
    if cur_title is not None or any(x.strip() for x in cur_buf): flush()
    if not sections:
        sections=[{"section_id":1,"title":"","path":"","body":_normalize_section_body(md_text),"page_range":"","sheet":""}]
    return sections

# ▶ OCR/캡션을 "페이지별"로 모아주는 함수 (핵심)
def _group_image_text_by_page(img_chunks: List[Tuple[str, Dict[str, Any]]]) -> Dict[int, str]:
    page_buckets: Dict[int, List[str]] = defaultdict(list)
    for text, meta in img_chunks:
        if not text.strip(): continue
        pg = int(meta.get("page","0")) if str(meta.get("page","")).isdigit() else 0
        page_buckets[pg].append(text.strip())
    # 같은 페이지 텍스트 결합
    return {pg: "\n\n".join(v) for pg,v in page_buckets.items()}

def _map_images_to_sections(img_by_page: Dict[int, str], sections: List[Dict[str, Any]]) -> Dict[int, int]:
    # page -> section_id
    out: Dict[int,int] = {}
    wins=[]
    for s in sections:
        pr=s.get("page_range") or ""
        try:
            if "-" in pr: a,b=pr.split("-",1); mn,mx=int(a),int(b)
            elif pr: mn=mx=int(pr)
            else: continue
            wins.append((s["section_id"],mn,mx))
        except Exception: pass
    for pg in img_by_page.keys():
        best=None; bestd=10**9
        for sid,mn,mx in wins:
            if mn<=pg<=mx: best=sid; bestd=0; break
            d=mn-pg if pg<mn else pg-mx
            if d<bestd: best,bestd=sid,d
        if best: out[pg]=best
    return out

def _merge_image_texts(sections: List[Dict[str, Any]], img_by_page: Dict[int,str], page_to_sid: Dict[int,int]):
    if not img_by_page: return
    sec_map={s["section_id"]:s for s in sections}
    for pg, txt in img_by_page.items():
        sid = page_to_sid.get(pg)
        if sid and sid in sec_map:
            s=sec_map[sid]
            if txt not in s["body"]:
                s["body"] = (s["body"] + "\n\n[이미지 텍스트]\n" + txt).strip()

# ── 섹션→청킹 (+사후 병합)
def _chunk_section_text(sec: Dict[str,Any], splitter: SemanticTextSplitter) -> List[str]:
    body=sec.get("body","").strip()
    if not body: return []
    parts = splitter.split(body)
    merged=[]; buf=[]; blen=0; bcnt=0
    def flush_buf():
        nonlocal buf, blen, bcnt
        if buf:
            merged.append("\n".join(buf).strip())
            buf=[]; blen=0; bcnt=0
    for p in parts:
        # 최소 가비지만 제거
        if re.fullmatch(r"[-–—]+", p) or re.fullmatch(r"\[PAGE\s*\d+\]", p, flags=re.I): continue
        if re.fullmatch(r"\d+\.?", p): continue
        if not p.strip(): continue
        buf.append(p.strip()); blen+=len(p); bcnt+=max(1, p.count(".")+p.count("!")+p.count("?"))
        if blen>=MIN_CHARS_PER_CHUNK and bcnt>=MIN_SENTS_PER_CHUNK: flush_buf()
        elif blen>=SEM_MAX_CHARS: flush_buf()
    flush_buf()
    if len(merged)>=2 and len(merged[-1])<MIN_CHARS_PER_CHUNK:
        merged[-2]=(merged[-2]+"\n"+merged[-1]).strip(); merged.pop()
    return merged

# ── 서비스
class IngestService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n","\n"," ",""])
        self.sem_splitter = SemanticTextSplitter()

    # 원문→Markdown (간결화)
    def _docx_to_md(self, path: Path) -> Tuple[str, Dict]:
        d=docx.Document(str(path))
        lines=[]; toc=[]; h_count=0; tbl_count=0
        for p in d.paragraphs:
            t=(p.text or "").strip()
            if not t: continue
            lvl=_docx_heading_level(p)
            if lvl:
                h_count+=1; lines.append("#"*lvl+" "+t)
                if   lvl==1: toc.append(t)
                elif lvl==2: toc.append((toc[-1]+">" if toc else "")+t)
                elif lvl==3:
                    base=toc[-1] if toc else ""; toc.append((base+">" if base else "")+t)
                continue
            lines.append(t)
        for tb in d.tables:
            rows=[]
            for row in tb.rows:
                cells=[_escape_md_cell(c.text) for c in row.cells]
                if any(cells): rows.append(cells)
            if rows:
                header=rows[0]; lines.append("")
                lines.extend(["| "+" | ".join(header)+" |","| "+" | ".join(["---"]*len(header))+" |"])
                for r in rows[1:]:
                    lines.append("| "+" | ".join(r)+" |")
                lines.append("")
        return "\n\n".join(lines).strip(), {"source_type":"docx","stats":{"headings":h_count,"tables":tbl_count,"sheets":0,"pages":0,"empty_rate":0.0},"toc":toc,"normalizer_ver":"md-v1"}

    def _pdf_to_md(self, path: Path) -> Tuple[str, Dict]:
        lines=[]; pages=0; empty=0; h_count=0
        with pdfplumber.open(str(path)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                pages+=1
                text=(page.extract_text() or "").replace("\r","\n")
                text=re.sub(r"\n{3,}","\n\n", text)
                if PDF_ADD_PAGE_MARKERS: lines.append("\n---\n[PAGE %d]\n---\n" % idx)
                if not text.strip(): empty+=1; continue
                for raw in text.split("\n"):
                    ln=raw.strip()
                    if not ln: continue
                    if PDF_HEADING_HEURISTIC and _is_pdf_heading(ln): h_count+=1; lines.append("## "+ln)
                    else: lines.append(ln)
        md="\n".join(lines).strip()
        meta={"source_type":"pdf","stats":{"headings":h_count,"tables":0,"sheets":0,"pages":pages,"empty_rate":round((empty/pages) if pages else 0.0,3)},"toc":[],"normalizer_ver":"md-v1"}
        return md, meta

    def _xlsx_to_md_schema_sample(self, path: Path) -> Tuple[str, Dict]:
        wb=openpyxl.load_workbook(str(path), data_only=True, read_only=True)
        if not wb.worksheets: raise ValueError("엑셀에 워크시트가 없습니다.")
        lines=[]; sheet_count=0; dq=deque
        for ws in wb.worksheets:
            try:
                if XLSX_SKIP_HIDDEN_SHEETS and getattr(ws,"sheet_state","visible")!="visible": continue
            except Exception: pass
            sheet_count+=1; lines.append(f"### [Sheet] {ws.title}")
            it={"values_only": True}
            if XLSX_MAX_COLS_PER_SHEET>0: it["max_col"]=XLSX_MAX_COLS_PER_SHEET
            header=None; top=[]; bottom=dq(maxlen=XLSX_SAMPLE_BOTTOM); scan=[]; idx=0
            for row in ws.iter_rows(**it):
                idx+=1; last=-1
                for i,v in enumerate(row):
                    sv=(str(v).strip() if v is not None else "")
                    if sv!="": last=i
                if last<0: continue
                r=list(row[:last+1])
                if header is None:
                    header=[str(c).strip() if c is not None else f"col{i+1}" for i,c in enumerate(r)]; continue
                if len(top)<XLSX_SAMPLE_TOP: top.append(r)
                bottom.append(r)
                if len(scan)<XLSX_SCHEMA_SCAN_ROWS: scan.append(r)
                if idx>XLSX_MAX_ROWS_PER_SHEET: break
            if header is None: lines.append("(empty sheet)"); continue
            lines.append("- columns:")
            for ci in range(len(header)):
                col=header[ci] or f"col{ci+1}"
                vals=[(r[ci] if ci<len(r) else None) for r in scan]
                non=[v for v in vals if v not in (None,"")]
                def as_type(v):
                    s=str(v)
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s): return "date"
                    if re.fullmatch(r"-?\d+$", s): return "int"
                    if re.fullmatch(r"-?\d+\.\d+", s): return "float"
                    return "text"
                inferred="text"
                if non:
                    from collections import Counter
                    inferred=Counter(as_type(v) for v in non).most_common(1)[0][0]
                null_rate=1.0-(len(non)/max(1,len(vals)))
                uniq_rate=(len(set(map(lambda x: str(x), non)))/max(1,len(non))) if non else 0.0
                lines.append(f"  - {col}: {inferred} (null {null_rate:.1%}, unique {uniq_rate:.1%})")
        md="\n".join(lines).strip()
        return md, {"source_type":"xlsx","stats":{"headings":0,"tables":0,"sheets":sheet_count,"pages":0,"empty_rate":0.0},"toc":[],"normalizer_ver":"md-v1"}

    def to_markdown(self, file_path: str, verbose: bool=True) -> Tuple[str, Dict]:
        p=Path(file_path); ext=p.suffix.lower()
        if verbose: print(f"  📝 Markdown 정규화: {p.name} ({ext})")
        actual=_detect_office_kind(p)
        try:
            if ext==".pdf": return self._pdf_to_md(p)
            if ext==".docx" or (actual=="docx" and ext!=".xlsx"): return self._docx_to_md(p)
            if ext==".xlsx" or (actual=="xlsx" and ext!=".docx"): return self._xlsx_to_md_schema_sample(p)
            if actual=="docx": return self._docx_to_md(p)
            if actual=="xlsx": return self._xlsx_to_md_schema_sample(p)
            if verbose: print(f"  ⚠️ 미지원 확장자: {ext}")
            return "", {"source_type":"unknown","stats":{},"toc":[],"normalizer_ver":"md-v1"}
        except Exception as e:
            if verbose: print(f"  ❌ Markdown 변환 오류 ({p.name}): {e}")
            return "", {"source_type":"error","stats":{},"toc":[],"normalizer_ver":"md-v1"}

    # Chroma
    def get_chroma_collection(self, collection_name: Optional[str]=None):
        name=collection_name or COLLECTION_NAME
        try:
            chroma = chromadb.CloudClient(
                tenant=os.getenv("CHROMA_TENANT"),
                database=os.getenv("CHROMA_DATABASE"),
                api_key=os.getenv("CHROMA_API_KEY"),
            )
            return chroma.get_or_create_collection(name=name)
        except Exception as e:
            print(f"Chroma Cloud 오류: {e} → 로컬 폴백")
            try:
                Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
                cli=chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False, is_persistent=True))
                return cli.get_or_create_collection(name=name)
            except Exception as e2:
                print(f"로컬 실패: {e2} → 인메모리")
                cli=chromadb.Client(); return cli.get_or_create_collection(name=name)

    def _collect_image_chunks(self, file_path: str, source_type: str) -> List[Tuple[str, Dict[str, Any]]]:
        if not ENABLE_OCR and not ENABLE_VLM: return []
        p=Path(file_path)
        blobs=[]
        if source_type=="pdf": blobs+=_extract_images_from_pdf(p)
        elif source_type in ("docx","xlsx"): blobs+=_extract_images_from_office_zip(p)
        elif p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".gif",".tiff",".webp"):
            try: blobs.append((p.read_bytes(), {}))
            except Exception: pass
        out=[]
        for img_bytes, meta in blobs:
            ocr=_ocr_image_bytes(img_bytes) if ENABLE_OCR else ""
            if ocr and len(ocr)>=OCR_MIN_CHARS_TO_KEEP:
                out.append((ocr,{**meta,"content_kind":"image_ocr"}))
            if ENABLE_VLM and (len(ocr)<OCR_COVERAGE_MIN_CHARS):
                cap=_vlm_caption_image(img_bytes)
                if cap: out.append((cap,{**meta,"content_kind":"image_vlm"}))
        return out

    def ingest_single_file_with_metadata(self, file_path: str, *, collection_name: str, extra_meta: Dict[str,Any], show_preview: bool=True) -> Tuple[int,bool]:
        print(f"📂 입력 파일: {file_path} (collection={collection_name})")
        try:
            # 1) 정규화
            if NORMALIZE_TO_MD:
                md_text, md_meta = self.to_markdown(file_path, verbose=True)
                raw_text = md_text
            else:
                raw_text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                md_meta = {"normalizer_ver":"none","source_type":"unknown"}
            if not raw_text.strip():
                print("❌ 비어있는 문서"); return 0, False
            print(f"✅ 로드 완료: {len(raw_text):,} chars")
            source_type = md_meta.get("source_type","unknown")

            # 2) 섹션 생성
            sections = _split_md_into_sections(raw_text, source_type)
            print(f"🧩 섹션: {len(sections)}개")

            # 3) OCR/캡션 수집 → 페이지별 집계
            img_chunks = self._collect_image_chunks(file_path, source_type)
            img_by_page = _group_image_text_by_page(img_chunks)
            
            print(f"🖼️ 이미지 처리: {len(img_chunks)}개 이미지, {len(img_by_page)}개 페이지")

            # 섹션이 하나도 실질 내용이 없거나 매우 적으면(이미지 PDF) → 페이지 섹션 생성
            total_body_chars = sum(len(s.get("body","")) for s in sections)
            total_img_chars = sum(len(text) for text in img_by_page.values())
            has_body = total_body_chars > 100  # 100자 이상이면 실질 내용 있음
            
            print(f"📊 텍스트 추출: {total_body_chars}자, 이미지 텍스트: {total_img_chars}자 (이미지 PDF 여부: {not has_body})")
            
            # 이미지 텍스트가 본문보다 많으면 이미지 PDF로 처리
            condition1 = not has_body and img_by_page
            condition2 = total_img_chars > total_body_chars * 1.5
            print(f"🔍 조건 체크: has_body={has_body}, img_by_page={bool(img_by_page)}, condition1={condition1}")
            print(f"🔍 조건 체크: total_img_chars={total_img_chars}, total_body_chars*1.5={total_body_chars*1.5}, condition2={condition2}")
            
            if condition1 or condition2:
                print(f"🖼️ 이미지 PDF로 처리: {len(sections)}개 섹션 → {len(img_by_page)}개 페이지 섹션")
                sections=[]
                for pg,text in sorted(img_by_page.items()):
                    sections.append({"section_id":len(sections)+1,"title":f"Page {pg}","path":f"Page {pg}",
                                     "body":_normalize_section_body(text),"page_range":str(pg),"sheet":""})
                    print(f"  📄 Page {pg}: {len(text)}자")
                img_by_page = {}  # 이미 본문으로 들어갔으니 별도 병합 안 함

            # 4) 페이지→섹션 매핑 후 본문 뒤에 병합 (이미지 PDF가 아닌 경우만)
            elif img_by_page:
                page_to_sid = _map_images_to_sections(img_by_page, sections)
                _merge_image_texts(sections, img_by_page, page_to_sid)

            # 5) 섹션 내부 시맨틱 청킹 (+사후 병합)
            chunks=[]; metas=[]
            
            # 이미지 PDF인지 확인 (페이지별 섹션인지)
            is_image_pdf = len(sections) > 1 and all(sec.get("title", "").startswith("Page ") for sec in sections)
            
            if is_image_pdf:
                print(f"🖼️ 이미지 PDF 청킹: 페이지별 독립 청크 생성")
                # 이미지 PDF: 각 페이지를 독립 청크로 처리
                for sec in sections:
                    chunks.append(sec["body"])
                    metas.append({
                        "section_id": sec["section_id"],
                        "section_title": sec.get("title") or "",
                        "section_path": sec.get("path") or "",
                        "page": sec.get("page_range") or "",
                        "sheet": sec.get("sheet") or "",
                        "content_kind": "image_text",
                        "content_type": source_type,
                    })
            else:
                # 일반 PDF: 시맨틱 청킹
                for sec in sections:
                    parts=_chunk_section_text(sec, self.sem_splitter)
                    for p in parts:
                        chunks.append(p)
                        metas.append({
                            "section_id": sec["section_id"],
                            "section_title": sec.get("title") or "",
                            "section_path": sec.get("path") or "",
                            "page": sec.get("page_range") or "",
                            "sheet": sec.get("sheet") or "",
                            "content_kind": "text",
                            "content_type": source_type,
                        })
            # ▶ 최종 안전장치: 남은 작은 조각은 앞 청크에 병합 (이미지 PDF 제외)
            if not is_image_pdf:
                compact=[]
                for c in chunks:
                    if compact and len(c)<MIN_CHARS_PER_CHUNK:
                        compact[-1]=(compact[-1]+"\n"+c).strip()
                    else:
                        compact.append(c)
                if len(compact)>=2 and len(compact[-1])<MIN_CHARS_PER_CHUNK:
                    compact[-2]=(compact[-2]+"\n"+compact[-1]).strip(); compact.pop()
                chunks=compact
            else:
                print(f"🖼️ 이미지 PDF: 병합 로직 스킵, {len(chunks)}개 청크 유지")

            # Chroma Cloud 300개 제한 체크
            MAX_CHUNKS_FOR_CLOUD = 250  # 안전 마진을 위해 250개로 제한
            if len(chunks) > MAX_CHUNKS_FOR_CLOUD:
                print(f"⚠️ 청크 수 초과: {len(chunks)}개 → {MAX_CHUNKS_FOR_CLOUD}개로 제한")
                # 상위 청크들만 선택 (더 중요한 내용 우선)
                chunks = chunks[:MAX_CHUNKS_FOR_CLOUD]
                metas = metas[:MAX_CHUNKS_FOR_CLOUD]
            
            print(f"🧱 최종 청크: {len(chunks)}개")
            if not chunks: print("❌ 청킹 결과 없음"); return 0, False
            if show_preview:
                for i,c in enumerate(chunks[:3]): print(f"  [Chunk {i}] {len(c)} chars / {c[:120]}...")

            # 6) 임베딩
            print("⚙️ 임베딩 생성...")
            embeddings=embed_texts_batched(chunks)
            if not embeddings: print("❌ 임베딩 실패"); return 0, False
            print(f"✅ 임베딩 완료: {len(embeddings)} x {len(embeddings[0])}")

            # 7) 저장(증분 삭제)
            col=self.get_chroma_collection(collection_name)
            file_name=Path(file_path).name
            try:
                existing = col.get(where={"doc_id": extra_meta.get("doc_id")}) if extra_meta.get("doc_id") else col.get(where={"source": file_name})
                if existing and existing.get("ids"): col.delete(ids=existing["ids"]); print(f"🗑 기존 {len(existing['ids'])}개 삭제")
            except Exception as e: print(f"⚠️ 기존 삭제 오류: {e}")
            base_id=Path(file_path).stem; ids=[f"{base_id}-{i}" for i in range(len(chunks))]
            file_hash=_sha256(raw_text)
            final_metas=[]
            for i,(c,m) in enumerate(zip(chunks, metas)):
                meta={
                    "source": file_name, "chunk_idx": i,
                    "content_type": m["content_type"], "content_kind": m["content_kind"],
                    "normalizer_ver": md_meta.get("normalizer_ver","none"),
                    "file_hash": file_hash, "chunk_hash": _sha256(c),
                    "chunk_tokens": _estimate_tokens(c), "overlap_used": CHUNK_OVERLAP,
                    "sheet": m["sheet"], "page": m["page"],
                    "section_id": m["section_id"], "section_title": m["section_title"], "section_path": m["section_path"],
                }
                if isinstance(extra_meta, dict): meta.update(extra_meta)
                final_metas.append(meta)
            col.add(ids=ids, metadatas=final_metas, embeddings=embeddings, documents=chunks)
            print(f"🎉 저장 완료: {len(chunks)} chunks → '{collection_name}'")

            # 8) TF-IDF
            try:
                SparseIndexer(SPARSE_INDEX_PATH, collection_name).upsert_and_save(ids, chunks, final_metas)
            except Exception as e:
                print(f"⚠️ TF-IDF 실패: {e}")

            return len(chunks), True

        except Exception as e:
            print(f"❌ 오류: {e}"); return 0, False

# ── CLI
def main():
    print("="*80); print("📚 문서 임베딩 (섹션+시맨틱+OCR 페이지 집계)"); print("="*80)
    if len(sys.argv)<2:
        print("사용법:\n  python -m app.rag.internal_data_rag.internal_ingest <파일경로> [<컬렉션명>]"); sys.exit(1)
    path=sys.argv[1]; collection=sys.argv[2] if len(sys.argv)>2 else COLLECTION_NAME
    p=Path(path)
    try:
        if p.is_file():
            svc=IngestService()
            _, ok=svc.ingest_single_file_with_metadata(str(p), collection_name=collection, extra_meta={}, show_preview=True)
            sys.exit(0 if ok else 1)
        else:
            print(f"❌ 경로 없음: {path}"); sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ 중단됨"); sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예외: {e}"); sys.exit(1)

if __name__=="__main__":
    main()
