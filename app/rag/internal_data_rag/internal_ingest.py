# app/rag/internal_data_rag/internal_ingest.py
# -*- coding: utf-8 -*-
# 문서 임베딩 및 ChromaDB 저장 서비스

import os
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pdfplumber
import docx
import openpyxl
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# ─────────────────────────────────────────────────────────
# 환경 변수 로드
# ─────────────────────────────────────────────────────────
load_dotenv()

# Chroma/Collection
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "inside_data")

# 청킹 파라미터 (필요시 .env로 조절)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))       # 청크 크기
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # 오버랩

# 엑셀 파일 처리 제한 설정
XLSX_MAX_ROWS_PER_SHEET = int(os.getenv("XLSX_MAX_ROWS_PER_SHEET", "10000"))
XLSX_MAX_COLS_PER_SHEET = int(os.getenv("XLSX_MAX_COLS_PER_SHEET", "512"))
XLSX_SKIP_HIDDEN_SHEETS = os.getenv("XLSX_SKIP_HIDDEN_SHEETS", "true").lower() == "true"

# 임베딩 API 요청 배치 제한
EMBED_MAX_TOKENS_PER_REQUEST = int(os.getenv("EMBED_MAX_TOKENS_PER_REQUEST", "280000"))
EMBED_MAX_ITEMS_PER_REQUEST = int(os.getenv("EMBED_MAX_ITEMS_PER_REQUEST", "256"))

# tiktoken은 선택적
try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None

# OpenAI
client = OpenAI()


# ─────────────────────────────────────────────────────────
# 유틸: 실제 Office Open XML 포맷 스니핑(.docx/.xlsx 구분)
# ─────────────────────────────────────────────────────────
def _detect_office_kind(path: Path) -> Optional[str]:
    """
    ZIP 기반 Office 문서의 실제 종류를 추정:
      - 'docx'  : word/document.xml 존재
      - 'xlsx'  : xl/workbook.xml 존재
      - None    : ZIP 아님 또는 Office OpenXML 아님
    """
    try:
        if not zipfile.is_zipfile(path):
            return None
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
        if any(n.startswith("word/") for n in names):
            return "docx"
        if any(n.startswith("xl/") for n in names):
            return "xlsx"
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# 임베딩 배치 유틸
# ─────────────────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    """임베딩 토큰 대략치. tiktoken 있으면 정확, 없으면 문자수/4 근사."""
    if _TIKTOKEN_ENC is not None:
        try:
            return len(_TIKTOKEN_ENC.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def embed_texts_batched(texts: List[str]) -> List[List[float]]:
    """토큰/아이템 예산을 지켜가며 여러 번으로 나눠 임베딩."""
    if not texts:
        return []

    batches: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0

    for t in texts:
        tk = _estimate_tokens(t)

        # 단일 청크가 예산을 넘더라도(거의 없지만) 단독 배치로 보냄
        if tk > EMBED_MAX_TOKENS_PER_REQUEST:
            if current:
                batches.append(current)
                current, current_tokens = [], 0
            batches.append([t])
            continue

        if current and (
            current_tokens + tk > EMBED_MAX_TOKENS_PER_REQUEST
            or len(current) >= EMBED_MAX_ITEMS_PER_REQUEST
        ):
            batches.append(current)
            current, current_tokens = [], 0

        current.append(t)
        current_tokens += tk

    if current:
        batches.append(current)

    all_embeddings: List[List[float]] = []
    for i, batch in enumerate(batches, 1):
        print(f"  🔎 임베딩 배치 {i}/{len(batches)} (items={len(batch)}) 요청 중...")
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_embeddings.extend([d.embedding for d in resp.data])

    return all_embeddings


# ─────────────────────────────────────────────────────────
# 서비스 클래스
# ─────────────────────────────────────────────────────────
class IngestService:
    """문서 임베딩 및 ChromaDB 저장을 담당하는 서비스 클래스"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        # 지원되는 파일 확장자
        self.supported_extensions = {".pdf", ".docx", ".xlsx", ".csv", ".txt"}

    # ========================= 파일 파싱 =========================
    def read_pdf(self, path: Path) -> str:  # PDF 파일 파싱
        texts = []
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t.strip():
                        texts.append(t)
        except Exception as e:
            raise ValueError(f"PDF 로드 실패: {type(e).__name__}: {e}")
        return "\n\n".join(texts)

    def read_docx(self, path: Path) -> str:  # DOCX 파일 파싱
        try:
            d = docx.Document(str(path))
        except Exception as e:
            raise ValueError(f"DOCX 로드 실패: {type(e).__name__}: {e}")
        acc: List[str] = []
        acc.extend([p.text for p in d.paragraphs if p.text and p.text.strip()])
        # 테이블 추출(간단)
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    acc.append(" | ".join(cells))
        return "\n".join(acc)

    def read_xlsx(self, path: Path) -> str:  # XLSX 파일 파싱 (폭주 방지 트리밍/캡 적용)
        wb = None
        try:
            wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
            
            if not wb.worksheets:
                raise ValueError("엑셀에 워크시트가 없습니다.")

            acc: List[str] = []
            for ws in wb.worksheets:
                # 숨김 시트 스킵 옵션
                try:
                    if XLSX_SKIP_HIDDEN_SHEETS and getattr(ws, "sheet_state", "visible") != "visible":
                        continue
                except Exception:
                    pass

                acc.append(f"\n### [Sheet] {ws.title}")
                rows = 0

                # 열 상한 캡을 openpyxl 레벨에서 바로 적용
                iter_kwargs = {"values_only": True}
                if XLSX_MAX_COLS_PER_SHEET and XLSX_MAX_COLS_PER_SHEET > 0:
                    iter_kwargs["max_col"] = XLSX_MAX_COLS_PER_SHEET

                for row in ws.iter_rows(**iter_kwargs):
                    if rows >= XLSX_MAX_ROWS_PER_SHEET:
                        acc.append(f"...(truncated at {XLSX_MAX_ROWS_PER_SHEET} rows)")
                        break

                    # 행 우측의 빈 열 트리밍: 실제 값이 있는 마지막 열까지만 사용
                    last = -1
                    # (열 캡이 적용된 범위 내에서만 검사)
                    for i, v in enumerate(row):
                        sv = (str(v).strip() if v is not None else "")
                        if sv != "":
                            last = i

                    if last < 0:
                        continue  # 완전 빈 행은 스킵

                    # 최종 사용할 열 폭 결정
                    width = last + 1
                    if XLSX_MAX_COLS_PER_SHEET and XLSX_MAX_COLS_PER_SHEET > 0:
                        width = min(width, XLSX_MAX_COLS_PER_SHEET)

                    # 최종 문자열 구성
                    row_vals = []
                    for v in row[:width]:
                        row_vals.append("" if v is None else str(v).strip())

                    acc.append(" | ".join(row_vals))
                    rows += 1

            return "\n".join(acc)
            
        except Exception as e:
            # 암호화/손상/비정상 구조 등 명확한 메시지 전달
            raise ValueError(f"엑셀 로드 실패: {type(e).__name__}: {e}")
        finally:
            # Excel 워크북 명시적으로 닫기 (임시 파일 정리 문제 해결)
            if wb is not None:
                try:
                    wb.close()
                except Exception:
                    pass

    def read_csv(self, path: Path) -> str:  # CSV 파일 파싱
        """CSV 파일을 텍스트로 변환 (테이블 형태 유지)"""
        import csv
        try:
            acc: List[str] = []
            with open(path, 'r', encoding='utf-8-sig', newline='') as f:
                # CSV 방언 자동 감지 시도
                try:
                    sample = f.read(2048)
                    f.seek(0)
                    dialect = csv.Sniffer().sniff(sample)
                    reader = csv.reader(f, dialect)
                except Exception:
                    # 감지 실패 시 기본 설정 사용
                    f.seek(0)
                    reader = csv.reader(f)
                
                for row_num, row in enumerate(reader):
                    if row_num > 10000:  # 행 수 제한 (메모리 보호)
                        acc.append("...(truncated at 10000 rows)")
                        break
                    
                    # 빈 행 스킵
                    if not any(cell.strip() for cell in row):
                        continue
                    
                    # 테이블 형태로 파이프 구분자 사용
                    acc.append(" | ".join(str(cell).strip() for cell in row))
                    
            return "\n".join(acc)
            
        except UnicodeDecodeError:
            # UTF-8 실패 시 다른 인코딩 시도
            try:
                with open(path, 'r', encoding='cp949', newline='') as f:
                    reader = csv.reader(f)
                    acc = []
                    for row_num, row in enumerate(reader):
                        if row_num > 10000:
                            acc.append("...(truncated at 10000 rows)")
                            break
                        if not any(cell.strip() for cell in row):
                            continue
                        acc.append(" | ".join(str(cell).strip() for cell in row))
                    return "\n".join(acc)
            except Exception as e:
                raise ValueError(f"CSV 인코딩 로드 실패: {type(e).__name__}: {e}")
        except Exception as e:
            raise ValueError(f"CSV 로드 실패: {type(e).__name__}: {e}")

    def read_txt(self, path: Path) -> str:  # 일반 텍스트 파일 파싱
        """일반 텍스트 파일 읽기 (다양한 인코딩 지원)"""
        encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                    if content.strip():  # 빈 파일이 아니면 성공
                        return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                raise ValueError(f"텍스트 파일 로드 실패: {type(e).__name__}: {e}")
        
        # 모든 인코딩 실패
        raise ValueError(f"텍스트 파일 인코딩을 감지할 수 없습니다: {path}")

    def load_text(self, file_path: str, verbose: bool = True) -> str:
        """확장자 + 실제 포맷 스니핑으로 적절한 파서 선택"""
        p = Path(file_path)
        ext = p.suffix.lower()

        if verbose:
            print(f"  📄 파일 파싱 중: {p.name} ({ext})")

        actual = _detect_office_kind(p)  # 실제 포맷 스니핑(ZIP 기반 Office 문서의 실제 종류를 추정)

        try:
            if ext == ".pdf":   # PDF 파일 파싱
                return self.read_pdf(p)

            if ext == ".docx" or (actual == "docx" and ext != ".xlsx"):  # DOCX 파일 파싱
                if verbose and ext != ".docx" and actual == "docx":
                    print(" ⚠️ 확장자와 다른 실제 포맷(docx) 감지 → docx 파서 사용")
                return self.read_docx(p)

            if ext == ".xlsx" or (actual == "xlsx" and ext != ".docx"):  # XLSX 파일 파싱
                if verbose and ext != ".xlsx" and actual == "xlsx":
                    print("  ⚠️ 확장자와 다른 실제 포맷(xlsx) 감지 → xlsx 파서 사용")
                return self.read_xlsx(p)

            if ext == ".csv":  # CSV 파일 파싱
                return self.read_csv(p)

            if ext == ".txt":  # 텍스트 파일 파싱
                return self.read_txt(p)

            # 마지막 보루: 실제 포맷 기준 시도
            if actual == "docx":
                if verbose:
                    print("  ⚠️ 확장자 미지원/불명이나 실제 포맷(docx) 감지 → docx 파서 사용")
                return self.read_docx(p)
            if actual == "xlsx":
                if verbose:
                    print("  ⚠️ 확장자 미지원/불명이나 실제 포맷(xlsx) 감지 → xlsx 파서 사용")
                return self.read_xlsx(p)

            if verbose:
                print(f"  ⚠️ 지원하지 않는 파일 형식: {ext} (실제 포맷 미확인)")
            return ""

        except Exception as e:
            if verbose:
                print(f"  ❌ 파일 읽기 오류 ({p.name}): {e}")
            return ""

    # ========================= Chroma 헬퍼 =========================
    def get_chroma_collection(self, collection_name: Optional[str] = None):
        """
        (수정) 회사 코드별로 컬렉션을 분리하기 위해 collection_name 주입 허용.
        - collection_name 이 None이면 기존 환경변수 COLLECTION_NAME 사용.
        """
        name = collection_name or COLLECTION_NAME
        try:
            # ChromaDB 디렉토리 확인 및 생성
            Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
            chroma = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                ),
            )
            return chroma.get_or_create_collection(name=name)
        except Exception as e:
            print(f"ChromaDB 초기화 오류: {str(e)}")
            print("새로운 ChromaDB 인스턴스로 재시도 중...")
            chroma = chromadb.Client()
            return chroma.get_or_create_collection(name=name)

    # ========================= (신규) 외부 메타 병합 + 컬렉션 지정 =========================
    def ingest_single_file_with_metadata(
        self,
        file_path: str,
        *,
        collection_name: str,
        extra_meta: Dict[str, Any],
        show_preview: bool = True
    ) -> Tuple[int, bool]:
        """
        (신규) 파일 하나를 인덱싱하면서, 각 청크의 메타데이터에 extra_meta 를 병합하여 저장.
        - collection_name : 회사 코드(예: 'CAESAR2024') → 회사별 컬렉션 분리
        - extra_meta      : {'doc_id': int, 'company_id': int, 'user_id': Optional[int], 'is_private': bool}
        - return          : (chunks_count, success_flag)
        """
        print(f"📂 입력 파일: {file_path} (collection={collection_name})")
        try:
            # 1) 파일 로드 및 검증
            raw_text = self.load_text(file_path, verbose=False)
            if not raw_text.strip():
                print(f"❌ 빈 파일이거나 읽기 실패: {file_path}")
                return 0, False

            print(f"✅ 파일 로드 완료, 전체 길이: {len(raw_text):,} chars")

            # 2) 텍스트 청킹
            chunks = self.text_splitter.split_text(raw_text)

            # 각 청크의 텍스트 길이 출력(옵션)
            if show_preview:
                for i, c in enumerate(chunks[:3]):
                    print(f"  [Chunk {i}] {len(c):,} chars / preview: {c[:100]}...")

            print(f"🪓 청킹 완료 → 총 {len(chunks)} chunks")
            if not chunks:
                print("❌ 청킹 결과가 비어 있습니다.")
                return 0, False

            # 3) 임베딩 생성
            print("⚙️ 임베딩 생성 중...")
            embeddings = embed_texts_batched(chunks)
            if not embeddings:
                print("❌ 임베딩 생성 실패(빈 입력).")
                return 0, False
            print(f"✅ 임베딩 완료 → shape: {len(embeddings)} x {len(embeddings[0])}")

            # 4) 회사 코드 컬렉션으로 저장
            collection = self.get_chroma_collection(collection_name)

            # 📝 기존 동일 문서 청크 삭제(doc_id 기반으로 중복 방지)
            file_name = Path(file_path).name
            try:
                # doc_id가 있으면 해당 문서의 청크만 삭제, 없으면 파일명으로 삭제
                if extra_meta and "doc_id" in extra_meta:
                    existing = collection.get(where={"doc_id": extra_meta["doc_id"]})
                else:
                    existing = collection.get(where={"source": file_name})
                
                if existing and existing.get("ids"):
                    collection.delete(ids=existing["ids"])
                    print(f"🗑 기존 {len(existing['ids'])} 청크 삭제")
            except Exception as e:
                print(f"⚠️ 기존 청크 삭제 중 오류: {e}")
                pass

            # 새 데이터 추가
            base_id = Path(file_path).stem
            ids = [f"{base_id}-{i}" for i in range(len(chunks))]

            # 기존 메타 유지 + extra_meta 병합
            metadatas = []
            for i in range(len(chunks)):
                m = {
                    "source": file_name,       # 기존 메타
                    "chunk_idx": i,            # 기존 메타
                }
                if isinstance(extra_meta, dict):
                    m.update(extra_meta)       # ← 병합: doc_id/company_id/user_id/is_private
                metadatas.append(m)

            collection.add(
                ids=ids,
                metadatas=metadatas,
                embeddings=embeddings,
                documents=chunks,
            )

            print(f"🎉 완료! {len(chunks)} chunks → Chroma collection '{collection_name}' 저장")
            return len(chunks), True

        except Exception as e:
            print(f"❌ 파일 처리 중 오류 발생: {str(e)}")
            return 0, False

    # 기존 단일 파일 처리 메서드는 더 이상 사용하지 않음 (ingest_single_file_with_metadata 사용)
    # 다중 파일 처리 기능은 관리자 업로드에서 사용하지 않으므로 제거

    # 불필요한 편의 함수 제거 - IngestService 클래스를 직접 사용

# ========================= CLI =========================
def main():
    print("=" * 80)
    print("📚 문서 임베딩 서비스")
    print("=" * 80)

    if len(sys.argv) < 2:
        print("사용법:")
        print("  테스트용 파일 처리: python ingest_service.py <파일경로>")
        print("\n📝 실제 관리자 업로드는 /api/admin/files/upload 엔드포인트를 사용하세요.")
        print("\n예시:")
        print("  python ingest_service.py ./storage/data/document.pdf")
        sys.exit(1)

    path = sys.argv[1]

    try:
        path_obj = Path(path)

        if path_obj.is_file():  # 단일 파일 처리
            print("📄 단일 파일 모드")
            # 개별 파일 테스트용 - 실제 관리자 업로드는 file_ingest_service.py 사용
            svc = IngestService()
            success = svc.ingest_single_file_with_metadata(
                str(path_obj),
                collection_name=COLLECTION_NAME,
                extra_meta={},
                show_preview=True
            )
            sys.exit(0 if success[1] else 1)

        else:
            print(f"❌ 경로가 존재하지 않습니다: {path}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n❌ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
