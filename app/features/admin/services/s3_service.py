# app/features/admin/services/s3_service.py
import boto3, uuid, hashlib, mimetypes
from app.core.config import settings

# boto3 S3 클라이언트
s3 = boto3.client(
    "s3",
    region_name=settings.S3_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
)

def sha256_bytes(data: bytes) -> str:
    """파일 바이트 → SHA256 hex"""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

# 텍스트 UTF-8 정규화 유틸
_POSSIBLE_ENCODINGS = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]

def _normalize_text_bytes(orig_bytes: bytes) -> tuple[bytes, str]:
    """
    텍스트로 보이는 바이트를 UTF-8 로 재인코딩.
    성공하면 (정규화된 bytes, 'utf-8') 반환, 실패하면 (원본 bytes, '') 반환.
    """
    for enc in _POSSIBLE_ENCODINGS:
        try:
            s = orig_bytes.decode(enc)
            return s.encode("utf-8"), "utf-8"
        except Exception:
            continue
    try:
        import chardet  # optional
        det = chardet.detect(orig_bytes or b"")
        enc = (det.get("encoding") or "").lower()
        if enc:
            s = orig_bytes.decode(enc, errors="strict")
            return s.encode("utf-8"), "utf-8"
    except Exception:
        pass
    return orig_bytes, ""

def _ensure_metadata_charset(bucket: str, key: str, want_ct: str):
    """
    S3 객체가 이미 있을 때 ContentType 에 charset 이 없으면 메타데이터만 교체.
    (S3 는 메타 변경을 위해 copy_object + MetadataDirective='REPLACE' 가 필요)
    """
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        current_ct = (head.get("ContentType") or "").lower()
        if "charset=" in current_ct:
            return
        # 메타데이터 교체
        s3.copy_object(
            Bucket=bucket,
            Key=key,
            CopySource={"Bucket": bucket, "Key": key},
            ContentType=want_ct,
            MetadataDirective="REPLACE",
        )
    except Exception:
        # 실패해도 업로드 자체에는 영향 없음
        pass

# ── CSV 엑셀 호환 유틸: UTF-8 BOM + CRLF 개행으로 보정
def _prepare_csv_for_excel(utf8_bytes: bytes) -> bytes:
    """
    Excel에서 한글이 깨지지 않도록:
      - 줄바꿈을 CRLF(\r\n)로 통일
      - UTF-8 BOM(0xEF 0xBB 0xBF) 추가
    """
    text = utf8_bytes.decode("utf-8")
    # 기존 CRLF → LF 정규화 후 다시 CRLF로
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    b = text.encode("utf-8")
    BOM = b"\xef\xbb\xbf"
    return b if b.startswith(BOM) else (BOM + b)

# 내용 주소화(content-addressable) 업로드 — 동일 내용이면 물리 파일 재사용
def put_file_if_absent(file_bytes: bytes, *, orig_name: str, checksum_hex: str) -> tuple[str, int, str, bool]:
    """
    파일 바이트를 '내용 해시(checksum_hex)'를 키로 사용하여 S3에 저장한다.
    이미 존재하면 업로드를 건너뛰고 기존 객체를 재사용한다.
    return: (s3_url, size, checksum_hex, uploaded_new)
    """
    # ── 확장자/콘텐츠 타입 추정
    ext = ""
    if "." in orig_name:
        ext = "." + orig_name.split(".")[-1].lower()
    content_type, _ = mimetypes.guess_type(orig_name)
    ct = content_type or "application/octet-stream"

    # ── 텍스트면 UTF-8 정규화 + charset
    text_like_exts = {".txt", ".csv", ".md", ".log"}
    is_text_like = (ct.startswith("text/")) or (ext in text_like_exts)

    body = file_bytes
    if is_text_like:
        # 1) 우선 UTF-8로 정규화
        body, charset = _normalize_text_bytes(file_bytes)

        # 2) CSV라면 Excel 호환 보정(BOM + CRLF)
        if ext == ".csv":
            body = _prepare_csv_for_excel(body)
            # mimetypes 가 text/plain 으로 추정하는 경우도 있으므로 text/csv 로 교정
            if not ct.startswith("text/csv"):
                ct = "text/csv"

        # 3) Content-Type에 charset이 없다면 붙임
        if charset and "charset=" not in ct:
            ct = f"{ct}; charset={charset}"

    # ❗핵심: 정규화/보정된 바이트로 해시를 다시 계산해서 키 생성
    checksum_final = sha256_bytes(body)
    key = f"uploads/{checksum_final}{ext}"  # ← 정규화 이후 내용 기반 키

    size = len(body)

    # 이미 존재하는지 HEAD로 검사
    try:
        s3.head_object(Bucket=settings.S3_BUCKET, Key=key)
        uploaded_new = False
        # 존재하지만 charset 이 없다면 메타데이터만 교체
        if is_text_like:
            _ensure_metadata_charset(settings.S3_BUCKET, key, ct)
    except s3.exceptions.ClientError:
        s3.put_object(
            Bucket=settings.S3_BUCKET,
            Key=key,
            Body=body,            # ← 정규화/보정된 바이트 업로드
            ContentType=ct,       # ← charset 포함
        )
        uploaded_new = True

    url = f"https://{settings.S3_BUCKET}.s3.{settings.S3_REGION}.amazonaws.com/{key}"
    # 반환 checksum 은 "정규화/보정 이후 바이트 기준"으로 돌려준다.
    return url, size, checksum_final, uploaded_new

def delete_object_by_url(s3_url: str) -> None:
    """
    S3 URL을 key로 환산하여 삭제한다. (관리자 삭제 시 사용)
    """
    prefix = f"https://{settings.S3_BUCKET}.s3.{settings.S3_REGION}.amazonaws.com/"
    if not s3_url.startswith(prefix):
        return
    key = s3_url[len(prefix):]
    s3.delete_object(Bucket=settings.S3_BUCKET, Key=key)
