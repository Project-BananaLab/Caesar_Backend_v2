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

# 내용 주소화(content-addressable) 업로드 — 동일 내용이면 물리 파일 재사용
def put_file_if_absent(file_bytes: bytes, *, orig_name: str, checksum_hex: str) -> tuple[str, int, str, bool]:
    """
    파일 바이트를 '내용 해시(checksum_hex)'를 키로 사용하여 S3에 저장한다.
    이미 존재하면 업로드를 건너뛰고 기존 객체를 재사용한다.
    return: (s3_url, size, checksum_hex, uploaded_new)
    """
    size = len(file_bytes)
    ext = ""
    if "." in orig_name:
        ext = "." + orig_name.split(".")[-1].lower()

    key = f"uploads/{checksum_hex}{ext}"  # ← 핵심: 내용 기반 키
    content_type, _ = mimetypes.guess_type(orig_name)

    # 이미 존재하는지 HEAD로 검사
    try:
        s3.head_object(Bucket=settings.S3_BUCKET, Key=key)
        uploaded_new = False
    except s3.exceptions.ClientError:
        s3.put_object(
            Bucket=settings.S3_BUCKET,
            Key=key,
            Body=file_bytes,
            ContentType=content_type or "application/octet-stream",
        )
        uploaded_new = True

    url = f"https://{settings.S3_BUCKET}.s3.{settings.S3_REGION}.amazonaws.com/{key}"
    return url, size, checksum_hex, uploaded_new

def delete_object_by_url(s3_url: str) -> None:
    """
    S3 URL을 key로 환산하여 삭제한다. (관리자 삭제 시 사용)
    """
    prefix = f"https://{settings.S3_BUCKET}.s3.{settings.S3_REGION}.amazonaws.com/"
    if not s3_url.startswith(prefix):
        return
    key = s3_url[len(prefix):]
    s3.delete_object(Bucket=settings.S3_BUCKET, Key=key)
