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

def put_file_and_checksum(file_bytes: bytes, *, orig_name: str) -> tuple[str, int, str]:
    """
    파일 바이트를 S3에 업로드하고, URL/사이즈/체크섬을 반환한다.
    - file_bytes     : 파일 바이트
    - orig_name   : 원본 파일명 (확장자/Content-Type 추정)
    return        : (s3_url, size, checksum_sha256)
    """
    size = len(file_bytes)
    checksum = sha256_bytes(file_bytes)
    ext = ""
    if "." in orig_name:
        ext = "." + orig_name.split(".")[-1].lower()
    key = f"uploads/{uuid.uuid4().hex}{ext}"
    content_type, _ = mimetypes.guess_type(orig_name)
    s3.put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=content_type or "application/octet-stream",
    )
    url = f"https://{settings.S3_BUCKET}.s3.{settings.S3_REGION}.amazonaws.com/{key}"
    return url, size, checksum

def delete_object_by_url(s3_url: str) -> None:
    """
    S3 URL을 key로 환산하여 삭제한다. (관리자 삭제 시 사용)
    """
    prefix = f"https://{settings.S3_BUCKET}.s3.{settings.S3_REGION}.amazonaws.com/"
    if not s3_url.startswith(prefix):
        return
    key = s3_url[len(prefix):]
    s3.delete_object(Bucket=settings.S3_BUCKET, Key=key)
