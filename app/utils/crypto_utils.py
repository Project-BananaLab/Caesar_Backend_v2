import os
import json
from typing import Union, Dict, Any
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 환경 변수에서 키 로드
key = os.getenv("ENCRYPTION_KEY")
if not key:
    raise ValueError("❌ ENCRYPTION_KEY not found in environment variables.")
f = Fernet(key.encode())


def encrypt_data(data: Union[str, Dict[str, Any], list, int, float, bool]) -> bytes:
    """
    문자열 또는 JSON 직렬화 가능한 데이터를 암호화하여 bytes 반환 (BLOB 저장용).
    """
    if isinstance(data, str):
        plain_text = data
    else:
        plain_text = json.dumps(data, ensure_ascii=False)

    return f.encrypt(plain_text.encode("utf-8"))  # bytes 그대로 반환


def decrypt_data(encrypted_data: Union[bytes, memoryview, None], return_type: str = "auto") -> Union[str, Dict[str, Any], list, None]:
    """
    암호화된 데이터를 복호화합니다.

    Args:
        encrypted_data: 암호화된 데이터 (bytes, memoryview, BLOB에서 가져온 값)
        return_type: 'string', 'json', 'auto'
    """
    if not encrypted_data:
        return None

    # memoryview나 다른 타입을 bytes로 변환
    if isinstance(encrypted_data, memoryview):
        encrypted_data = encrypted_data.tobytes()
    elif not isinstance(encrypted_data, bytes):
        # 다른 타입의 경우 bytes로 변환 시도
        try:
            encrypted_data = bytes(encrypted_data)
        except (TypeError, ValueError):
            raise ValueError(f"❌ Unsupported data type for decryption: {type(encrypted_data)}")

    try:
        decrypted_string = f.decrypt(encrypted_data).decode("utf-8")
    except InvalidToken:
        raise ValueError("❌ Invalid or corrupted encrypted data.")
    except Exception as e:
        raise ValueError(f"❌ Decryption failed: {str(e)}")

    if return_type == "string":
        return decrypted_string
    elif return_type == "json":
        return json.loads(decrypted_string)
    else:  # auto
        try:
            return json.loads(decrypted_string)
        except json.JSONDecodeError:
            return decrypted_string