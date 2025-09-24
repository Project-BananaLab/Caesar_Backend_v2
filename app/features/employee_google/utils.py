# utils.py
# 민감한 데이터(API 키)를 암호화하고 복호화하는 함수를 제공합니다.

import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# .env 파일에 저장된 암호화 키를 가져옵니다.
# 이 키는 반드시 32바이트여야 하며, 한 번 설정한 후에는 변경하면 안 됩니다.
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
cipher_suite = Fernet(ENCRYPTION_KEY.encode())

def encrypt_data(data: str) -> bytes:
    """
    문자열 데이터를 암호화하여 바이트로 반환합니다.
    :param data: 암호화할 문자열
    :return: 암호화된 바이트 데이터
    """
    if not data:
        return None
    # 문자열을 바이트로 인코딩한 후 암호화합니다.
    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data

def decrypt_data(encrypted_data: bytes) -> str:
    """
    암호화된 바이트 데이터를 복호화하여 문자열로 반환합니다.
    :param encrypted_data: 복호화할 바이트 데이터
    :return: 복호화된 문자열
    """
    if not encrypted_data:
        return None
    # 데이터를 복호화한 후 바이트를 문자열로 디코딩합니다.
    decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
    return decrypted_data
