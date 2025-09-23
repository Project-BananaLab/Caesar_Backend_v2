# security.py

import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Union

from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 환경 변수 로드 ---
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

# 비밀번호 해싱을 위한 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 데이터 암호화를 위한 Fernet 인스턴스 생성
# ENCRYPTION_KEY가 없으면 에러 발생
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY가 .env 파일에 설정되지 않았습니다.")
fernet = Fernet(ENCRYPTION_KEY.encode())

# --- 비밀번호 관련 함수 ---
def verify_password(plain_password, hashed_password):
    """일반 비밀번호와 해시된 비밀번호를 비교합니다."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """비밀번호를 해시하여 반환합니다."""
    return pwd_context.hash(password)

# --- JWT 토큰 관련 함수 ---
def create_access_token(data: dict):
    """JWT 액세스 토큰을 생성합니다."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- 데이터 암호화/복호화 함수 ---
def encrypt_data(data: Union[str, Dict[Any, Any]]) -> bytes:
    """문자열 또는 JSON(dict) 데이터를 암호화합니다."""
    # 데이터가 dict 타입이면 JSON 문자열로 변환
    if isinstance(data, dict):
        data_str = json.dumps(data)
    else:
        data_str = str(data)
    
    # UTF-8로 인코딩 후 암호화
    return fernet.encrypt(data_str.encode('utf-8'))

def decrypt_data(encrypted_data: bytes) -> Union[str, Dict[Any, Any], None]:
    """암호화된 데이터를 복호화합니다."""
    if not encrypted_data:
        return None
    try:
        # 데이터 복호화 후 UTF-8로 디코딩
        decrypted_str = fernet.decrypt(encrypted_data).decode('utf-8')
        
        # JSON 형태로 변환 시도
        try:
            return json.loads(decrypted_str)
        except json.JSONDecodeError:
            # JSON 변환 실패 시 일반 문자열로 반환
            return decrypted_str
            
    except InvalidToken:
        # 토큰이 유효하지 않은 경우 (데이터가 손상되었거나 키가 다른 경우)
        print("오류: 암호화된 데이터가 유효하지 않습니다.")
        return None