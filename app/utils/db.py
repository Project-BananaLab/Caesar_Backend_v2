# backend/db.py
from app.utils.env_loader import env_tokens

# 사용자별 OAuth 토큰 저장소
user_tokens = {
    "user_123": {
        "google": env_tokens["google"],
        "slack": env_tokens["slack"],
        "notion": env_tokens["notion"],
    }
}


def get_user_tokens(user_id: str) -> dict:
    """사용자 ID로 토큰 정보 조회"""
    return user_tokens.get(user_id, {})


def save_user_tokens(user_id: str, service: str, tokens: dict):
    """사용자 토큰 저장"""
    if user_id not in user_tokens:
        user_tokens[user_id] = {}
    user_tokens[user_id][service] = tokens


def get_service_token(user_id: str, service: str) -> dict:
    """특정 서비스의 토큰 조회"""
    return user_tokens.get(user_id, {}).get(service, {})
