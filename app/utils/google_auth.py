# backend/google_auth.py
import os
import json
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from dotenv import load_dotenv

load_dotenv()


def build_google_flow(credentials_path: str, scopes: list):
    """Google OAuth Flow 객체 생성 (웹 애플리케이션 방식)"""
    with open(credentials_path, "r") as f:
        client_config = json.load(f)

    web_conf = client_config.get("web")
    if not web_conf:
        raise ValueError("credentials 파일이 web 타입이 아님")

    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", web_conf["redirect_uris"][0])

    flow = Flow.from_client_config(
        client_config,
        scopes=scopes,
        redirect_uri=redirect_uri,
    )
    return flow


def exchange_code_for_tokens(flow: Flow, code: str, token_path: str):
    """콜백에서 받은 code로 토큰 교환 후 저장"""
    flow.fetch_token(code=code)
    creds = flow.credentials

    # 토큰 파일 저장 (DB 연동으로 대체 가능)
    with open(token_path, "wb") as token_file:
        pickle.dump(creds, token_file)

    return creds


def load_saved_credentials(token_path: str):
    """이미 저장된 토큰 불러오기 및 자동 갱신"""
    creds = None
    if os.path.exists(token_path):
        with open(token_path, "rb") as token_file:
            creds = pickle.load(token_file)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(token_path, "wb") as token_file:
                pickle.dump(creds, token_file)
            print(f"✅ 토큰 갱신 완료: {token_path}")
        except Exception as e:
            print(f"❌ 토큰 갱신 실패: {e}")
            creds = None
    return creds


def get_google_service_credentials(service: str, user_id: str):
    """서비스별 Google 인증 정보 반환 (tools에서 사용)"""
    from app.utils.env_loader import env_tokens

    # 서비스별 토큰 경로 매핑
    service_token_paths = {
        "google": "credentials/google_calendar_token.pickle",
        "google_drive": "credentials/google_drive_token.pickle",
    }

    token_path = service_token_paths.get(service)
    if not token_path:
        raise ValueError(f"지원하지 않는 서비스: {service}")

    # 저장된 토큰이 있으면 불러오기
    creds = load_saved_credentials(token_path)

    if creds and creds.valid:
        return creds

    # 토큰이 없거나 유효하지 않은 경우
    # OAuth 인증이 필요함을 알리는 예외 발생
    raise Exception(
        f"{service} 서비스의 유효한 토큰이 없습니다. "
        f"http://localhost:8080/auth/google/login 에서 인증을 먼저 진행해주세요."
    )
