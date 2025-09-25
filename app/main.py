# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import os
import json
import requests
from app.agents.agent import run_agent
from app.utils.db import (
    get_user_tokens,
    save_user_tokens,
    get_service_token,
    Base,
    engine,
)
from app.core.config import settings
from dotenv import load_dotenv

from app.agents.routers.agent_router import router as agent_router
from app.features.login.company.routers import router as company_login_router
from app.features.admin.routers.files import router as admin_files_router
from app.features.employee_google.employee import router as employee_router
from app.features.chat.router.chat import router as chat_router
from app.features.channel.router.channel import router as channel_router

load_dotenv()

app = FastAPI(
    title="Multi-Service Agent API",
    description="Google Calendar, Drive, Slack, Notion을 통합 관리하는 AI Agent API",
    version="1.0.0",
)

# ✅ 개발환경: 프론트 도메인만 명시 (와일드카드 X)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

def add_cors_middleware(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,  # ← * 대신 정확한 Origin
        allow_credentials=True,  # 쿠키/인증 허용
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],  # 필요시 좁혀도 됨
        expose_headers=["*"],
    )

add_cors_middleware(app)

@app.on_event("startup")
def on_startup():
    # 서버 시작 시 테이블 생성 (이미 있으면 Skip)
    Base.metadata.create_all(bind=engine)

# 라우터 등록
app.include_router(agent_router)
app.include_router(employee_router)
app.include_router(company_login_router)  # 회사 로그인
app.include_router(admin_files_router)      # 회사(관리자) 문서 업로드/목록/삭제
app.include_router(chat_router)
app.include_router(channel_router)

# Google OAuth 설정 로드
try:
    with open("google_auth/gcp-oauth.keys.json") as f:
        google_conf = json.load(f)["web"]

    CLIENT_ID = google_conf["client_id"]
    CLIENT_SECRET = google_conf["client_secret"]
    REDIRECT_URI = google_conf["redirect_uris"][0]  # 첫 번째 redirect URI 사용
    AUTH_URI = google_conf["auth_uri"]
    TOKEN_URI = google_conf["token_uri"]

    SCOPES = [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/drive",
    ]

    print(f"✅ Google OAuth 설정 로드 완료")
    print(f"📍 Redirect URI: {REDIRECT_URI}")

except FileNotFoundError:
    print(
        "❌ Google OAuth 설정 파일을 찾을 수 없습니다: google_auth/gcp-oauth.keys.json"
    )
    CLIENT_ID = CLIENT_SECRET = REDIRECT_URI = AUTH_URI = TOKEN_URI = None
    SCOPES = []
except Exception as e:
    print(f"❌ Google OAuth 설정 로드 실패: {e}")
    CLIENT_ID = CLIENT_SECRET = REDIRECT_URI = AUTH_URI = TOKEN_URI = None
    SCOPES = []


# 요청 모델
class TokenRequest(BaseModel):
    user_id: str
    service: str
    tokens: Dict[str, Any]


class TokenResponse(BaseModel):
    user_id: str
    tokens: Dict[str, Any]


# 환경변수에서 OpenAI API 키 가져오기 (기본값)
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@app.get("/")
async def root():
    """API 상태 확인"""
    # 토큰 파일 존재 여부 확인
    import os

    calendar_token_exists = os.path.exists("credentials/google_calendar_token.pickle")
    drive_token_exists = os.path.exists("credentials/google_drive_token.pickle")

    return {
        "message": "Multi-Service Agent API가 실행 중입니다",
        "services": ["Google Calendar", "Google Drive", "Slack", "Notion"],
        "status": "running",
        "oauth_available": CLIENT_ID is not None,
        "redirect_uri": REDIRECT_URI,
        "google_auth_status": {
            "calendar_token": "✅ 있음" if calendar_token_exists else "❌ 없음",
            "drive_token": "✅ 있음" if drive_token_exists else "❌ 없음",
            "auth_needed": not (calendar_token_exists and drive_token_exists),
        },
        "auth_links": {
            "login": "http://localhost:8080/auth/google/login",
            "status": "http://localhost:8080/",
        },
    }


@app.get("/auth/google/login")
async def google_login():
    """Google OAuth 인증 시작"""
    if not CLIENT_ID:
        raise HTTPException(
            status_code=500, detail="Google OAuth 설정이 로드되지 않았습니다."
        )

    # URL 인코딩을 위한 import
    from urllib.parse import quote

    # 스코프를 URL 인코딩
    scopes_encoded = quote(" ".join(SCOPES))
    redirect_uri_encoded = quote(REDIRECT_URI)

    # Google OAuth 인증 URL 생성
    auth_url = (
        f"{AUTH_URI}?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={redirect_uri_encoded}"
        f"&scope={scopes_encoded}"
        f"&access_type=offline"
        f"&prompt=consent"
        f"&state=oauth_login"
    )

    print(f"🔗 Google 인증 URL 생성:")
    print(f"   Client ID: {CLIENT_ID[:20]}...")
    print(f"   Redirect URI: {REDIRECT_URI}")
    print(f"   Scopes: {SCOPES}")
    print(f"   Full URL: {auth_url}")

    return RedirectResponse(url=auth_url)


@app.get("/auth/google/callback")
async def google_callback(
    code: str = None, user_id: str = "user_123", error: str = None, state: str = None
):
    """Google OAuth 콜백 처리"""
    print(f"📥 콜백 수신:")
    print(f"   Code: {code[:20] + '...' if code else 'None'}")
    print(f"   Error: {error}")
    print(f"   State: {state}")
    print(f"   User ID: {user_id}")

    if not CLIENT_ID:
        raise HTTPException(
            status_code=500, detail="Google OAuth 설정이 로드되지 않았습니다."
        )

    # OAuth 오류 체크
    if error:
        return {
            "status": "error",
            "error": error,
            "message": f"Google 인증 중 오류가 발생했습니다: {error}",
            "redirect_to_login": "http://localhost:8080/auth/google/login",
        }

    # 인증 코드 체크
    if not code:
        return {
            "status": "error",
            "message": "인증 코드가 제공되지 않았습니다. Google 인증을 다시 시도해주세요.",
            "redirect_to_login": "http://localhost:8080/auth/google/login",
        }

    try:
        # 인증 코드로 토큰 교환
        token_data = {
            "code": code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        }

        print(f"🔄 토큰 교환 요청: {TOKEN_URI}")
        resp = requests.post(TOKEN_URI, data=token_data)

        if resp.status_code != 200:
            print(f"❌ 토큰 교환 실패: {resp.status_code} - {resp.text}")
            raise HTTPException(status_code=400, detail=f"토큰 교환 실패: {resp.text}")

        tokens = resp.json()
        print(f"✅ 토큰 교환 성공: {user_id}")

        # 토큰 저장 (DB와 pickle 파일 모두)
        save_user_tokens(user_id, "google", tokens)

        # Google API 클라이언트에서 사용할 수 있도록 pickle 파일로도 저장
        import pickle
        from google.oauth2.credentials import Credentials

        # OAuth 토큰을 Google Credentials 객체로 변환
        creds = Credentials(
            token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
            token_uri=TOKEN_URI,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            scopes=SCOPES,
        )

        # Calendar와 Drive 토큰 모두 저장 (같은 OAuth 앱 사용)
        with open("credentials/google_calendar_token.pickle", "wb") as f:
            pickle.dump(creds, f)
        with open("credentials/google_drive_token.pickle", "wb") as f:
            pickle.dump(creds, f)

        print(f"✅ Pickle 토큰 파일 저장 완료")

        return {
            "status": "success",
            "message": f"Google 인증이 완료되었습니다. 사용자 ID: {user_id}",
            "user_id": user_id,
            "tokens_saved": True,
        }

    except requests.RequestException as e:
        print(f"❌ 네트워크 오류: {e}")
        raise HTTPException(status_code=500, detail=f"네트워크 오류: {str(e)}")
    except Exception as e:
        print(f"❌ 토큰 교환 중 오류: {e}")
        raise HTTPException(
            status_code=500, detail=f"토큰 교환 중 오류가 발생했습니다: {str(e)}"
        )


# Agent 관련 엔드포인트는 app/agents/routers/agent_router.py로 이동됨


@app.get("/tokens/{user_id}")
async def get_tokens(user_id: str):
    """사용자 토큰 조회"""
    try:
        tokens = get_user_tokens(user_id)
        if not tokens:
            raise HTTPException(
                status_code=404,
                detail=f"사용자 '{user_id}'의 토큰 정보를 찾을 수 없습니다.",
            )

        # 민감한 정보 마스킹
        masked_tokens = {}
        for service, token_data in tokens.items():
            masked_tokens[service] = {}
            for key, value in token_data.items():
                if isinstance(value, str) and len(value) > 10:
                    # 토큰 값 마스킹 (앞 4자리 + *** + 뒤 4자리)
                    masked_tokens[service][key] = f"{value[:4]}***{value[-4:]}"
                else:
                    masked_tokens[service][key] = value

        return TokenResponse(user_id=user_id, tokens=masked_tokens)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"토큰 조회 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/tokens")
async def save_tokens(request: TokenRequest):
    """사용자 토큰 저장"""
    try:
        save_user_tokens(
            user_id=request.user_id, service=request.service, tokens=request.tokens
        )

        return {
            "status": "success",
            "message": f"사용자 '{request.user_id}'의 {request.service} 토큰이 저장되었습니다.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"토큰 저장 중 오류가 발생했습니다: {str(e)}"
        )


@app.get("/tokens/{user_id}/{service}")
async def get_service_tokens(user_id: str, service: str):
    """특정 서비스 토큰 조회"""
    try:
        tokens = get_service_token(user_id, service)
        if not tokens:
            raise HTTPException(
                status_code=404,
                detail=f"사용자 '{user_id}'의 {service} 토큰을 찾을 수 없습니다.",
            )

        # 토큰 값 마스킹
        masked_tokens = {}
        for key, value in tokens.items():
            if isinstance(value, str) and len(value) > 10:
                masked_tokens[key] = f"{value[:4]}***{value[-4:]}"
            else:
                masked_tokens[key] = value

        return {"user_id": user_id, "service": service, "tokens": masked_tokens}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"토큰 조회 중 오류가 발생했습니다: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "message": "서버가 정상적으로 작동 중입니다"}


@app.get("/services")
async def list_services():
    """지원하는 서비스 목록"""
    return {
        "services": [
            {
                "name": "Google Calendar",
                "description": "일정 관리 및 조회",
                "tools": [
                    "list_calendar_events",
                    "create_calendar_event",
                    "update_calendar_event",
                    "delete_calendar_event",
                ],
            },
            {
                "name": "Google Drive",
                "description": "파일 관리 및 공유",
                "tools": [
                    "list_drive_files",
                    "create_drive_folder",
                    "share_drive_file",
                    "rename_drive_file",
                    "delete_drive_file",
                    "upload_drive_file",
                ],
            },
            {
                "name": "Slack",
                "description": "메시지 전송 및 채널 관리",
                "tools": [
                    "send_slack_message",
                    "list_slack_channels",
                    "get_slack_messages",
                    "update_slack_message",
                    "delete_slack_message",
                ],
            },
            {
                "name": "Notion",
                "description": "페이지 관리 및 검색",
                "tools": [
                    "list_notion_content",
                    "create_notion_page",
                    "get_notion_content",
                    "update_notion_page",
                    "delete_notion_page",
                ],
            },
        ]
    }


if __name__ == "__main__":
    import uvicorn

    # redirect_uris와 일치하는 8080 포트에서 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
