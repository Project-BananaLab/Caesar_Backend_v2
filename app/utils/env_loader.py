from dotenv import load_dotenv
import os
import json

load_dotenv()  # .env 파일 로드

# LangSmith 설정 (선택적)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT", "caesar-agent")

if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = langchain_project
    print(f"🔍 LangSmith 추적 활성화: {langchain_project}")
else:
    print("💡 LangSmith 추적 비활성화 (.env에 LANGCHAIN_API_KEY 추가하면 활성화)")

# Google credentials 파일 경로
GOOGLE_CALENDAR_CREDENTIALS_PATH = "google_auth/google_calendar_credentials.json"
GOOGLE_CALENDAR_TOKEN_PATH = "google_auth/google_calendar_token.pickle"
GOOGLE_DRIVE_TOKEN_PATH = "google_auth/google_drive_token.pickle"
GCP_OAUTH_KEYS_PATH = "google_auth/gcp-oauth.keys.json"


# Google OAuth 클라이언트 정보 로드
def load_google_credentials():
    """Google OAuth 클라이언트 자격 증명 로드"""
    try:
        # Calendar credentials 파일에서 로드 (기본)
        with open(GOOGLE_CALENDAR_CREDENTIALS_PATH, "r") as f:
            cred_data = json.load(f)
            return cred_data["web"]
    except FileNotFoundError:
        try:
            # GCP OAuth keys 파일에서 로드 (대안)
            with open(GCP_OAUTH_KEYS_PATH, "r") as f:
                cred_data = json.load(f)
                return cred_data["web"]
        except FileNotFoundError:
            print("Google credentials 파일을 찾을 수 없습니다.")
            return {}


google_creds = load_google_credentials()

env_tokens = {
    "google": {
        "credentials_path": GOOGLE_CALENDAR_CREDENTIALS_PATH,
        "token_path": GOOGLE_CALENDAR_TOKEN_PATH,
        "client_id": google_creds.get("client_id"),
        "client_secret": google_creds.get("client_secret"),
        "token_uri": google_creds.get(
            "token_uri", "https://oauth2.googleapis.com/token"
        ),
        "scopes": [
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/drive",
        ],
    },
    "google_drive": {
        "credentials_path": GOOGLE_CALENDAR_CREDENTIALS_PATH,  # 동일한 OAuth 앱 사용
        "token_path": GOOGLE_DRIVE_TOKEN_PATH,
        "client_id": google_creds.get("client_id"),
        "client_secret": google_creds.get("client_secret"),
        "token_uri": google_creds.get(
            "token_uri", "https://oauth2.googleapis.com/token"
        ),
        "scopes": [
            "https://www.googleapis.com/auth/drive",
        ],
    },
    "slack": {
        "bot_token": os.getenv("SLACK_BOT_TOKEN"),
        "user_token": os.getenv("SLACK_USER_TOKEN"),
    },
    "notion": {"token": os.getenv("NOTION_TOKEN")},
}
