# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ───────── Database ─────────
    DB_URL: str

    # ───────── JWT ─────────
    JWT_SECRET_KEY: str
    ALGORITHM: str
    ACCESS_EXPIRES_MIN: int = 30

    # ───────── S3 ─────────
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    S3_REGION: str
    S3_BUCKET: str

    # ───────── ChromaDB ─────────
    CHROMA_PATH: str

    # ───────── OpenAI ─────────
    OPENAI_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
