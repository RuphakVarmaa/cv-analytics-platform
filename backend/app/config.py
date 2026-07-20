from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional


class Settings(BaseSettings):
    ML_SERVICE_URL: str = "http://cv-ml-service.fly.dev"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/cv_analytics"
    REDIS_URL: str = "redis://localhost:6379"
    JWT_SECRET: str = "change-me-in-production-use-a-long-random-secret"
    JWT_ALGORITHM: str = "HS256"
    MAX_CONCURRENT_SESSIONS: int = 10
    QUEUE_MAXSIZE: int = 30
    DEMO_VIDEO_PATH: str = "/app/demo/demo.mp4"
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: str = "*"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        # asyncpg requires postgresql:// not postgres://
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql://", 1)
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": True}


settings = Settings()
