import os

from pydantic import (
    SecretStr,
)
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()



class Constants(BaseSettings):
    LOG_LEVEL: str | None = os.getenv("PYTHON_LOG_LEVEL", "INFO")
    AUTH_SECRET: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None

    # PostgreSQL Configuration
    POSTGRES_USER: str | None = os.getenv("POSTGRES_USER", None)
    POSTGRES_PASSWORD: SecretStr | None = os.getenv("POSTGRES_PASSWORD", None)
    POSTGRES_HOST: str | None = os.getenv("POSTGRES_HOST", "0.0.0.0")
    POSTGRES_PORT: int | None = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str | None = os.getenv("POSTGRES_DB", None)

    # Langfuse
    LANGFUSE_TRACING_ENVIRONMENT: str | None = os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "production")


constants = Constants()
