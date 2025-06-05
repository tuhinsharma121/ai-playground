import os

from dotenv import load_dotenv
from pydantic import (
    SecretStr,
)
from pydantic_settings import BaseSettings

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

    # SnowflakeConfiguration/Snowflake配置
    SNOWFLAKE_ACCOUNT: str = os.getenv("SNOWFLAKE_ACCOUNT", None)

    # Configuration - equivalent to your Node.js constants
    SSO_CLIENT_ID: str = os.getenv("SSO_CLIENT_ID", None)
    SSO_CLIENT_SECRET: str = os.getenv("SSO_CLIENT_SECRET", None)

    # OAuth2 Configuration - equivalent to your credentials config
    AUTHORIZATION_BASE_URL: str = f'https://{SNOWFLAKE_ACCOUNT}.snowflakecomputing.com/oauth/authorize'
    TOKEN_URL: str = f'https://{SNOWFLAKE_ACCOUNT}.snowflakecomputing.com/oauth/token-request'


constants = Constants()
