"""
Application configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Central configuration for the TruthLens backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    app_name: str = "TruthLens API"
    app_version: str = "0.1.0"
    debug: bool = False

    # LLM Configuration (Groq + Llama 3)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # CORS
    frontend_origin: str = "http://localhost:5173"


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
