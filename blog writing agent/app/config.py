"""
app/config.py
─────────────
Centralised application configuration loaded from environment variables.

Usage:
    from app.config import cfg
    print(cfg.gemini_model)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    # ── LLM ──────────────────────────────────────────────────────────────────
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    gemini_image_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-preview-image-generation")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    )

    # ── API Keys ──────────────────────────────────────────────────────────────
    google_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "")
    )
    tavily_api_key: str = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "")
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/blogwriter",
        )
    )
    db_pool_size: int = field(
        default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "10"))
    )
    db_max_overflow: int = field(
        default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "20"))
    )

    # ── LLM Behaviour ─────────────────────────────────────────────────────────
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    retry_wait_min: float = field(
        default_factory=lambda: float(os.getenv("RETRY_WAIT_MIN", "2"))
    )
    retry_wait_max: float = field(
        default_factory=lambda: float(os.getenv("RETRY_WAIT_MAX", "30"))
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )

    # ── Image Generation ──────────────────────────────────────────────────────
    image_quality: str = field(
        default_factory=lambda: os.getenv("IMAGE_QUALITY", "medium")
    )
    max_images_per_blog: int = field(
        default_factory=lambda: int(os.getenv("MAX_IMAGES_PER_BLOG", "3"))
    )

    # ── Research ──────────────────────────────────────────────────────────────
    max_research_results: int = field(
        default_factory=lambda: int(os.getenv("MAX_RESEARCH_RESULTS", "6"))
    )
    max_evidence_items: int = field(
        default_factory=lambda: int(os.getenv("MAX_EVIDENCE_ITEMS", "20"))
    )

    # ── Blog Generation ───────────────────────────────────────────────────────
    min_tasks: int = 5
    max_tasks: int = 9
    min_section_words: int = 120
    max_section_words: int = 550

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = field(
        default_factory=lambda: os.getenv("OUTPUT_DIR", "output")
    )
    images_dir: str = "images"

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "Blog Writing Agent"
    app_version: str = "1.0.0"
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )

    def validate(self) -> None:
        """Raise if critical config is missing."""
        if not self.google_api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_database(self) -> bool:
        return bool(self.database_url)


# Singleton instance — import this everywhere
cfg = AppConfig()
