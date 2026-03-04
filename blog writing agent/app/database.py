"""
app/database.py
───────────────
PostgreSQL persistence layer using SQLAlchemy async ORM + pgvector.

Features:
- Async engine with connection pooling (pool_size=10, max_overflow=20)
- pgvector integration for semantic blog search via text-embedding-004
- Graceful fallback: if DB is unreachable, all methods return silently
  (sessions still work in-memory via st.session_state)

Tables:
    blog_sessions — one row per generated blog session

Usage:
    from app.database import db
    await db.save_session(record)
    sessions = await db.list_sessions()
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, text
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import cfg
from app.models import BlogSessionRecord
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── ORM Base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class BlogSessionORM(Base):
    """SQLAlchemy ORM model for blog_sessions table."""
    __tablename__ = "blog_sessions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    topic = Column(String(1024), nullable=False, index=True)
    blog_title = Column(String(512), nullable=False)
    mode = Column(String(32), nullable=False, default="closed_book")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    final_md = Column(Text, nullable=False)
    seo = Column(JSONB, nullable=True)
    social = Column(JSONB, nullable=True)
    quality = Column(JSONB, nullable=True)
    # pgvector — stored as TEXT (JSON array) for maximum compatibility;
    # swap to Vector(768) when pgvector extension is confirmed available
    embedding_json = Column(Text, nullable=True)

    def to_record(self) -> BlogSessionRecord:
        emb = None
        if self.embedding_json:
            try:
                emb = json.loads(self.embedding_json)
            except Exception:
                emb = None
        return BlogSessionRecord(
            id=str(self.id),
            topic=self.topic,
            blog_title=self.blog_title,
            mode=self.mode,
            created_at=self.created_at,
            final_md=self.final_md,
            seo=self.seo,
            social=self.social,
            quality=self.quality,
            embedding=emb,
        )


# ── Engine & session factory ───────────────────────────────────────────────────

def _create_engine():
    return create_async_engine(
        cfg.database_url,
        pool_size=cfg.db_pool_size,
        max_overflow=cfg.db_max_overflow,
        echo=cfg.debug,
        pool_pre_ping=True,          # detect stale connections
        pool_recycle=3600,           # recycle connections every hour
    )


# ── Database manager ──────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Async PostgreSQL database manager.
    All public methods fail silently if the DB is unreachable
    (log warning, return default value).
    """

    def __init__(self) -> None:
        self._engine = None
        self._session_factory = None
        self._ready = False

    async def init(self) -> None:
        """Initialise the engine, create tables, enable pgvector."""
        try:
            self._engine = _create_engine()
            self._session_factory = async_sessionmaker(
                self._engine, expire_on_commit=False, class_=AsyncSession
            )
            async with self._engine.begin() as conn:
                # Enable pgvector extension (no-op if already installed)
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                # Create tables
                await conn.run_sync(Base.metadata.create_all)
            self._ready = True
            logger.info("Database initialised (PostgreSQL + pgvector)")
        except Exception as exc:
            logger.warning(
                "Database unavailable — sessions will not be persisted. (%s)", exc
            )
            self._ready = False

    async def _session(self):
        if not self._ready or self._session_factory is None:
            return None
        return self._session_factory()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def save_session(self, record: BlogSessionRecord) -> bool:
        """Persist a blog session. Returns True on success."""
        factory = self._session_factory
        if not self._ready or factory is None:
            return False
        try:
            emb_json = json.dumps(record.embedding) if record.embedding else None
            row = BlogSessionORM(
                id=record.id,
                topic=record.topic,
                blog_title=record.blog_title,
                mode=record.mode,
                created_at=record.created_at,
                final_md=record.final_md,
                seo=record.seo,
                social=record.social,
                quality=record.quality,
                embedding_json=emb_json,
            )
            async with factory() as session:
                session.add(row)
                await session.commit()
            logger.info("Session saved | id=%s | title='%s'", record.id, record.blog_title)
            return True
        except Exception as exc:
            logger.warning("Failed to save session: %s", exc)
            return False

    async def list_sessions(self, limit: int = 50) -> List[BlogSessionRecord]:
        """Return most recent sessions, newest first."""
        factory = self._session_factory
        if not self._ready or factory is None:
            return []
        try:
            from sqlalchemy import select
            async with factory() as session:
                result = await session.execute(
                    select(BlogSessionORM)
                    .order_by(BlogSessionORM.created_at.desc())
                    .limit(limit)
                )
                rows = result.scalars().all()
            return [r.to_record() for r in rows]
        except Exception as exc:
            logger.warning("Failed to list sessions: %s", exc)
            return []

    async def load_session(self, session_id: str) -> Optional[BlogSessionRecord]:
        """Load a single session by ID."""
        factory = self._session_factory
        if not self._ready or factory is None:
            return None
        try:
            from sqlalchemy import select
            async with factory() as session:
                result = await session.execute(
                    select(BlogSessionORM).where(BlogSessionORM.id == session_id)
                )
                row = result.scalar_one_or_none()
            return row.to_record() if row else None
        except Exception as exc:
            logger.warning("Failed to load session %s: %s", session_id, exc)
            return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID. Returns True on success."""
        factory = self._session_factory
        if not self._ready or factory is None:
            return False
        try:
            from sqlalchemy import delete as sql_delete
            async with factory() as session:
                await session.execute(
                    sql_delete(BlogSessionORM).where(BlogSessionORM.id == session_id)
                )
                await session.commit()
            logger.info("Session deleted | id=%s", session_id)
            return True
        except Exception as exc:
            logger.warning("Failed to delete session %s: %s", session_id, exc)
            return False

    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[BlogSessionRecord]:
        """
        Semantic search using pgvector cosine similarity.
        Requires pgvector extension + embedding_json column.
        Returns top_k most similar past sessions.
        """
        factory = self._session_factory
        if not self._ready or factory is None:
            return []
        try:
            # Raw SQL using pgvector <=> operator (cosine distance)
            vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            raw_sql = text(
                f"""
                SELECT id, topic, blog_title, mode, created_at, final_md,
                       seo, social, quality, embedding_json
                FROM blog_sessions
                WHERE embedding_json IS NOT NULL
                ORDER BY embedding_json::vector <=> '{vec_str}'::vector
                LIMIT :top_k
                """
            )
            async with factory() as session:
                result = await session.execute(raw_sql, {"top_k": top_k})
                rows = result.fetchall()

            records = []
            for row in rows:
                records.append(
                    BlogSessionRecord(
                        id=str(row.id),
                        topic=row.topic,
                        blog_title=row.blog_title,
                        mode=row.mode,
                        created_at=row.created_at,
                        final_md=row.final_md,
                        seo=row.seo,
                        social=row.social,
                        quality=row.quality,
                    )
                )
            return records
        except Exception as exc:
            logger.warning("Semantic search failed: %s", exc)
            return []

    async def close(self) -> None:
        """Dispose the engine and close all connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database engine disposed.")


# ── Module-level singleton ────────────────────────────────────────────────────

db = DatabaseManager()
