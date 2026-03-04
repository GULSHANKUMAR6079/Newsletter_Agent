"""
app/models.py
─────────────
All Pydantic schemas and TypedDicts for the Blog Writing Agent.

Exports:
    Task, Plan, EvidenceItem, RouterDecision, EvidencePack,
    ImageSpec, GlobalImagePlan, SEOOutput, SocialContent, QualityScore,
    BlogSessionRecord, State
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Section-level schemas ──────────────────────────────────────────────────────

class Task(BaseModel):
    """A single section-writing task."""
    id: int
    title: str
    goal: str = Field(..., description="One sentence: what the reader gains from this section.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., ge=120, le=550, description="Target word count for this section.")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    """Full blog outline produced by the orchestrator."""
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


# ── Research schemas ───────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """A single piece of web research evidence."""
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD"
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    """Router node output — decides research mode and queries."""
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5, ge=1, le=10)


class EvidencePack(BaseModel):
    """Batch of EvidenceItems returned by the research synthesizer."""
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ── Image schemas ──────────────────────────────────────────────────────────────

class ImageSpec(BaseModel):
    """Spec for a single diagram/image to generate."""
    placeholder: str = Field(..., description='e.g. [[IMAGE_1]]')
    filename: str = Field(..., description='Save under images/, e.g. qkv_flow.png')
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    """Image planning output: markdown with placeholders + image specs."""
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


# ── SEO schema ─────────────────────────────────────────────────────────────────

class SEOOutput(BaseModel):
    """SEO metadata generated for the blog post."""
    slug: str = Field(..., description="URL-friendly slug, e.g. 'how-langgraph-works'")
    meta_description: str = Field(
        ..., max_length=160, description="Search engine meta description ≤160 chars."
    )
    keywords: List[str] = Field(..., min_length=5, max_length=15, description="Primary + secondary keywords.")
    estimated_reading_time_minutes: int = Field(..., ge=1)
    focus_keyword: str = Field(..., description="Single primary keyword to optimise for.")
    og_title: str = Field(..., description="Open Graph title (can differ from H1).")
    canonical_url_hint: str = Field(
        default="", description="Suggested canonical URL path, e.g. /blog/how-langgraph-works"
    )


# ── Social content schema ──────────────────────────────────────────────────────

class Tweet(BaseModel):
    """A single tweet in a thread."""
    position: int
    text: str = Field(..., max_length=280)


class SocialContent(BaseModel):
    """Social media content generated alongside the blog post."""
    linkedin_post: str = Field(
        ..., max_length=3000, description="LinkedIn post with hashtags and CTA."
    )
    twitter_thread: List[Tweet] = Field(
        ..., min_length=5, max_length=12, description="Twitter/X thread."
    )
    hashtags: List[str] = Field(default_factory=list)


# ── Quality review schema ──────────────────────────────────────────────────────

class QualityScore(BaseModel):
    """Self-critique quality review of the generated blog post."""
    overall: float = Field(..., ge=1, le=10, description="Overall quality 1–10.")
    accuracy: float = Field(..., ge=1, le=10)
    clarity: float = Field(..., ge=1, le=10)
    depth: float = Field(..., ge=1, le=10)
    originality: float = Field(..., ge=1, le=10)
    seo_friendliness: float = Field(..., ge=1, le=10)
    suggestions: List[str] = Field(
        ..., min_length=1, max_length=10,
        description="Actionable improvement suggestions."
    )
    verdict: Literal["publish", "revise", "reject"] = "publish"


# ── DB record schema ───────────────────────────────────────────────────────────

class BlogSessionRecord(BaseModel):
    """Schema for a blog session stored in PostgreSQL."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str
    blog_title: str
    mode: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    final_md: str
    seo: Optional[Dict] = None
    social: Optional[Dict] = None
    quality: Optional[Dict] = None
    embedding: Optional[List[float]] = None  # pgvector field


# ── LangGraph State ────────────────────────────────────────────────────────────

from typing import TypedDict  # noqa: E402


class State(TypedDict, total=False):
    # Input
    topic: str
    as_of: str
    recency_days: int

    # Routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]

    # Planning
    plan: Optional[Plan]

    # Workers — uses reducer to accumulate sections from parallel workers
    sections: Annotated[List[tuple[int, str]], operator.add]

    # Reducer / image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    # New post-processing fields
    seo: Optional[dict]            # SEOOutput.model_dump()
    social: Optional[dict]         # SocialContent.model_dump()
    quality: Optional[dict]        # QualityScore.model_dump()

    # Final output
    final: str

    # Session
    session_id: str
