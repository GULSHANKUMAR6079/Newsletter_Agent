"""
tests/test_models.py
────────────────────
Unit tests for Pydantic schema validation.
"""

import pytest
from pydantic import ValidationError

from app.models import (
    Task, Plan, EvidenceItem, RouterDecision,
    SEOOutput, SocialContent, QualityScore, BlogSessionRecord, Tweet,
)


# ── Task ──────────────────────────────────────────────────────────────────────

class TestTask:
    def test_valid_task(self):
        t = Task(
            id=1,
            title="Introduction",
            goal="Understand what LangGraph is.",
            bullets=["What is LangGraph", "Why it matters", "Key concepts"],
            target_words=250,
        )
        assert t.id == 1
        assert t.requires_code is False

    def test_task_requires_code_flag(self):
        t = Task(
            id=2,
            title="Code examples",
            goal="Show a working LangGraph example.",
            bullets=["Setup", "State definition", "Graph compilation"],
            target_words=400,
            requires_code=True,
        )
        assert t.requires_code is True

    def test_task_too_few_bullets(self):
        with pytest.raises(ValidationError):
            Task(
                id=1, title="X", goal="Y",
                bullets=["only one bullet"],
                target_words=200,
            )

    def test_task_word_count_too_low(self):
        with pytest.raises(ValidationError):
            Task(
                id=1, title="X", goal="Y",
                bullets=["A", "B", "C"],
                target_words=50,  # below 120 minimum
            )

    def test_task_word_count_too_high(self):
        with pytest.raises(ValidationError):
            Task(
                id=1, title="X", goal="Y",
                bullets=["A", "B", "C"],
                target_words=999,  # above 550 maximum
            )


# ── Plan ──────────────────────────────────────────────────────────────────────

class TestPlan:
    def _make_task(self, id_: int) -> Task:
        return Task(
            id=id_, title=f"Section {id_}",
            goal=f"Goal {id_}",
            bullets=["A", "B", "C"],
            target_words=300,
        )

    def test_valid_plan(self):
        plan = Plan(
            blog_title="LangGraph Deep Dive",
            audience="ML Engineers",
            tone="Technical",
            tasks=[self._make_task(i) for i in range(1, 6)],
        )
        assert plan.blog_kind == "explainer"
        assert len(plan.tasks) == 5

    def test_plan_news_roundup_kind(self):
        plan = Plan(
            blog_title="Weekly AI News",
            audience="AI Enthusiasts",
            tone="Casual",
            blog_kind="news_roundup",
            tasks=[self._make_task(i) for i in range(1, 4)],
        )
        assert plan.blog_kind == "news_roundup"


# ── EvidenceItem ──────────────────────────────────────────────────────────────

class TestEvidenceItem:
    def test_valid_evidence(self):
        e = EvidenceItem(
            title="LangGraph Paper",
            url="https://example.com/langgraph",
            snippet="A framework for stateful agents.",
        )
        assert e.url.startswith("https://")
        assert e.published_at is None

    def test_evidence_with_date(self):
        e = EvidenceItem(
            title="Blog Post",
            url="https://blog.example.com/post-1",
            published_at="2025-01-15",
        )
        assert e.published_at == "2025-01-15"


# ── RouterDecision ────────────────────────────────────────────────────────────

class TestRouterDecision:
    def test_closed_book(self):
        r = RouterDecision(
            needs_research=False,
            mode="closed_book",
            reason="Evergreen topic",
        )
        assert r.needs_research is False
        assert r.queries == []

    def test_mode_validation(self):
        with pytest.raises(ValidationError):
            RouterDecision(
                needs_research=True,
                mode="invalid_mode",  # not in Literal
                reason="test",
            )


# ── SEOOutput ─────────────────────────────────────────────────────────────────

class TestSEOOutput:
    def test_valid_seo(self):
        seo = SEOOutput(
            slug="how-langgraph-works",
            meta_description="Learn how LangGraph builds stateful AI agents. Complete guide with code examples.",
            keywords=["langgraph", "ai agents", "stateful agents"],
            estimated_reading_time_minutes=7,
            focus_keyword="langgraph stateful agents",
            og_title="LangGraph: Build Stateful AI Agents",
        )
        assert len(seo.meta_description) <= 160

    def test_meta_description_too_long(self):
        with pytest.raises(ValidationError):
            SEOOutput(
                slug="test",
                meta_description="x" * 161,  # exceeds 160 chars
                keywords=["kw1"],
                estimated_reading_time_minutes=5,
                focus_keyword="test",
                og_title="Test title",
            )


# ── QualityScore ──────────────────────────────────────────────────────────────

class TestQualityScore:
    def test_valid_score(self):
        q = QualityScore(
            overall=8.5,
            accuracy=9.0,
            clarity=8.0,
            depth=8.5,
            originality=7.5,
            seo_friendliness=8.0,
            suggestions=["Add code example to section 3"],
            verdict="publish",
        )
        assert q.verdict == "publish"

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            QualityScore(
                overall=11.0,  # > 10
                accuracy=9.0, clarity=8.0, depth=8.5,
                originality=7.5, seo_friendliness=8.0,
                suggestions=["test"],
                verdict="publish",
            )

    def test_verdict_values(self):
        for verdict in ("publish", "revise", "reject"):
            q = QualityScore(
                overall=7.0, accuracy=7.0, clarity=7.0,
                depth=7.0, originality=7.0, seo_friendliness=7.0,
                suggestions=["ok"], verdict=verdict,
            )
            assert q.verdict == verdict


# ── SocialContent ─────────────────────────────────────────────────────────────

class TestSocialContent:
    def test_valid_social(self):
        tweets = [Tweet(position=i, text=f"Tweet {i} text here.") for i in range(1, 6)]
        s = SocialContent(
            linkedin_post="Here is a great LinkedIn post about AI agents! #AI #LangGraph",
            twitter_thread=tweets,
        )
        assert len(s.twitter_thread) >= 5

    def test_tweet_too_long(self):
        with pytest.raises(ValidationError):
            Tweet(position=1, text="x" * 281)  # exceeds 280 chars


# ── BlogSessionRecord ─────────────────────────────────────────────────────────

class TestBlogSessionRecord:
    def test_defaults(self):
        r = BlogSessionRecord(
            topic="LangGraph",
            blog_title="Guide to LangGraph",
            mode="closed_book",
            final_md="# Guide\n\nContent here.",
        )
        assert r.id is not None
        assert r.created_at is not None
        assert r.embedding is None
