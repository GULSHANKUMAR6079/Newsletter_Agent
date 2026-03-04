"""
tests/test_graph_nodes.py
─────────────────────────
Unit tests for graph node logic using mocked LLM calls.
All LLM calls are patched — no real API calls are made.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.models import (
    EvidenceItem, Plan, QualityScore, RouterDecision, SEOOutput,
    SocialContent, Task, Tweet,
)


# ── router_node ───────────────────────────────────────────────────────────────

class TestRouterNode:
    def _make_state(self):
        return {
            "topic": "How LangGraph works",
            "as_of": "2025-02-27",
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "seo": None,
            "social": None,
            "quality": None,
            "final": "",
            "session_id": "test-123",
        }

    @patch("app.graph.router.llm")
    def test_router_closed_book(self, mock_llm):
        decision = RouterDecision(
            needs_research=False, mode="closed_book", reason="Evergreen topic"
        )
        mock_llm.with_structured_output.return_value.invoke.return_value = decision

        from app.graph.router import router_node
        result = router_node(self._make_state())

        assert result["needs_research"] is False
        assert result["mode"] == "closed_book"
        assert result["recency_days"] == 3650

    @patch("app.graph.router.llm")
    def test_router_open_book(self, mock_llm):
        decision = RouterDecision(
            needs_research=True,
            mode="open_book",
            reason="News topic",
            queries=["latest AI news", "AI weekly"],
        )
        mock_llm.with_structured_output.return_value.invoke.return_value = decision

        from app.graph.router import router_node
        result = router_node(self._make_state())

        assert result["needs_research"] is True
        assert result["mode"] == "open_book"
        assert result["recency_days"] == 7
        assert len(result["queries"]) == 2

    def test_route_next_research(self):
        from app.graph.router import route_next
        assert route_next({"needs_research": True}) == "research"

    def test_route_next_orchestrator(self):
        from app.graph.router import route_next
        assert route_next({"needs_research": False}) == "orchestrator"


# ── research_node ─────────────────────────────────────────────────────────────

class TestResearchNode:
    def _make_state(self):
        return {
            "topic": "LangGraph",
            "as_of": "2025-02-27",
            "mode": "hybrid",
            "needs_research": True,
            "queries": ["LangGraph tutorial", "LangGraph vs CrewAI"],
            "evidence": [],
            "plan": None,
            "recency_days": 45,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "seo": None,
            "social": None,
            "quality": None,
            "final": "",
            "session_id": "test-123",
        }

    @patch("app.graph.research._tavily_search", return_value=[])
    def test_no_results_returns_empty_evidence(self, mock_search):
        from app.graph.research import research_node
        result = research_node(self._make_state())
        assert result == {"evidence": []}

    @patch("app.graph.research.llm")
    @patch("app.graph.research._tavily_search")
    def test_returns_evidence_items(self, mock_search, mock_llm):
        from app.models import EvidencePack
        mock_search.return_value = [
            {"title": "LangGraph Guide", "url": "https://example.com/lg", "snippet": "A guide."}
        ]
        pack = EvidencePack(evidence=[
            EvidenceItem(title="LangGraph Guide", url="https://example.com/lg", snippet="A guide.")
        ])
        mock_llm.with_structured_output.return_value.invoke.return_value = pack

        from app.graph.research import research_node
        result = research_node(self._make_state())
        assert len(result["evidence"]) == 1
        assert result["evidence"][0].url == "https://example.com/lg"


# ── SEO node ──────────────────────────────────────────────────────────────────

class TestSEONode:
    def _make_state(self, final_md="# Test Blog\n\n" + " ".join(["word"] * 500)):
        task = Task(id=1, title="Intro", goal="G", bullets=["A","B","C"], target_words=250)
        plan = Plan(blog_title="Test Blog", audience="Devs", tone="Technical", tasks=[task])
        return {
            "topic": "Test", "as_of": "2025-02-27", "mode": "closed_book",
            "needs_research": False, "queries": [], "evidence": [], "plan": plan,
            "recency_days": 3650, "sections": [], "merged_md": "", "final": final_md,
            "md_with_placeholders": "", "image_specs": [], "seo": None,
            "social": None, "quality": None, "session_id": "t",
        }

    def test_seo_skipped_when_no_final(self):
        from app.graph.seo import seo_node
        state = self._make_state(final_md="")
        result = seo_node(state)
        assert result == {"seo": None}

    @patch("app.graph.seo.llm")
    def test_seo_returns_dict(self, mock_llm):
        seo = SEOOutput(
            slug="test-blog", meta_description="A test blog about testing.",
            keywords=["testing", "pytest"], estimated_reading_time_minutes=5,
            focus_keyword="pytest testing", og_title="Test Blog: pytest Guide",
        )
        mock_llm.with_structured_output.return_value.invoke.return_value = seo
        from app.graph.seo import seo_node
        result = seo_node(self._make_state())
        assert isinstance(result["seo"], dict)
        assert result["seo"]["slug"] == "test-blog"


# ── Quality reviewer ──────────────────────────────────────────────────────────

class TestQualityReviewerNode:
    def _make_state(self, final_md="# Blog\n\n" + " ".join(["word"] * 300)):
        task = Task(id=1, title="X", goal="G", bullets=["A","B","C"], target_words=250)
        plan = Plan(blog_title="Blog", audience="All", tone="Casual", tasks=[task])
        return {
            "topic": "X", "as_of": "2025-02-27", "mode": "closed_book",
            "needs_research": False, "queries": [], "evidence": [], "plan": plan,
            "recency_days": 0, "sections": [], "merged_md": "", "final": final_md,
            "md_with_placeholders": "", "image_specs": [], "seo": None,
            "social": None, "quality": None, "session_id": "t",
        }

    def test_reviewer_skipped_when_no_final(self):
        from app.graph.reviewer import quality_reviewer_node
        state = self._make_state(final_md="")
        result = quality_reviewer_node(state)
        assert result == {"quality": None}

    @patch("app.graph.reviewer.llm")
    def test_reviewer_returns_dict(self, mock_llm):
        q = QualityScore(
            overall=8.0, accuracy=9.0, clarity=8.0, depth=7.5,
            originality=7.0, seo_friendliness=8.5,
            suggestions=["Good work overall."], verdict="publish",
        )
        mock_llm.with_structured_output.return_value.invoke.return_value = q
        from app.graph.reviewer import quality_reviewer_node
        result = quality_reviewer_node(self._make_state())
        assert result["quality"]["verdict"] == "publish"
        assert result["quality"]["overall"] == 8.0
