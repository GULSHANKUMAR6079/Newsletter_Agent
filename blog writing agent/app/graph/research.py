"""
app/graph/research.py
─────────────────────
Research node: web search via Tavily + LLM evidence synthesis.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import cfg
from app.models import EvidenceItem, EvidencePack, State
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

llm = ChatGoogleGenerativeAI(
    model=cfg.gemini_model,
    temperature=0.2,  # low temp for factual extraction
    google_api_key=cfg.google_api_key,
)

# ── Tavily helper ──────────────────────────────────────────────────────────────

def _tavily_search(query: str, max_results: int = 6) -> List[dict]:
    """Run a single Tavily search query. Returns [] if Tavily is unavailable."""
    if not cfg.has_tavily:
        logger.debug("Tavily key not set — skipping search for '%s'", query)
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query}) or []
        out: List[dict] = []
        for r in results:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        logger.debug("Tavily '%s' → %d results", query, len(out))
        return out
    except Exception as exc:
        logger.warning("Tavily search failed for '%s': %s", query, exc)
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


# ── System prompt ──────────────────────────────────────────────────────────────
RESEARCH_SYSTEM = """\
You are a research synthesizer for a technical blog writing system.

Given raw web search results, produce clean EvidenceItem objects.

Rules:
- Only include items with a non-empty URL.
- Prefer relevant + authoritative sources (official docs, papers, reputable tech media).
- Normalise published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short, informative, and factual.
- Deduplicate by URL.
- Do NOT invent facts not present in the raw results.
"""


# ── Node ───────────────────────────────────────────────────────────────────────
def research_node(state: State) -> dict:
    """Collect web evidence for the given queries and synthesize into EvidenceItems."""
    queries = (state.get("queries") or [])[:10]
    logger.info("Research | %d queries to run", len(queries))

    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=cfg.max_research_results))

    if not raw:
        logger.warning("Research | No raw results found — proceeding without evidence")
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack: EvidencePack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state.get('recency_days', 45)}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    # Deduplicate by URL
    dedup: dict = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    # For open-book mode, filter to recent results only
    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state.get("recency_days", 7)))
        before = len(evidence)
        evidence = [
            e for e in evidence
            if (d := _iso_to_date(e.published_at)) and d >= cutoff
        ]
        logger.info(
            "Research | open_book date filter: %d → %d items (cutoff %s)",
            before, len(evidence), cutoff
        )

    logger.info("Research | Final evidence items: %d", len(evidence))
    return {"evidence": evidence}
