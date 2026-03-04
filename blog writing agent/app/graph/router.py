"""
app/graph/router.py
───────────────────
Router node: decides whether web research is needed before planning.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import cfg
from app.models import RouterDecision, State
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=cfg.gemini_model,
    temperature=cfg.llm_temperature,
    google_api_key=cfg.google_api_key,
)

# ── System prompt ──────────────────────────────────────────────────────────────
ROUTER_SYSTEM = """\
You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book  (needs_research=false): evergreen concepts, timeless technical topics.
- hybrid       (needs_research=true):  evergreen + needs up-to-date examples/tools/models.
- open_book    (needs_research=true):  volatile weekly/news/"latest"/pricing/policy topics.

If needs_research=true:
  - Output 3–10 high-signal, scoped search queries.
  - For open_book weekly roundup, include queries reflecting last 7 days.

Be decisive. When in doubt, prefer hybrid over closed_book.
"""


# ── Node ───────────────────────────────────────────────────────────────────────
def router_node(state: State) -> dict:
    """Classify the topic and choose a research mode."""
    logger.info("Router | topic='%s' | as_of=%s", state["topic"], state["as_of"])

    decider = llm.with_structured_output(RouterDecision)
    decision: RouterDecision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"As-of date: {state['as_of']}"
                )
            ),
        ]
    )

    recency_map = {"open_book": 7, "hybrid": 45, "closed_book": 3650}
    recency_days = recency_map.get(decision.mode, 45)

    logger.info(
        "Router decision | mode=%s | needs_research=%s | queries=%d | recency_days=%d",
        decision.mode,
        decision.needs_research,
        len(decision.queries),
        recency_days,
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    """Conditional edge: route to 'research' or skip to 'orchestrator'."""
    return "research" if state.get("needs_research") else "orchestrator"
