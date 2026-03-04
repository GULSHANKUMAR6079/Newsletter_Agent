"""
app/graph/orchestrator.py
─────────────────────────
Orchestrator node: produces a structured blog outline (Plan) from the topic + evidence.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import cfg
from app.models import Plan, State
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

llm = ChatGoogleGenerativeAI(
    model=cfg.gemini_model,
    temperature=cfg.llm_temperature,
    google_api_key=cfg.google_api_key,
)

ORCH_SYSTEM = """\
You are a senior technical writer and developer advocate.

Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words (120–550).
- Tags per task: flexible; do not force a fixed taxonomy.
- blog_title must be compelling and SEO-friendly.

Grounding by mode:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks
  requires_research=True and requires_citations=True.
- open_book (news_roundup):
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested.
  - If evidence is weak, reflect that in the plan (don't invent events).

Output must match the Plan schema exactly.
"""


def orchestrator_node(state: State) -> dict:
    """Generate a structured Plan from the topic and any gathered evidence."""
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    logger.info("Orchestrator | mode=%s | evidence_items=%d", mode, len(evidence))

    planner = llm.with_structured_output(Plan)
    forced_kind = "news_roundup" if mode == "open_book" else None

    plan: Plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state.get('recency_days', 45)})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    if forced_kind:
        plan.blog_kind = "news_roundup"

    logger.info(
        "Orchestrator | plan='%s' | tasks=%d | kind=%s",
        plan.blog_title, len(plan.tasks), plan.blog_kind
    )
    return {"plan": plan}
