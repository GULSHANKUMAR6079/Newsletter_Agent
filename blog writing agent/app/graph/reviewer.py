"""
app/graph/reviewer.py
─────────────────────
Quality reviewer node: self-critique of the generated blog post using
a structured scoring rubric.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import cfg
from app.models import QualityScore, State
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

llm = ChatGoogleGenerativeAI(
    model=cfg.gemini_model,
    temperature=0.2,  # Low temperature for consistent, objective scoring
    google_api_key=cfg.google_api_key,
)

REVIEWER_SYSTEM = """\
You are a senior editorial director at a top tech publication (e.g. Smashing Magazine,
CSS-Tricks, Netflix Tech Blog).

Review the provided technical blog post against a strict quality rubric.

Scoring rubric (each 1–10):
- accuracy:       Are all claims correct? No hallucinations or outdated info?
- clarity:         Is the writing clear, well-structured, and easy to follow?
- depth:           Does it go beyond surface level? Does it add genuine insight?
- originality:     Is there a unique angle, example, or framing vs. generic posts?
- seo_friendliness: Are key terms used naturally? Is it structured for scanability?
- overall:         Weighted holistic score (accuracy 30%, depth 25%, clarity 20%,
                   originality 15%, seo 10%).

verdict:
- "publish"  → overall ≥ 7.5
- "revise"   → 5.0 ≤ overall < 7.5
- "reject"   → overall < 5.0

suggestions: 2-5 actionable, specific improvements (not vague like "add more detail").
  Example: "Section 3 claims X but does not cite a source — add a reference."

Output must match QualityScore schema exactly.
"""


def quality_reviewer_node(state: State) -> dict:
    """Self-critique the blog post and return a structured quality score."""
    plan = state.get("plan")
    final_md = state.get("final", "")

    if not final_md:
        logger.warning("Reviewer | no final markdown found, skipping.")
        return {"quality": None}

    logger.info(
        "Reviewer | evaluating '%s'",
        plan.blog_title if plan else "unknown"
    )

    scorer = llm.with_structured_output(QualityScore)
    quality: QualityScore = scorer.invoke(
        [
            SystemMessage(content=REVIEWER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title if plan else 'Unknown'}\n"
                    f"Blog kind: {plan.blog_kind if plan else 'explainer'}\n"
                    f"Audience: {plan.audience if plan else 'general'}\n\n"
                    f"Full blog post:\n{final_md[:8000]}"
                )
            ),
        ]
    )

    logger.info(
        "Reviewer | overall=%.1f | verdict=%s | accuracy=%.1f | clarity=%.1f | "
        "depth=%.1f | originality=%.1f | seo=%.1f",
        quality.overall, quality.verdict,
        quality.accuracy, quality.clarity,
        quality.depth, quality.originality, quality.seo_friendliness,
    )
    return {"quality": quality.model_dump()}
