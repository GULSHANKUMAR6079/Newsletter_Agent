"""
app/graph/seo.py
────────────────
SEO node: generates slug, meta description, keywords, reading time,
focus keyword, OG title, and canonical URL hint for the blog post.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import cfg
from app.models import SEOOutput, State
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

llm = ChatGoogleGenerativeAI(
    model=cfg.gemini_model,
    temperature=0.3,
    google_api_key=cfg.google_api_key,
)

SEO_SYSTEM = """\
You are an expert SEO strategist and technical content marketer.

Given a blog post (title + full markdown), generate production-ready SEO metadata.

Requirements:
- slug: lowercase, hyphenated, 3-7 words, describes the content precisely.
- meta_description: 120–160 chars, active voice, includes focus keyword, compelling CTA.
- keywords: 5–15 terms ordered by importance (primary first, then secondary/LSI).
- focus_keyword: the single most important keyword phrase (2-4 words).
- estimated_reading_time_minutes: based on ~238 wpm average adult reading speed.
- og_title: 50-60 chars, slightly more click-bait than the H1 (A/B test friendly).
- canonical_url_hint: /blog/<slug>

Output must match SEOOutput schema exactly.
"""


def seo_node(state: State) -> dict:
    """Generate SEO metadata for the final blog post."""
    plan = state.get("plan")
    final_md = state.get("final", "")

    if not final_md:
        logger.warning("SEO node | no final markdown found, skipping.")
        return {"seo": None}

    logger.info("SEO | generating metadata for '%s'", plan.blog_title if plan else "unknown")

    extractor = llm.with_structured_output(SEOOutput)
    seo: SEOOutput = extractor.invoke(
        [
            SystemMessage(content=SEO_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title if plan else 'Unknown'}\n"
                    f"Topic: {state.get('topic', '')}\n"
                    f"Audience: {plan.audience if plan else 'general'}\n\n"
                    f"Full blog markdown:\n{final_md[:6000]}"
                )
            ),
        ]
    )

    logger.info(
        "SEO | slug='%s' | focus_kw='%s' | read_time=%dmin | keywords=%d",
        seo.slug, seo.focus_keyword, seo.estimated_reading_time_minutes, len(seo.keywords)
    )
    return {"seo": seo.model_dump()}
