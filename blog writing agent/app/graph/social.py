"""
app/graph/social.py
───────────────────
Social content node: generates a LinkedIn post and Twitter/X thread
from the completed blog post.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import cfg
from app.models import SocialContent, State
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

llm = ChatGoogleGenerativeAI(
    model=cfg.gemini_model,
    temperature=0.8,  # Higher creativity for social copy
    google_api_key=cfg.google_api_key,
)

SOCIAL_SYSTEM = """\
You are a world-class B2B tech content marketer and ghostwriter.

Given a technical blog post, produce high-engagement social media content.

LinkedIn post requirements:
- Max 3000 characters.
- Hook in first 2 lines (visible before "see more").
- Use short paragraphs (1-2 sentences), line breaks for scanability.
- Include 3-5 relevant hashtags at the end.
- End with a clear CTA (comment, share, or read the full post).
- Tone: insightful, direct, professional-but-human.
- Do NOT use generic openers like "Excited to share" or "I'm thrilled".

Twitter/X thread requirements:
- 6-10 tweets, each ≤280 characters.
- Tweet 1: hook (bold claim or surprising stat).
- Tweet 2-N: one key insight per tweet.
- Final tweet: CTA + link placeholder [BLOG_URL].
- Number each tweet: "1/N", "2/N", etc.
- No hashtags inside tweets (put them only in the last tweet if needed).

Output must match SocialContent schema exactly.
"""


def social_node(state: State) -> dict:
    """Generate LinkedIn post and Twitter thread from the final blog post."""
    plan = state.get("plan")
    final_md = state.get("final", "")

    if not final_md:
        logger.warning("Social node | no final markdown found, skipping.")
        return {"social": None}

    logger.info(
        "Social | generating content for '%s'",
        plan.blog_title if plan else "unknown"
    )

    extractor = llm.with_structured_output(SocialContent)
    social: SocialContent = extractor.invoke(
        [
            SystemMessage(content=SOCIAL_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title if plan else 'Unknown'}\n"
                    f"Topic: {state.get('topic', '')}\n"
                    f"Audience: {plan.audience if plan else 'technical professionals'}\n"
                    f"Tone: {plan.tone if plan else 'professional'}\n\n"
                    f"Blog post (truncated to 5000 chars):\n{final_md[:5000]}"
                )
            ),
        ]
    )

    logger.info(
        "Social | linkedin=%d chars | twitter_thread=%d tweets",
        len(social.linkedin_post), len(social.twitter_thread)
    )
    return {"social": social.model_dump()}
