"""
app/graph/worker.py
───────────────────
Worker node: writes a single blog section in parallel.
Includes retry logic via tenacity for resilience against transient LLM errors.
"""

from __future__ import annotations

import time
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.config import cfg
from app.models import EvidenceItem, Plan, Task
from app.utils.logging_utils import get_logger
import logging

logger = get_logger(__name__)

llm = ChatGoogleGenerativeAI(
    model=cfg.gemini_model,
    temperature=cfg.llm_temperature,
    google_api_key=cfg.google_api_key,
)

WORKER_SYSTEM = """\
You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Quality standards:
- Write like an expert explaining to a smart peer, NOT a beginner tutorial.
- Use concrete examples, analogies, and precise language.
- Avoid fluff, filler phrases, and excessive bullet lists in prose.
- Cover ALL bullets in order; transition naturally between them.
- Target word count: ±15% of the specified target_words.
- Output ONLY section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials.
  Focus on events, implications, and analysis.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy
  claim unless supported by provided Evidence URLs. Attach a Markdown link ([Source](URL))
  for each supported claim. If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal, runnable code snippet.
"""


@retry(
    stop=stop_after_attempt(cfg.max_retries),
    wait=wait_exponential(
        multiplier=1,
        min=cfg.retry_wait_min,
        max=cfg.retry_wait_max,
    ),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _invoke_llm(messages: list) -> str:
    """Invoke the LLM with retry logic."""
    return llm.invoke(messages).content.strip()


def worker_node(payload: dict) -> dict:
    """
    Write a single blog section.
    Called in parallel via LangGraph's Send API.
    """
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence: List[EvidenceItem] = [
        EvidenceItem(**e) for e in payload.get("evidence", [])
    ]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[: cfg.max_evidence_items]
    )

    logger.info(
        "Worker | task_id=%d | '%s' | target_words=%d",
        task.id, task.title, task.target_words
    )
    t0 = time.monotonic()

    try:
        section_md = _invoke_llm(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog title: {plan.blog_title}\n"
                        f"Audience: {plan.audience}\n"
                        f"Tone: {plan.tone}\n"
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Constraints: {plan.constraints}\n"
                        f"Topic: {payload['topic']}\n"
                        f"Mode: {payload.get('mode')}\n"
                        f"As-of: {payload.get('as_of')} "
                        f"(recency_days={payload.get('recency_days')})\n\n"
                        f"Section title: {task.title}\n"
                        f"Goal: {task.goal}\n"
                        f"Target words: {task.target_words}\n"
                        f"Tags: {task.tags}\n"
                        f"requires_research: {task.requires_research}\n"
                        f"requires_citations: {task.requires_citations}\n"
                        f"requires_code: {task.requires_code}\n"
                        f"Bullets:{bullets_text}\n\n"
                        f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                    )
                ),
            ]
        )
    except Exception as exc:
        logger.error("Worker task_id=%d failed after retries: %s", task.id, exc)
        section_md = (
            f"## {task.title}\n\n"
            f"> ⚠️ **Section generation failed**: {exc}\n\n"
            f"*Please regenerate this section.*"
        )

    elapsed = time.monotonic() - t0
    logger.info(
        "Worker | task_id=%d done | %.1fs | ~%d words",
        task.id, elapsed, len(section_md.split())
    )
    return {"sections": [(task.id, section_md)]}
