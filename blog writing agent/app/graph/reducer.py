"""
app/graph/reducer.py
────────────────────
Reducer subgraph nodes:
  1. merge_content    – stitches parallel worker sections into one markdown doc
  2. decide_images    – plans image placements + prompts
  3. generate_and_place_images – generates images via Gemini and substitutes placeholders
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import cfg
from app.models import GlobalImagePlan, State
from app.utils.file_utils import safe_slug, write_output
from app.utils.image_utils import generate_and_save
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

llm = ChatGoogleGenerativeAI(
    model=cfg.gemini_model,
    temperature=0.3,
    google_api_key=cfg.google_api_key,
)

DECIDE_IMAGES_SYSTEM = f"""\
You are an expert technical editor deciding whether diagrams/images are needed.

Rules:
- Max {cfg.max_images_per_blog} images total.
- Each image must materially improve understanding (flow diagram, architecture, table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images are needed: md_with_placeholders must equal input markdown and images=[].
- Avoid decorative images; prefer technical diagrams with clear labels.
- For news_roundup blogs: prefer no images (news digests read better without diagrams).

Return strictly GlobalImagePlan schema.
"""


# ── Node 1: Merge ─────────────────────────────────────────────────────────────

def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without a plan.")

    ordered_sections = [
        md for _, md in sorted(state["sections"], key=lambda x: x[0])
    ]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    logger.info(
        "Reducer.merge | title='%s' | sections=%d | ~%d words",
        plan.blog_title, len(ordered_sections), len(merged_md.split())
    )
    return {"merged_md": merged_md}


# ── Node 2: Decide images ─────────────────────────────────────────────────────

def decide_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    planner = llm.with_structured_output(GlobalImagePlan)
    image_plan: GlobalImagePlan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{state['merged_md']}"
                )
            ),
        ]
    )

    logger.info(
        "Reducer.decide_images | images_planned=%d", len(image_plan.images)
    )
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


# ── Node 3: Generate & place images ───────────────────────────────────────────

def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs") or []

    # Determine output directory
    output_dir = Path(cfg.output_dir)
    images_dir = output_dir / cfg.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    if not image_specs:
        logger.info("Reducer.images | No images requested, writing markdown only.")
        filename = f"{safe_slug(plan.blog_title)}.md"
        write_output(md, filename, output_dir)
        return {"final": md}

    for spec in image_specs:
        placeholder = spec["placeholder"]
        out_path = images_dir / spec["filename"]

        saved = generate_and_save(
            prompt=spec["prompt"],
            out_path=out_path,
            model=cfg.gemini_image_model,
        )

        if saved:
            img_md = f"![{spec['alt']}](images/{spec['filename']})\n*{spec['caption']}*"
            md = md.replace(placeholder, img_md)
        else:
            # Graceful fallback — keep placeholder as a blockquote note
            fallback = (
                f"> **[IMAGE: {spec.get('caption', '')}]**\n>\n"
                f"> *Alt:* {spec.get('alt', '')}\n>\n"
                f"> *Prompt:* {spec.get('prompt', '')}\n"
            )
            md = md.replace(placeholder, fallback)

    filename = f"{safe_slug(plan.blog_title)}.md"
    write_output(md, filename, output_dir)
    logger.info("Reducer | Final markdown written → output/%s", filename)
    return {"final": md}
