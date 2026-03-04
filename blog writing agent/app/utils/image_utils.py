"""
app/utils/image_utils.py
────────────────────────
Gemini image generation and local image resolution helpers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_image_bytes(prompt: str, model: str = "gemini-2.0-flash-preview-image-generation") -> bytes:
    """
    Call Google Gemini to generate an image.

    Args:
        prompt:  Text prompt describing the desired image.
        model:   Gemini image model identifier.

    Returns:
        Raw PNG/JPEG bytes.

    Raises:
        RuntimeError: If no image bytes are returned (quota, safety, SDK).
    """
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    logger.debug("Generating image | model=%s | prompt=%.80s…", model, prompt)

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Normalise across SDK versions
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            logger.debug("Image bytes received (%d bytes)", len(inline.data))
            return inline.data  # type: ignore[return-value]

    raise RuntimeError("No inline image bytes found in response parts.")


def resolve_image_path(src: str) -> Path:
    """Resolve a potentially relative image src to an absolute Path."""
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def save_image(img_bytes: bytes, out_path: Path) -> None:
    """Write raw image bytes to disk, creating parent dirs as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(img_bytes)
    logger.info("Image saved → %s (%d bytes)", out_path, len(img_bytes))


def generate_and_save(
    prompt: str,
    out_path: Path,
    model: str = "gemini-2.0-flash-preview-image-generation",
    skip_if_exists: bool = True,
) -> Optional[Path]:
    """
    High-level helper: generate an image and save it.

    Returns the path on success, None on failure (logs the error).
    """
    if skip_if_exists and out_path.exists():
        logger.debug("Image already exists, skipping: %s", out_path)
        return out_path
    try:
        img_bytes = generate_image_bytes(prompt, model=model)
        save_image(img_bytes, out_path)
        return out_path
    except Exception as exc:
        logger.warning("Image generation failed for '%s': %s", out_path.name, exc)
        return None
