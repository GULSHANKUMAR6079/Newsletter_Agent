"""
app/utils/file_utils.py
───────────────────────
File and text helpers: slugification, zip bundling, markdown title extraction.
"""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional


# ── Slugification ──────────────────────────────────────────────────────────────

def safe_slug(title: str) -> str:
    """Convert an arbitrary string to a safe filesystem / URL slug."""
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


# ── Markdown helpers ───────────────────────────────────────────────────────────

def extract_title_from_md(md: str, fallback: str = "Untitled") -> str:
    """Return the first ATX H1 heading from markdown, or fallback."""
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


def word_count(text: str) -> int:
    """Return approximate word count of a string."""
    return len(text.split())


def reading_time_minutes(text: str, wpm: int = 238) -> int:
    """Estimate reading time in minutes (average adult: ~238 wpm)."""
    return max(1, round(word_count(text) / wpm))


# ── Zip bundling ───────────────────────────────────────────────────────────────

def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    """
    Pack the markdown file + all images into a single ZIP.
    Returns raw ZIP bytes for :func:`st.download_button`.
    """
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    """Pack all files in images_dir into a ZIP. Returns None if dir is empty."""
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    files = [p for p in images_dir.rglob("*") if p.is_file()]
    if not files:
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=str(p))
    return buf.getvalue()


def export_html(md_text: str, title: str = "Blog Post") -> str:
    """
    Convert markdown text to a self-contained HTML string with embedded styles.
    Uses a minimal inline renderer (no heavy deps required at import time).
    """
    try:
        import markdown as md_lib  # type: ignore
        body = md_lib.markdown(md_text, extensions=["extra", "codehilite", "toc"])
    except ImportError:
        # Fallback: wrap raw markdown in a <pre> tag
        body = f"<pre>{md_text}</pre>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title}</title>
  <style>
    body {{ font-family: 'Georgia', serif; max-width: 820px; margin: 40px auto;
           padding: 0 20px; color: #1a1a1a; line-height: 1.75; }}
    h1,h2,h3 {{ font-family: 'Helvetica Neue', sans-serif; }}
    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }}
    pre  {{ background: #1e1e2e; color: #cdd6f4; padding: 16px;
            border-radius: 8px; overflow-x: auto; }}
    blockquote {{ border-left: 4px solid #7c3aed; margin: 0; padding-left: 16px;
                  color: #555; }}
    img {{ max-width: 100%; border-radius: 8px; }}
    a {{ color: #7c3aed; }}
  </style>
</head>
<body>{body}</body>
</html>"""


# ── Output file helpers ────────────────────────────────────────────────────────

def write_output(content: str, filename: str, output_dir: Path) -> Path:
    """Write content to output_dir/filename, creating dirs as needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / filename
    out.write_text(content, encoding="utf-8")
    return out
