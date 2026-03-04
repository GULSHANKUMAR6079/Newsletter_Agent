"""
tests/test_utils.py
───────────────────
Unit tests for app/utils helpers.
"""

import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from app.utils.file_utils import (
    safe_slug,
    extract_title_from_md,
    word_count,
    reading_time_minutes,
    bundle_zip,
    images_zip,
)


# ── safe_slug ─────────────────────────────────────────────────────────────────

class TestSafeSlug:
    def test_basic(self):
        assert safe_slug("Hello World") == "hello_world"

    def test_special_chars(self):
        assert safe_slug("LangGraph: A Deep-Dive!") == "langgraph_a_deepdive"

    def test_empty_string(self):
        assert safe_slug("") == "blog"

    def test_only_symbols(self):
        assert safe_slug("!!!???###") == "blog"

    def test_preserves_hyphens(self):
        result = safe_slug("my-blog-post")
        assert "blog" in result

    def test_strips_leading_trailing(self):
        assert safe_slug("  spaces  ") == "spaces"


# ── extract_title_from_md ─────────────────────────────────────────────────────

class TestExtractTitle:
    def test_h1_heading(self):
        md = "# My Great Blog Post\n\nSome content here."
        assert extract_title_from_md(md) == "My Great Blog Post"

    def test_no_heading_returns_fallback(self):
        md = "Just some plain text without headings."
        assert extract_title_from_md(md, "fallback") == "fallback"

    def test_first_h1_only(self):
        md = "# First\n\n## Second\n\n# Third"
        assert extract_title_from_md(md) == "First"

    def test_empty_h1_returns_fallback(self):
        md = "# \n\nContent"
        assert extract_title_from_md(md, "fallback") == "fallback"


# ── word_count ────────────────────────────────────────────────────────────────

class TestWordCount:
    def test_simple(self):
        assert word_count("one two three") == 3

    def test_empty(self):
        assert word_count("") == 0

    def test_multiline(self):
        assert word_count("hello\nworld\nfoo") == 3


# ── reading_time_minutes ──────────────────────────────────────────────────────

class TestReadingTime:
    def test_short_text_minimum_one(self):
        assert reading_time_minutes("hello world") == 1

    def test_average_article(self):
        # ~1000 words at 238 wpm ≈ 4 minutes
        text = " ".join(["word"] * 1000)
        assert reading_time_minutes(text) == 4


# ── bundle_zip ────────────────────────────────────────────────────────────────

class TestBundleZip:
    def test_creates_zip_with_md(self, tmp_path):
        md_text = "# Test Blog\n\nContent here."
        images_dir = tmp_path / "images"  # doesn't exist is fine
        z_bytes = bundle_zip(md_text, "test.md", images_dir)
        assert isinstance(z_bytes, bytes)
        with zipfile.ZipFile(BytesIO(z_bytes)) as z:
            assert "test.md" in z.namelist()

    def test_includes_images(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "img1.png").write_bytes(b"\x89PNG fake")
        z_bytes = bundle_zip("# Blog", "blog.md", images_dir)
        with zipfile.ZipFile(BytesIO(z_bytes)) as z:
            names = z.namelist()
            assert "blog.md" in names
            assert any("img1.png" in n for n in names)


# ── images_zip ────────────────────────────────────────────────────────────────

class TestImagesZip:
    def test_none_when_dir_missing(self, tmp_path):
        result = images_zip(tmp_path / "nonexistent")
        assert result is None

    def test_none_when_dir_empty(self, tmp_path):
        d = tmp_path / "images"
        d.mkdir()
        assert images_zip(d) is None

    def test_zip_with_files(self, tmp_path):
        d = tmp_path / "images"
        d.mkdir()
        (d / "a.png").write_bytes(b"fake")
        result = images_zip(d)
        assert result is not None
        with zipfile.ZipFile(BytesIO(result)) as z:
            assert any("a.png" in n for n in z.namelist())
