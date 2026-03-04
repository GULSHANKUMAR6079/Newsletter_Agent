"""
frontend/app.py
───────────────
Production-grade Streamlit frontend for the Blog Writing Agent.

Tabs:
  🧩 Plan | 🔎 Evidence | 📝 Preview | 🎨 SEO | 📱 Social | ⭐ Quality
  🖼️ Images | 📚 History | 🧾 Logs
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st

# ── Path setup so `app` package is importable ───────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import cfg
from app.database import db
from app.graph.graph import app as graph_app
from app.models import BlogSessionRecord
from app.utils.file_utils import (
    bundle_zip,
    export_html,
    extract_title_from_md,
    images_zip,
    reading_time_minutes,
    safe_slug,
    word_count,
)
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ───────────────────────────────────────────────────────────────
_CSS_PATH = Path(__file__).parent / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ── DB init (run once per session) ───────────────────────────────────────────
@st.cache_resource
def _init_db():
    try:
        asyncio.run(db.init())
    except Exception as e:
        logger.warning("DB init skipped: %s", e)

_init_db()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run an async coroutine from sync Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except Exception:
        return asyncio.run(coro)


def try_stream(inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Stream graph updates, fall back to full invoke."""
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass
    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_state(current: Dict, payload: Any) -> Dict:
    if isinstance(payload, dict):
        if len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
            current.update(next(iter(payload.values())))
        else:
            current.update(payload)
    return current


def get_plan_dict(out: dict) -> Optional[dict]:
    plan = out.get("plan")
    if plan is None:
        return None
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    if isinstance(plan, dict):
        return plan
    return json.loads(json.dumps(plan, default=str))


def copy_btn(label: str, text: str, key: str):
    """Render a copy-to-clipboard button with JS."""
    safe = text.replace("`", "\\`").replace("\\", "\\\\")
    st.markdown(
        f"""
        <button onclick="navigator.clipboard.writeText(`{safe}`)"
                class="copy-btn">📋 {label}</button>
        """,
        unsafe_allow_html=True,
    )


# ── Image renderer ────────────────────────────────────────────────────────────
import re

_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAP_RE    = re.compile(r"^\*(?P<cap>.+)\*$")


def render_markdown_with_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return
    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        if md[last : m.start()]:
            parts.append(("md", md[last : m.start()]))
        parts.append(("img", f"{m.group('alt')}|||{m.group('src')}"))
        last = m.end()
    if md[last:]:
        parts.append(("md", md[last:]))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
        else:
            alt, src = payload.split("|||", 1)
            caption = None
            if i + 1 < len(parts) and parts[i + 1][0] == "md":
                nxt = parts[i + 1][1].lstrip()
                first_line = nxt.splitlines()[0].strip() if nxt.strip() else ""
                mcap = _CAP_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    parts[i + 1] = ("md", "\n".join(nxt.splitlines()[1:]))
            if src.startswith(("http://", "https://")):
                st.image(src, caption=caption or alt or None, use_container_width=True)
            else:
                p = Path(src.strip().lstrip("./")).resolve()
                if p.exists():
                    st.image(str(p), caption=caption or alt or None, use_container_width=True)
                else:
                    st.warning(f"Image not found: `{src}`")
        i += 1


# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("last_out", None),
    ("logs", []),
    ("current_session_id", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="brand-icon">✍️</span>
            <div>
                <div class="brand-title">Blog Writing Agent</div>
                <div class="brand-sub">Powered by Gemini + LangGraph</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 📝 New Blog")
    topic = st.text_area(
        "Topic",
        placeholder="e.g. How LangGraph enables stateful AI agents",
        height=100,
        label_visibility="collapsed",
    )
    as_of = st.date_input("As-of date", value=date.today())

    with st.expander("⚙️ Advanced Settings", expanded=False):
        tone_preset = st.selectbox(
            "Tone",
            ["Technical & Precise", "Conversational", "Academic", "Beginner-Friendly", "Executive"],
        )
        target_words = st.slider("Target words per section", 120, 550, 300, 10)
        session_name = st.text_input(
            "Session label (optional)",
            placeholder="e.g. Q1 LangGraph post",
        )

    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)
    st.divider()

    # Past blogs from DB
    st.markdown("### 📚 History")
    hist_sessions = _run_async(db.list_sessions(limit=30))
    if not hist_sessions:
        st.caption("No saved sessions found.")
    else:
        hist_options = {
            f"{s.blog_title}  ·  {s.created_at.strftime('%b %d')}": s.id
            for s in hist_sessions
        }
        selected_label = st.radio(
            "Select",
            list(hist_options.keys()),
            index=0,
            label_visibility="collapsed",
        )
        col1, col2 = st.columns(2)
        if col1.button("📂 Load", use_container_width=True):
            sid = hist_options[selected_label]
            record = _run_async(db.load_session(sid))
            if record:
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": record.final_md,
                    "seo": record.seo,
                    "social": record.social,
                    "quality": record.quality,
                }
                st.session_state["current_session_id"] = record.id
                st.rerun()
        if col2.button("🗑️ Delete", use_container_width=True):
            sid = hist_options[selected_label]
            _run_async(db.delete_session(sid))
            st.success("Deleted.")
            st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────
(
    tab_plan, tab_evidence, tab_preview,
    tab_seo, tab_social, tab_quality,
    tab_images, tab_logs,
) = st.tabs([
    "🧩 Plan", "🔎 Evidence", "📝 Preview",
    "🎨 SEO", "📱 Social", "⭐ Quality",
    "🖼️ Images", "🧾 Logs",
])

run_logs: List[str] = []


def log(msg: str):
    run_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Run graph ─────────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "seo": None,
        "social": None,
        "quality": None,
        "final": "",
        "session_id": str(uuid4()),
    }

    status = st.status("🔄 Running Blog Writing Agent…", expanded=True)
    progress = st.empty()
    current_state: Dict[str, Any] = {}
    last_node = None

    NODE_LABELS = {
        "router": "🔀 Routing topic…",
        "research": "🔍 Researching the web…",
        "orchestrator": "🧩 Planning outline…",
        "worker": "✍️ Writing sections (parallel)…",
        "reducer": "🔗 Merging + generating images…",
        "seo": "🎨 Generating SEO metadata…",
        "social": "📱 Crafting social content…",
        "reviewer": "⭐ Running quality review…",
    }

    for kind, payload in try_stream(inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1:
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                label = NODE_LABELS.get(node_name, f"➡️ `{node_name}`")
                status.write(label)
                last_node = node_name
            current_state = extract_state(current_state, payload)
            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "evidence_count": len(current_state.get("evidence") or []),
                "tasks_planned": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                "sections_done": len(current_state.get("sections") or []),
                "images": len(current_state.get("image_specs") or []),
            }
            progress.json(summary)
            log(f"[{kind}] node={node_name} | {json.dumps(payload, default=str)[:800]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            st.session_state["logs"].extend(run_logs)

            # Save to PostgreSQL
            plan_d = get_plan_dict(out) or {}
            record = BlogSessionRecord(
                topic=topic.strip(),
                blog_title=plan_d.get("blog_title", extract_title_from_md(out.get("final", ""), topic)),
                mode=out.get("mode", "closed_book"),
                final_md=out.get("final", ""),
                seo=out.get("seo"),
                social=out.get("social"),
                quality=out.get("quality"),
            )
            st.session_state["current_session_id"] = record.id
            saved = _run_async(db.save_session(record))
            if saved:
                log(f"Session saved to PostgreSQL | id={record.id}")

            status.update(label="✅ Blog generated!", state="complete", expanded=False)
            progress.empty()


# ── Render result ─────────────────────────────────────────────────────────────
out = st.session_state.get("last_out")

if out:
    # ── Plan tab ──────────────────────────────────────────────────────────────
    with tab_plan:
        plan_dict = get_plan_dict(out)
        if not plan_dict:
            st.info("No plan found. Generate a blog to see its outline.")
        else:
            st.markdown(f"## {plan_dict.get('blog_title', '')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Audience", plan_dict.get("audience", "—"))
            c2.metric("Tone", plan_dict.get("tone", "—"))
            c3.metric("Kind", plan_dict.get("blog_kind", "—"))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame([
                    {
                        "#": t.get("id"),
                        "Section": t.get("title"),
                        "Words": t.get("target_words"),
                        "Research": "✅" if t.get("requires_research") else "—",
                        "Citations": "✅" if t.get("requires_citations") else "—",
                        "Code": "✅" if t.get("requires_code") else "—",
                        "Tags": ", ".join(t.get("tags") or []),
                    }
                    for t in tasks
                ]).sort_values("#")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("📄 Full task details (JSON)"):
                    st.json(tasks)

    # ── Evidence tab ──────────────────────────────────────────────────────────
    with tab_evidence:
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence collected. Topic was closed-book or Tavily key not set.")
        else:
            st.metric("Sources found", len(evidence))
            rows = []
            for e in evidence:
                ed = e.model_dump() if hasattr(e, "model_dump") else e
                rows.append({
                    "Title": ed.get("title", ""),
                    "Published": ed.get("published_at", ""),
                    "Source": ed.get("source", ""),
                    "URL": ed.get("url", ""),
                })
            df_ev = pd.DataFrame(rows)
            # Make URLs clickable
            st.dataframe(
                df_ev,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "URL": st.column_config.LinkColumn("URL"),
                },
            )

    # ── Preview tab ───────────────────────────────────────────────────────────
    with tab_preview:
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No content yet — generate a blog first.")
        else:
            wc = word_count(final_md)
            rt = reading_time_minutes(final_md)
            m1, m2 = st.columns(2)
            m1.metric("Word count", f"{wc:,}")
            m2.metric("Reading time", f"{rt} min")

            st.divider()
            render_markdown_with_images(final_md)
            st.divider()

            plan_d = get_plan_dict(out) or {}
            blog_title = plan_d.get("blog_title") or extract_title_from_md(final_md, "blog")
            slug = safe_slug(blog_title)
            md_filename = f"{slug}.md"
            output_dir = Path(cfg.output_dir)

            dl1, dl2, dl3 = st.columns(3)
            dl1.download_button(
                "⬇️ Markdown",
                data=final_md.encode(),
                file_name=md_filename,
                mime="text/markdown",
            )
            dl2.download_button(
                "📦 ZIP Bundle",
                data=bundle_zip(final_md, md_filename, output_dir / "images"),
                file_name=f"{slug}_bundle.zip",
                mime="application/zip",
            )
            html_content = export_html(final_md, title=blog_title)
            dl3.download_button(
                "🌐 HTML",
                data=html_content.encode(),
                file_name=f"{slug}.html",
                mime="text/html",
            )

    # ── SEO tab ───────────────────────────────────────────────────────────────
    with tab_seo:
        seo = out.get("seo")
        if not seo:
            st.info("SEO metadata not yet generated.")
        else:
            st.markdown("## 🎨 SEO Metadata")
            s1, s2 = st.columns(2)
            s1.markdown(f"**Slug**")
            s1.code(seo.get("slug", ""), language="text")
            copy_btn("Copy slug", seo.get("slug", ""), "copy_slug")

            s2.markdown(f"**Focus keyword**")
            s2.info(seo.get("focus_keyword", ""))

            st.markdown("**Meta description**")
            meta = seo.get("meta_description", "")
            st.text_area("", value=meta, height=80, disabled=True, label_visibility="collapsed")
            copy_btn("Copy meta description", meta, "copy_meta")
            st.caption(f"{len(meta)}/160 characters")

            st.markdown("**OG Title**")
            og = seo.get("og_title", "")
            st.text_input("", value=og, disabled=True, label_visibility="collapsed")
            copy_btn("Copy OG title", og, "copy_og")

            st.markdown("**Keywords**")
            kw = seo.get("keywords", [])
            st.markdown(" ".join([f"`{k}`" for k in kw]))
            copy_btn("Copy all keywords", ", ".join(kw), "copy_kw")

            col1, col2 = st.columns(2)
            col1.metric("Reading time", f"{seo.get('estimated_reading_time_minutes', 0)} min")
            col2.markdown(f"**Canonical hint**: `{seo.get('canonical_url_hint', '')}`")

    # ── Social tab ────────────────────────────────────────────────────────────
    with tab_social:
        social = out.get("social")
        if not social:
            st.info("Social content not yet generated.")
        else:
            st.markdown("## 📱 Social Media Content")
            soc_tab1, soc_tab2 = st.tabs(["💼 LinkedIn", "🐦 Twitter / X"])

            with soc_tab1:
                linkedin = social.get("linkedin_post", "")
                st.text_area(
                    "LinkedIn post",
                    value=linkedin,
                    height=300,
                    disabled=True,
                    label_visibility="visible",
                )
                copy_btn("Copy LinkedIn post", linkedin, "copy_li")
                st.caption(f"{len(linkedin):,} / 3,000 characters")

                hashtags = social.get("hashtags", [])
                if hashtags:
                    st.markdown("**Suggested hashtags:** " + " ".join([f"`#{h}`" for h in hashtags]))

            with soc_tab2:
                tweets = social.get("twitter_thread", [])
                if tweets:
                    full_thread = "\n\n".join(
                        f"{t.get('position', i+1)}/ {t.get('text', '')}"
                        for i, t in enumerate(tweets)
                    )
                    copy_btn("Copy full thread", full_thread, "copy_thread")
                    for i, tweet in enumerate(tweets):
                        text_t = tweet.get("text", "")
                        with st.container():
                            st.markdown(
                                f"""
                                <div class="tweet-card">
                                    <span class="tweet-num">{tweet.get('position', i+1)}/</span>
                                    {text_t}
                                    <div class="tweet-chars">{len(text_t)}/280</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # ── Quality tab ───────────────────────────────────────────────────────────
    with tab_quality:
        quality = out.get("quality")
        if not quality:
            st.info("Quality review not yet generated.")
        else:
            st.markdown("## ⭐ Quality Review")
            verdict = quality.get("verdict", "publish")
            verdict_emojis = {"publish": "🟢 Ready to Publish", "revise": "🟡 Needs Revision", "reject": "🔴 Major Rework Needed"}
            verdict_colors = {"publish": "success", "revise": "warning", "reject": "error"}

            col_v, col_s = st.columns([1, 2])
            with col_v:
                st.markdown(
                    f"<div class='verdict-badge verdict-{verdict}'>{verdict_emojis.get(verdict, verdict)}</div>",
                    unsafe_allow_html=True,
                )
                st.metric("Overall Score", f"{quality.get('overall', 0):.1f}/10")

            with col_s:
                dims = ["accuracy", "clarity", "depth", "originality", "seo_friendliness"]
                dim_labels = ["Accuracy", "Clarity", "Depth", "Originality", "SEO"]
                scores = [quality.get(d, 0) for d in dims]

                try:
                    import plotly.graph_objects as go
                    fig = go.Figure(data=go.Scatterpolar(
                        r=scores + [scores[0]],
                        theta=dim_labels + [dim_labels[0]],
                        fill="toself",
                        fillcolor="rgba(124,58,237,0.25)",
                        line_color="#7c3aed",
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 10], tickfont_size=10),
                        ),
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                        margin=dict(l=40, r=40, t=20, b=20),
                        height=280,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    for label, score in zip(dim_labels, scores):
                        st.metric(label, f"{score:.1f}/10")

            st.markdown("### 🛠️ Suggestions")
            for suggestion in quality.get("suggestions", []):
                st.markdown(f"- {suggestion}")

    # ── Images tab ────────────────────────────────────────────────────────────
    with tab_images:
        specs = out.get("image_specs") or []
        images_dir = Path(cfg.output_dir) / "images"

        if not specs and not images_dir.exists():
            st.info("No images generated for this post.")
        else:
            if specs:
                with st.expander("Image plan"):
                    st.json(specs)
            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("`images/` folder exists but is empty.")
                else:
                    cols = st.columns(min(len(files), 3))
                    for i, p in enumerate(sorted(files)):
                        cols[i % 3].image(str(p), caption=p.name, use_container_width=True)
                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download all images (ZIP)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )

    # ── Logs tab ──────────────────────────────────────────────────────────────
    with tab_logs:
        st.markdown("## 🧾 Event Log")
        all_logs = st.session_state.get("logs", [])
        if run_logs:
            all_logs = run_logs + all_logs
        if all_logs:
            st.text_area(
                "Logs",
                value="\n".join(all_logs[-100:]),
                height=500,
                disabled=True,
                label_visibility="collapsed",
            )
            if st.button("🗑️ Clear logs"):
                st.session_state["logs"] = []
                st.rerun()
        else:
            st.info("No logs yet.")

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">✍️</div>
            <h2>Ready to write your next blog post?</h2>
            <p>Enter a topic in the sidebar and click <strong>Generate Blog</strong>.</p>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <div class="feature-label">Web Research</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🧩</div>
                    <div class="feature-label">AI Outline</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-label">Parallel Writing</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🎨</div>
                    <div class="feature-label">SEO Optimized</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📱</div>
                    <div class="feature-label">Social Content</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⭐</div>
                    <div class="feature-label">Quality Review</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
