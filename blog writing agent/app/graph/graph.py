"""
app/graph/graph.py
──────────────────
Assembles the full LangGraph blog-writing pipeline and exposes `app`.

Graph flow:
  START
    → router
    → (research | orchestrator)     [conditional]
    → orchestrator
    → worker × N                    [parallel fan-out via Send]
    → reducer (merge → decide_images → generate_and_place_images)
    → seo
    → social
    → reviewer
    → END
"""

from __future__ import annotations

import operator
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.models import State
from app.graph.router import router_node, route_next
from app.graph.research import research_node
from app.graph.orchestrator import orchestrator_node
from app.graph.worker import worker_node
from app.graph.reducer import merge_content, decide_images, generate_and_place_images
from app.graph.seo import seo_node
from app.graph.social import social_node
from app.graph.reviewer import quality_reviewer_node
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Fan-out: spawn one worker per task ────────────────────────────────────────

def fanout(state: State):
    """Return a Send message per task in the plan for parallel execution."""
    assert state["plan"] is not None, "fanout called without a plan"
    logger.info("Fanout | spawning %d parallel workers", len(state["plan"].tasks))
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


# ── Reducer subgraph ──────────────────────────────────────────────────────────

_reducer = StateGraph(State)
_reducer.add_node("merge_content", merge_content)
_reducer.add_node("decide_images", decide_images)
_reducer.add_node("generate_and_place_images", generate_and_place_images)
_reducer.add_edge(START, "merge_content")
_reducer.add_edge("merge_content", "decide_images")
_reducer.add_edge("decide_images", "generate_and_place_images")
_reducer.add_edge("generate_and_place_images", END)
reducer_subgraph = _reducer.compile()


# ── Main graph ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(State)

    g.add_node("router", router_node)
    g.add_node("research", research_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("worker", worker_node)
    g.add_node("reducer", reducer_subgraph)
    g.add_node("seo", seo_node)
    g.add_node("social", social_node)
    g.add_node("reviewer", quality_reviewer_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        route_next,
        {"research": "research", "orchestrator": "orchestrator"},
    )
    g.add_edge("research", "orchestrator")
    g.add_conditional_edges("orchestrator", fanout, ["worker"])
    g.add_edge("worker", "reducer")
    g.add_edge("reducer", "seo")
    g.add_edge("seo", "social")
    g.add_edge("social", "reviewer")
    g.add_edge("reviewer", END)

    return g


# Compiled app — import this in the frontend
app = build_graph().compile()
logger.info("LangGraph app compiled successfully.")
