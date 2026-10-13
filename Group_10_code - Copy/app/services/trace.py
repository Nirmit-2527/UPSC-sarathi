"""
upsc-sarathi-agentic :: app/services/trace.py

Turns a CrewAI `CrewOutput` (or a single `TaskOutput`) into a small list of
plain dicts the Streamlit frontend can render as an "agent trace" - showing
which agent did what, in order. This is purely cosmetic/explanatory (great
for demos/interviews: "here's what the agents actually did") and never
affects the actual answer returned to the user.
"""
from __future__ import annotations

from typing import List, Dict, Any


def _one(task_output) -> Dict[str, Any]:
    raw = task_output.raw or ""
    return {
        "agent": task_output.agent,
        "task": task_output.name or (task_output.summary or "")[:80],
        "output_preview": raw[:280] + ("…" if len(raw) > 280 else ""),
    }


def build_trace(crew_output) -> List[Dict[str, Any]]:
    """Accepts a CrewOutput (has .tasks_output) and returns the trace list."""
    tasks_output = getattr(crew_output, "tasks_output", None)
    if not tasks_output:
        return []
    return [_one(t) for t in tasks_output]
