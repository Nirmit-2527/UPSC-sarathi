"""Semantic PYQ search followed by tool-free structured formatting."""
from __future__ import annotations
from typing import Dict, Any
from crewai import Task, Crew, Process
from app.config import settings
from app.agents.agents import pyq_matcher_agent, structured_output_agent
from app.models.agent_schemas import PYQMatchResult
from app.services.trace import build_trace


def run_pyq_match(topic: str, k: int = 10) -> Dict[str, Any]:
    search_agent = pyq_matcher_agent()
    formatter = structured_output_agent()
    search_task = Task(
        description=(f'Use pyq_search to retrieve up to {k} previous year UPSC questions related to "{topic}". Return question text and only those year/exam_type/paper values supported by the retrieved source filename or content. Report recurrence patterns only when supported. Do not output JSON.'),
        expected_output="A plain-text PYQ research brief based on retrieved matches.", agent=search_agent,
    )
    format_task = Task(
        description=(f'Convert the research brief into PYQMatchResult for topic "{topic}". Include only retrieved matches; do not invent years, papers, or frequency counts. If no matches were found, use an empty matches list and state that clearly in frequency_insight.'),
        expected_output="A PYQMatchResult object.", agent=formatter, context=[search_task], output_pydantic=PYQMatchResult,
    )
    crew = Crew(agents=[search_agent, formatter], tasks=[search_task, format_task], process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    return {"pyq_result": format_task.output.pydantic, "agent_trace": build_trace(out)}
