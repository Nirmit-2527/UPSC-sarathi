"""Create source-grounded UPSC notes without mixing tools and structured output."""
from __future__ import annotations
from typing import Dict, Any, Optional
from crewai import Task, Crew, Process
from app.config import settings
from app.agents.agents import context_summarizer_agent, structured_output_agent
from app.models.agent_schemas import CondensedContext, NotesOutput
from app.services.trace import build_trace


def run_notes(topic: str, subject: Optional[str] = None) -> Dict[str, Any]:
    formatter = structured_output_agent()
    research_task = Task(
        description=(f'Use vector_search to find material on "{topic}"{f" (subject filter: {subject})" if subject else ""}. Retrieve broadly (k=8-10). Return a dense, faithful research brief preserving facts, figures, dates, citations/source pages, and gaps. Do not output JSON.'),
        expected_output="A source-grounded research brief.", agent=context_summarizer_agent(),
    )
    context_task = Task(
        description="Convert the research brief into CondensedContext with summary, key_citations, and coverage_gaps. Preserve all facts and do not invent missing details.",
        expected_output="A CondensedContext object.", agent=formatter, context=[research_task], output_pydantic=CondensedContext,
    )
    notes_task = Task(
        description=(f'Using ONLY the CondensedContext above, create topper-style notes on "{topic}". Use clear headers and crisp bullets (no paragraph over 3 lines), likely mains question framings in mains_angle, and memorisable prelims facts in prelims_facts. Include source references where present.'),
        expected_output="A NotesOutput object.", agent=formatter, context=[context_task], output_pydantic=NotesOutput,
    )
    crew = Crew(agents=[context_summarizer_agent(), formatter], tasks=[research_task, context_task, notes_task], process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    return {"notes": notes_task.output.pydantic, "agent_trace": build_trace(out)}
