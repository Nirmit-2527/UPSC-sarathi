"""
upsc-sarathi-agentic :: app/agents/crews/syllabus_crew.py

Carried forward from the non-agentic reference project's syllabus-mapper
feature, now as a one-task crew using the "fast" LLM tier (it's a
classification-style task, not a generative-writing one).
"""
from __future__ import annotations

from typing import Dict, Any

from crewai import Task, Crew, Process

from app.config import settings
from app.agents.agents import syllabus_mapper_agent
from app.models.agent_schemas import SyllabusMap
from app.services.trace import build_trace


def run_syllabus_map(text: str) -> Dict[str, Any]:
    task = Task(
        description=(
            f'Map the following topic/text onto the UPSC syllabus:\n\n"""\n{text}\n"""\n\n'
            "Identify the specific prelims topics it touches, the GS1/GS2/GS3/GS4 mains topics, possible essay "
            "themes it could feed into, and how it links to current affairs."
        ),
        expected_output="A SyllabusMap JSON object.",
        agent=syllabus_mapper_agent(),
        output_pydantic=SyllabusMap,
    )
    crew = Crew(agents=[syllabus_mapper_agent()], tasks=[task], process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    return {"syllabus_map": task.output.pydantic, "agent_trace": build_trace(out)}
