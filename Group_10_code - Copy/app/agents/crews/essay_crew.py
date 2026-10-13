"""Generate a multidimensional UPSC essay using a research-then-format pipeline."""
from __future__ import annotations
from typing import Dict, Any
from crewai import Task, Crew, Process
from app.config import settings
from app.agents.agents import essay_architect_agent, structured_output_agent
from app.agents.guardrails import essay_guardrail
from app.models.agent_schemas import EssayAnswer
from app.utils.word_count import essay_word_range
from app.services.trace import build_trace


def run_essay(topic: str) -> Dict[str, Any]:
    lo, hi = essay_word_range()
    researcher = essay_architect_agent()
    formatter = structured_output_agent()
    research_task = Task(
        description=(f'Research the UPSC essay topic "{topic}" using vector_search several times across different angles: economic, social, ethical, historical, global/geopolitical, technological, philosophical/values-based. Return a source-grounded research brief with examples, data, and citations grouped by dimension. Do not write the essay JSON.'),
        expected_output="A multidimensional research brief with evidence and citations.", agent=researcher,
    )
    essay_task = Task(
        description=(
            f'Using only the preceding research brief, write a UPSC essay on "{topic}" as an EssayAnswer. '
            "Write an engaging hook introduction that establishes a thesis; 4-7 flowing-prose body paragraphs, each "
            "covering a distinct dimension and weaving in supported examples/data; and a visionary synthesising conclusion. "
            f"Target {lo}-{hi} words total. Do not invent specific data or citations absent from the brief."
        ),
        expected_output=f"An EssayAnswer object with introduction, 4-7 body_paragraphs, conclusion, and word_count; total {lo}-{hi} words.",
        agent=formatter, context=[research_task], output_pydantic=EssayAnswer,
        guardrail=essay_guardrail(), guardrail_max_retries=settings.guardrail_max_retries,
    )
    crew = Crew(agents=[researcher, formatter], tasks=[research_task, essay_task], process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    essay: EssayAnswer = essay_task.output.pydantic
    return {"essay": essay, "agent_trace": build_trace(out)}
