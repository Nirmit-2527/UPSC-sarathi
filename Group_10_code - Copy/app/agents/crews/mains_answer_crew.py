"""Generate source-grounded, schema-validated UPSC mains answers."""
from __future__ import annotations

from typing import Dict, Any, Optional
from crewai import Task, Crew, Process
from app.config import settings
from app.agents.agents import (context_summarizer_agent, answer_writer_agent, upsc_formatter_agent, pyq_matcher_agent, structured_output_agent)
from app.agents.guardrails import mains_answer_guardrail, groundedness_guardrail, combine_guardrails
from app.models.agent_schemas import CondensedContext, MainsAnswer, PYQMatchResult
from app.utils.word_count import marks_to_word_range
from app.services.trace import build_trace


def run_mains_answer(question: str, marks: int = 15, gs_paper: Optional[str] = None, include_pyq: bool = True) -> Dict[str, Any]:
    lo, hi = marks_to_word_range(marks)
    structured_agent = structured_output_agent()
    research_task = Task(
        description=(
            f'Use vector_search to research this UPSC mains question: "{question}". '
            f'{f"GS paper: {gs_paper}. " if gs_paper else ""}Retrieve broadly (k=8-10). Return a faithful research brief '
            'preserving source/page references, facts, article numbers, judgments, reports, schemes, statistics, and coverage gaps. '
            "Do not output JSON; the next task will structure this brief."
        ),
        expected_output="A detailed source-grounded research brief with source/page citations and gaps.",
        agent=context_summarizer_agent(),
    )
    context_task = Task(
        description=(
            "Convert the research brief into CondensedContext with summary, key_citations, and coverage_gaps. "
            "Preserve all factual details and citations. Never add facts not in the brief."
        ),
        expected_output="A CondensedContext object.", agent=structured_agent, context=[research_task], output_pydantic=CondensedContext,
    )
    writer_task = Task(
        description=(
            f'Using ONLY the CondensedContext produced previously, draft a complete answer to: "{question}". '
            "Address every part of the question and substantiate each claim with evidence present in that context. "
            "Flag any demand the sources do not cover."
        ),
        expected_output="A thorough prose draft covering all dimensions with source-grounded evidence.",
        agent=answer_writer_agent(), context=[context_task],
    )
    formatter_task = Task(
        description=(
            f'Reformat the draft for "{question}" into the MainsAnswer schema. Write a catchy 2-3 line introduction; '
            "the entire body as discrete BodyPoint items, each with a specific evidence field and valid evidence_type; "
            "and a balanced 2-3 line conclusion. "
            f"Target {lo}-{hi} total words for this {marks}-mark question. Use at least 3 body points. Set word_count accurately."
        ),
        expected_output="A MainsAnswer object meeting the schema and word-count target.",
        agent=upsc_formatter_agent(), context=[writer_task, context_task], output_pydantic=MainsAnswer,
        guardrail=(combine_guardrails(mains_answer_guardrail(marks), groundedness_guardrail(context_task)) if settings.enable_groundedness_check else mains_answer_guardrail(marks)),
        guardrail_max_retries=settings.guardrail_max_retries,
    )
    tasks = [research_task, context_task, writer_task, formatter_task]
    agents = [context_summarizer_agent(), structured_agent, answer_writer_agent(), upsc_formatter_agent()]
    pyq_format_task = None
    if include_pyq and settings.enable_pyq_matching_in_qa:
        pyq_search_task = Task(
            description=(f'Search the PYQ bank for previous year questions related to "{question}". Return retrieved matches and any supported recurrence pattern. Do not output JSON.'),
            expected_output="A plain-text PYQ research brief.", agent=pyq_matcher_agent(),
        )
        pyq_format_task = Task(
            description=(
                "Convert the PYQ research brief into PYQMatchResult. Only include matches actually present in the brief; "
                "do not invent years, papers, or frequency counts. If no results were found, use an empty matches list."
            ),
            expected_output="A PYQMatchResult object.", agent=structured_agent, context=[pyq_search_task], output_pydantic=PYQMatchResult,
        )
        tasks.extend([pyq_search_task, pyq_format_task])
        agents.append(pyq_matcher_agent())

    crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    mains_answer: MainsAnswer = formatter_task.output.pydantic
    mains_answer.marks = marks
    mains_answer.gs_paper = gs_paper
    pyq_result = pyq_format_task.output.pydantic if pyq_format_task is not None else None
    return {"mains_answer": mains_answer, "pyq_matches": pyq_result, "agent_trace": build_trace(out), "groundedness_checked": settings.enable_groundedness_check}
