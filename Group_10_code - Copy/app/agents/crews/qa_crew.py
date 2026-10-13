"""Grounded QA pipeline with Groq-compatible structured output handling."""
from __future__ import annotations

from typing import Dict, Any

from crewai import Task, Crew, Process

from app.config import settings
from app.agents.agents import (
    router_agent, direct_answerer_agent, context_summarizer_agent,
    answer_writer_agent, pyq_matcher_agent, structured_output_agent,
)
from app.models.agent_schemas import RouteDecision, CondensedContext, PYQMatchResult
from app.services.trace import build_trace


def _route(question: str):
    task = Task(
        description=(
            f'A UPSC aspirant asked: "{question}"\n\n'
            "Decide whether answering requires specific facts/syllabus/current affairs/PYQs from the vector database "
            "(use_rag=True), or whether it is a generic/meta/conversational question (use_rag=False). When unsure, "
            "prefer RAG. If use_rag is true, provide one short search query."
        ),
        expected_output="RouteDecision fields: use_rag, reasoning, search_query.",
        agent=router_agent(), output_pydantic=RouteDecision,
    )
    crew = Crew(agents=[router_agent()], tasks=[task], process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    decision = out.pydantic if out.pydantic is not None else RouteDecision(**out.json_dict)
    return decision, build_trace(out)


def _direct_answer(question: str):
    task = Task(
        description=f'Answer this UPSC aspirant question directly and helpfully: "{question}"',
        expected_output="A clear, accurate, concise answer.", agent=direct_answerer_agent(),
    )
    crew = Crew(agents=[direct_answerer_agent()], tasks=[task], process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    return out.raw, build_trace(out)


def _rag_answer(question: str, search_query: str, k: int, subject: str | None, include_pyq: bool):
    formatter = structured_output_agent()
    research_task = Task(
        description=(
            f'Use vector_search to find material relevant to "{search_query}" for this question: "{question}". '
            f'{f"Apply subject filter: {subject}. " if subject else ""}Retrieve k={k}. Return a faithful research brief '
            "that preserves facts, figures, dates, citations/source pages, and explicitly lists coverage gaps. "
            "Do not output a JSON object; return readable text for the next task to structure."
        ),
        expected_output="A source-grounded research brief with citations and coverage gaps.",
        agent=context_summarizer_agent(),
    )
    context_task = Task(
        description=(
            "Convert the preceding research brief into CondensedContext. Put the complete faithful summary in "
            "summary, source/page references in key_citations, and unsupported/uncovered parts in coverage_gaps. "
            "Do not add facts that are absent from the brief."
        ),
        expected_output="A CondensedContext object with summary, key_citations, coverage_gaps.",
        agent=formatter, context=[research_task], output_pydantic=CondensedContext,
    )
    writer_task = Task(
        description=(
            f'Using ONLY the condensed context from the previous task, answer: "{question}". Address each dimension, '
            "cite source/page where available, and acknowledge coverage gaps rather than inventing facts."
        ),
        expected_output="A clear, well-structured prose answer grounded strictly in the provided context.",
        agent=answer_writer_agent(), context=[context_task],
    )
    tasks = [research_task, context_task, writer_task]
    agents = [context_summarizer_agent(), formatter, answer_writer_agent()]
    pyq_format_task = None
    if include_pyq and settings.enable_pyq_matching_in_qa:
        pyq_search_task = Task(
            description=(
                f'Search the PYQ bank for previous year questions related to "{search_query}". Return matches, '
                "available years/papers, and any frequency pattern supported by retrieved results. Do not emit JSON."
            ),
            expected_output="A plain-text PYQ research brief based on retrieved matches.", agent=pyq_matcher_agent(),
        )
        pyq_format_task = Task(
            description=(
                "Convert the preceding PYQ research brief into PYQMatchResult. Include only matches present in the "
                "brief; do not infer missing years or claim unsupported frequency counts. If no matches were found, "
                "return an empty matches list and state that in frequency_insight."
            ),
            expected_output="A PYQMatchResult object with query_topic, matches, frequency_insight.",
            agent=formatter, context=[pyq_search_task], output_pydantic=PYQMatchResult,
        )
        tasks.extend([pyq_search_task, pyq_format_task])
        agents.append(pyq_matcher_agent())

    crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    condensed: CondensedContext = context_task.output.pydantic
    pyq_result = pyq_format_task.output.pydantic if pyq_format_task is not None else None
    return writer_task.output.raw, condensed, pyq_result, build_trace(out)


def run_qa(question: str, k: int = 6, subject: str | None = None, include_pyq: bool = True) -> Dict[str, Any]:
    decision, route_trace = _route(question)
    trace = list(route_trace)
    if not decision.use_rag:
        answer, sub_trace = _direct_answer(question)
        trace.extend(sub_trace)
        return {"answer": answer, "used_rag": False, "route_reason": decision.reasoning, "key_citations": [], "pyq_matches": None, "agent_trace": trace}
    answer, condensed, pyq_result, sub_trace = _rag_answer(question, decision.search_query, k, subject, include_pyq)
    trace.extend(sub_trace)
    return {"answer": answer, "used_rag": True, "route_reason": decision.reasoning, "key_citations": condensed.key_citations if condensed else [], "pyq_matches": pyq_result, "agent_trace": trace}
