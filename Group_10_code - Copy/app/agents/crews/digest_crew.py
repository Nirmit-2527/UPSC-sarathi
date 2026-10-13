"""
upsc-sarathi-agentic :: app/agents/crews/digest_crew.py

Feature #5 extension ("Daily current-affairs PDF upload/update into VectorDB
... Freedom to add/modify creative features"). When a current-affairs PDF is
ingested, this crew summarises it into a compact DigestOutput, which the
ingestion pipeline can optionally write back into the vector DB tagged
doc_type="digest" - giving future queries a fast, compact current-affairs
signal instead of always re-reading full raw articles.

This crew works directly off raw page text passed in by the ingestion
pipeline (NOT via vector_search), since at ingestion time the document isn't
guaranteed to be in the vector DB yet for the same session.
"""
from __future__ import annotations

from typing import Dict, Any

from crewai import Task, Crew, Process

from app.config import settings
from app.agents.agents import digest_curator_agent
from app.models.agent_schemas import DigestOutput
from app.services.trace import build_trace


def run_digest(source_file: str, date_tag: str, raw_text: str) -> Dict[str, Any]:
    # Keep the prompt bounded - a digest doesn't need every character of a
    # long current-affairs compilation, just enough to extract the signal.
    excerpt = raw_text[:12000]
    task = Task(
        description=(
            f'Summarise this current-affairs document ("{source_file}", dated {date_tag}) into a compact digest:\n\n'
            f'"""\n{excerpt}\n"""\n\n'
            "Produce: a headline_summary (2-4 sentences), prelims_facts (crisp, memorisable factual bullets), "
            "mains_points (analytical points an aspirant could use in a mains answer), and gs_paper_tags (which "
            "GS papers this is relevant to, e.g. GS-II, GS-III)."
        ),
        expected_output="A DigestOutput JSON object.",
        agent=digest_curator_agent(),
        output_pydantic=DigestOutput,
    )
    crew = Crew(agents=[digest_curator_agent()], tasks=[task], process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    digest: DigestOutput = task.output.pydantic
    digest.source_file = source_file
    digest.date_tag = date_tag
    return {"digest": digest, "agent_trace": build_trace(out)}
