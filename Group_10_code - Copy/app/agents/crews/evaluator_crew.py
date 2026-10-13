"""Answer evaluation with separate evidence retrieval and schema formatting."""
from __future__ import annotations
from typing import Dict, Any
from crewai import Task, Crew, Process
from app.config import settings
from app.agents.agents import mains_evaluator_agent, essay_evaluator_agent, structured_output_agent
from app.models.agent_schemas import MainsEvaluation, EssayEvaluation
from app.services.trace import build_trace


def run_mains_evaluation(question: str, answer_text: str) -> Dict[str, Any]:
    evaluator = mains_evaluator_agent()
    formatter = structured_output_agent()
    research_task = Task(
        description=(f'A candidate was asked: "{question}"\nTheir answer: """\n{answer_text}\n"""\nUse vector_search to check facts and find reference material the answer should have cited. Evaluate content/coverage (5), structure (3), examples/evidence (3), language (2), conclusion (2), total (15). Give concrete strengths, weaknesses, missed points, improvement tips, and improved introduction/conclusion. Return a detailed prose evaluation and retrieved references; do not output JSON.'),
        expected_output="A detailed evidence-based evaluation brief with sub-scores and references.", agent=evaluator,
    )
    format_task = Task(
        description="Convert the evaluation brief into MainsEvaluation. Preserve the rubric and ensure sub-scores sum to total (out of 15). Do not invent reference_sources; use only those in the brief.",
        expected_output="A MainsEvaluation object with all required fields.", agent=formatter, context=[research_task], output_pydantic=MainsEvaluation,
    )
    crew = Crew(agents=[evaluator, formatter], tasks=[research_task, format_task], process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    return {"evaluation": format_task.output.pydantic, "agent_trace": build_trace(out)}


def run_essay_evaluation(topic: str, essay_text: str) -> Dict[str, Any]:
    task = Task(
        description=(f'A candidate wrote this essay on "{topic}":\n"""\n{essay_text}\n"""\nEvaluate content & relevance (35), multidimensionality (25), structure & coherence (25), language (25), conclusion (15), total (125). List strengths, weaknesses, improvement tips, and missing dimensions.'),
        expected_output="An EssayEvaluation object with scores, feedback, and missing dimensions.", agent=essay_evaluator_agent(), output_pydantic=EssayEvaluation,
    )
    crew = Crew(agents=[essay_evaluator_agent()], tasks=[task], process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    return {"evaluation": task.output.pydantic, "agent_trace": build_trace(out)}
