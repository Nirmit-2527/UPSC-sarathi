"""Generate UPSC question batches with separate retrieval and schema formatting."""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from crewai import Task, Crew, Process
from app.config import settings
from app.agents.agents import mcq_setter_agent, mains_question_setter_agent, structured_output_agent
from app.models.agent_schemas import MCQBatch, MainsQuestionBatch
from app.services.trace import build_trace


def _run_batch(research_description: str, expected: str, schema, research_agent, result_key: str):
    formatter = structured_output_agent()
    research_task = Task(
        description=(research_description + "\nSearch with the available vector_search tool and return a source-grounded research brief. Do not produce the final structured JSON; a separate tool-free task will format it."),
        expected_output="A research brief with relevant retrieved facts, examples, and source references.",
        agent=research_agent,
    )
    format_task = Task(
        description=(
            f"Using only the research brief from the previous task, produce {expected}. "
            "Follow the requested schema exactly. Do not invent factual details or citations not supported by the brief. "
            "If the brief is incomplete, be transparent in sources and avoid unsupported claims."
        ),
        expected_output=expected, agent=formatter, context=[research_task], output_pydantic=schema,
    )
    crew = Crew(agents=[research_agent, formatter], tasks=[research_task, format_task], process=Process.sequential, verbose=settings.agents_verbose)
    out = crew.kickoff()
    return {result_key: format_task.output.pydantic, "agent_trace": build_trace(out)}


def run_mcq_batch(topic: str, n: int = 10, subject: Optional[str] = None) -> Dict[str, Any]:
    desc = (f'Find UPSC prelims source material on "{topic}"{f" (subject filter: {subject})" if subject else ""}. The final batch must contain exactly {n} rigorous MCQs, mixing single-best-answer and statement-based formats, with four options, correct answer, explanation, difficulty, and syllabus tag.')
    return _run_batch(desc, f"an MCQBatch JSON object with exactly {n} questions", MCQBatch, mcq_setter_agent(), "mcq_batch")


def run_mains_question_batch(topic: str, n: int = 5, gs_paper: Optional[str] = None) -> Dict[str, Any]:
    desc = (f'Research UPSC mains material on "{topic}"{f" (GS paper: {gs_paper})" if gs_paper else ""}. The final batch must contain exactly {n} exam-realistic mains questions, each with marks, word limit, GS paper, model outline, key points, and relevant recent examples only when supported by the brief.')
    return _run_batch(desc, f"a MainsQuestionBatch JSON object with exactly {n} questions", MainsQuestionBatch, mains_question_setter_agent(), "mains_question_batch")


def run_prelims_paper(topics: List[str], n: int = 20) -> Dict[str, Any]:
    topics_str = ", ".join(topics)
    desc = f"Research source material across these topics: {topics_str}. The final prelims mock paper must contain exactly {n} MCQs distributed roughly evenly across the topics, with subject_distribution populated and a realistic mix of difficulty."
    return _run_batch(desc, f"an MCQBatch JSON object with exactly {n} questions and populated subject_distribution", MCQBatch, mcq_setter_agent(), "mcq_batch")


def run_mains_paper(topic: str, gs_paper: str = "GS-II", n: int = 5) -> Dict[str, Any]:
    desc = f'Research material on "{topic}" for {gs_paper}. The final paper must contain exactly {n} questions for {gs_paper}, with marks, word limits, model outlines, and key points.'
    result = _run_batch(desc, f"a MainsQuestionBatch JSON object with exactly {n} questions, all tagged gs_paper={gs_paper}, and total_marks populated", MainsQuestionBatch, mains_question_setter_agent(), "mains_question_batch")
    batch = result["mains_question_batch"]
    for question in batch.questions:
        question.gs_paper = gs_paper
    batch.total_marks = batch.total_marks if batch.total_marks is not None else sum(q.marks for q in batch.questions)
    return result
