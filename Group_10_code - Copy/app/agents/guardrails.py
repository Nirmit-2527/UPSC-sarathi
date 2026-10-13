"""
upsc-sarathi-agentic :: app/agents/guardrails.py

CrewAI native guardrails: `Task(guardrail=callable, guardrail_max_retries=N)`.
Verified signature (crewai==1.14.7): the callable receives the TaskOutput
(or LiteAgentOutput) produced by the task and must return a tuple of
`(success: bool, result: Any)`. If `success` is False, CrewAI automatically
re-runs the task (feeding the agent the failure message) up to
`guardrail_max_retries` times before giving up - which is exactly the retry
behaviour we'd otherwise have had to hand-roll.

This is the mechanism that enforces the user's explicit mains-answer rules:
  4f) ~250-300 words total (word_count.marks_to_word_range)
  4g) ~1500 word essay (word_count.essay_word_range)
  4d) body in POINTS, not paragraphs        -> structural check on body_points
  4e) every point substantiated with evidence -> non-empty, non-trivial evidence per point

`groundedness_guardrail` below goes one step further than the structural
checks above: it actually verifies each evidence claim against the
retrieved context, using a nested one-task Crew built around the
FactChecker agent (app/agents/agents.py::fact_checker_agent, the project's
14th agent). Guardrails are just Python callables, so calling
`Crew(...).kickoff()` *inside* one - synchronously, mid-execution of the
outer Formatter task - is fully supported by CrewAI; this was verified
directly against a real installed crewai==1.14.7 before being wired in here
(a mocked-LLM nested-kickoff-inside-a-guardrail test was run to confirm the
outer task's guardrail-retry loop and the nested crew's own kickoff() don't
conflict or deadlock).
"""
# NOTE: do NOT add `from __future__ import annotations` to this module.
# crewai's Task(guardrail=...) validator inspects the guardrail's return
# annotation at runtime and requires the real object Tuple[bool, Any]; with
# postponed evaluation the annotation becomes the *string* "Tuple[bool, Any]"
# and Task() raises "If return type is annotated, it must be Tuple[bool, Any]".
from typing import Tuple, Any, List

from crewai import Task, Crew, Process
from crewai.tasks.task_output import TaskOutput

from app.config import settings
from app.agents.agents import fact_checker_agent
from app.models.agent_schemas import MainsAnswer, EssayAnswer, VerificationResult
from app.utils.word_count import (
    count_words,
    marks_to_word_range,
    essay_word_range,
    mains_answer_text,
    essay_text,
)


def _get_pydantic(output: TaskOutput, model_cls):
    """Pull the structured pydantic object out of a TaskOutput, tolerating
    both the `.pydantic` attribute (already-parsed) and a raw `.json_dict`
    fallback, since guardrails run before the calling code does its own
    parsing."""
    if getattr(output, "pydantic", None) is not None:
        return output.pydantic
    if getattr(output, "json_dict", None):
        return model_cls(**output.json_dict)
    # last resort: try raw text as JSON
    import json
    return model_cls(**json.loads(output.raw))


def mains_answer_guardrail(marks: int = 15):
    """Returns a guardrail callable bound to a specific mark-weightage
    (different marks -> different word_count target, per UPSC convention)."""
    lo, hi = marks_to_word_range(marks)

    def _guardrail(output: TaskOutput) -> Tuple[bool, Any]:
        try:
            answer: MainsAnswer = _get_pydantic(output, MainsAnswer)
        except Exception as e:
            return False, f"Could not parse a valid MainsAnswer JSON object: {e}. Re-emit strictly as the required schema."

        errors = []

        if len(answer.body_points) < 3:
            errors.append(
                f"The body has only {len(answer.body_points)} point(s). UPSC mains answers must have at least 3-5 "
                "distinct points covering the full demand of the question - rewrite the body as more discrete points."
            )

        weak_evidence = [
            bp.point[:60] for bp in answer.body_points
            if not bp.evidence or len(bp.evidence.strip()) < 10
        ]
        if weak_evidence:
            errors.append(
                "Every body point MUST be substantiated with concrete evidence (a constitutional article, court "
                "judgment, committee/commission report, scheme, statistic, or parliamentary report) - these points "
                f"are missing real evidence: {weak_evidence}. Add specific evidence to each."
            )

        if not answer.introduction or len(answer.introduction.strip()) < 15:
            errors.append("The introduction is missing or too short - it must be a catchy 2-3 line opening relevant to the question.")

        if not answer.conclusion or len(answer.conclusion.strip()) < 15:
            errors.append("The conclusion is missing or too short - it must be a balanced, forward-looking 2-3 line closer.")

        text = mains_answer_text(answer)
        wc = count_words(text)
        if wc < lo or wc > hi:
            errors.append(
                f"Total word count is {wc}, but a {marks}-mark UPSC mains answer must be between {lo} and {hi} "
                "words. Expand or trim the body points (do not pad the intro/conclusion) to hit this range."
            )

        if errors:
            return False, " ".join(errors)

        answer.word_count = wc
        return True, answer

    return _guardrail


def _run_fact_check(claims: List[tuple], source_text: str) -> VerificationResult:
    """claims: list of (point, evidence) tuples to check against source_text.

    Runs a tiny one-task Crew built around the FactChecker agent. Kept as a
    standalone function (not inlined into the guardrail) so it's
    independently callable/testable without needing a full guardrail/Task
    context around it.
    """
    if not claims or not source_text:
        # Nothing to verify against - don't block the answer on a missing
        # source (the structural guardrail above already enforces that
        # evidence fields are non-empty; this can only check groundedness
        # when there is actual source material to check against).
        return VerificationResult(verified_claims=[], all_supported=True, unsupported_claims=[])

    claims_block = "\n".join(
        f'{i + 1}. Point: "{point}" | Evidence claim: "{evidence}"'
        for i, (point, evidence) in enumerate(claims)
    )
    task = Task(
        description=(
            "You are given a set of evidence claims from a UPSC mains answer, and the source material the "
            "answer was supposedly drawn from.\n\n"
            f'SOURCE MATERIAL:\n"""\n{source_text}\n"""\n\n'
            f"CLAIMS TO VERIFY:\n{claims_block}\n\n"
            "For EACH claim, decide if it is genuinely supported by the source material above - never by whether "
            "it sounds plausible or is true in the real world, only by whether THIS source material actually "
            "contains or directly implies it. If the source material does not mention it, mark it unsupported "
            "even if it seems like a perfectly reasonable, true fact. In matching_source, quote or closely "
            "paraphrase the supporting part of the source when supported; briefly explain why not when "
            "unsupported."
        ),
        expected_output=(
            f"A VerificationResult JSON object with exactly {len(claims)} ClaimVerification entries, one per "
            "claim listed above, in the same order, plus all_supported and unsupported_claims filled in correctly."
        ),
        agent=fact_checker_agent(),
        output_pydantic=VerificationResult,
    )
    crew = Crew(agents=[fact_checker_agent()], tasks=[task], process=Process.sequential, verbose=settings.agents_verbose)
    crew.kickoff()

    result = task.output.pydantic
    if result is None:
        # Fail open on a verifier-side parsing error rather than blocking
        # the answer indefinitely on a problem in the checker itself - the
        # structural guardrail's evidence-non-empty check still applies
        # regardless.
        return VerificationResult(verified_claims=[], all_supported=True, unsupported_claims=[])
    return result


def groundedness_guardrail(context_task):
    """Returns a guardrail callable that verifies every BodyPoint.evidence
    in a MainsAnswer against the CondensedContext.summary produced earlier
    in the same crew by `context_task` (the Summarizer task - see
    app/agents/crews/mains_answer_crew.py). `context_task` is the live
    CrewAI Task object, passed in *before* the crew runs; because CrewAI's
    sequential process runs tasks in order, by the time this guardrail
    fires (after the Formatter task, which comes later in the task list),
    `context_task.output` is already populated.

    What this does and does not check: it verifies claims against
    CondensedContext.summary - the Summarizer's condensation of the
    retrieved chunks - not against the raw retrieved chunk text directly.
    The Summarizer agent is explicitly prompted to losslessly preserve
    every fact/figure/citation, so the summary is a reasonable, cheap proxy
    for "was this actually retrieved" without re-plumbing raw chunks
    through every downstream task. A stricter (and more expensive) version
    would check straight against the raw RetrievedChunk text instead.
    """
    def _guardrail(output: TaskOutput) -> Tuple[bool, Any]:
        try:
            answer: MainsAnswer = _get_pydantic(output, MainsAnswer)
        except Exception as e:
            return False, f"Could not parse a valid MainsAnswer JSON object: {e}. Re-emit strictly as the required schema."

        source_text = ""
        try:
            context_output = context_task.output
            condensed = context_output.pydantic if context_output is not None else None
            if condensed is not None:
                source_text = condensed.summary
            else:
                source_text = getattr(context_output, "raw", "") or ""
        except Exception:
            source_text = ""

        if not source_text:
            # No condensed context available to check against (shouldn't
            # normally happen - this guardrail is only ever wired up
            # downstream of a Summarizer task). Don't fail the answer over
            # a plumbing gap rather than an actual grounding problem.
            return True, answer

        claims = [(bp.point, bp.evidence) for bp in answer.body_points]
        verification = _run_fact_check(claims, source_text)

        if not verification.all_supported and verification.unsupported_claims:
            return False, (
                "The Evidence Verifier could not confirm the following claims against the retrieved source "
                f"material - they appear fabricated rather than retrieved: {verification.unsupported_claims}. "
                "Revise these points to use only evidence that is actually present in the retrieved context, or "
                "replace them with points you can substantiate from what was actually retrieved."
            )
        return True, answer

    return _guardrail


def combine_guardrails(*guardrails):
    """Composes multiple guardrail callables into one. CrewAI's `Task`
    accepts only a single `guardrail=` callable, but this project wants to
    check several independent concerns on the same Formatter task output -
    structure/word-count (mains_answer_guardrail) AND groundedness
    (groundedness_guardrail). Every guardrail in the chain is run
    regardless of earlier failures, so a single retry gets the full list of
    problems to fix at once rather than discovering them one at a time
    across several retries (which would burn through guardrail_max_retries
    needlessly).
    """
    def _combined(output: TaskOutput) -> Tuple[bool, Any]:
        errors = []
        last_ok_result = None
        for g in guardrails:
            ok, result = g(output)
            if ok:
                last_ok_result = result
            else:
                errors.append(str(result))
        if errors:
            return False, " ".join(errors)
        return True, last_ok_result

    return _combined


def essay_guardrail():
    lo, hi = essay_word_range()

    def _guardrail(output: TaskOutput) -> Tuple[bool, Any]:
        try:
            essay: EssayAnswer = _get_pydantic(output, EssayAnswer)
        except Exception as e:
            return False, f"Could not parse a valid EssayAnswer JSON object: {e}. Re-emit strictly as the required schema."

        errors = []

        if len(essay.body_paragraphs) < 4:
            errors.append(
                f"Only {len(essay.body_paragraphs)} body paragraph(s) were written. A UPSC essay must explore at "
                "least 4-7 distinct dimensions (e.g. economic, social, ethical, historical, global/geopolitical, "
                "technological) - add more dimensions."
            )

        dims = {p.dimension.strip().lower() for p in essay.body_paragraphs}
        if len(dims) < len(essay.body_paragraphs):
            errors.append("Some body paragraphs repeat the same dimension - each paragraph must cover a genuinely distinct angle.")

        if not essay.introduction or len(essay.introduction.strip()) < 30:
            errors.append("The introduction must be a substantive, engaging hook paragraph that sets up the thesis, not a one-liner.")

        if not essay.conclusion or len(essay.conclusion.strip()) < 30:
            errors.append("The conclusion must be a substantive, visionary, synthesising paragraph, not a one-liner.")

        text = essay_text(essay)
        wc = count_words(text)
        if wc < lo or wc > hi:
            errors.append(
                f"Total essay word count is {wc}, but the target is {lo}-{hi} words. Expand or trim body paragraphs "
                "to hit this range - do not pad the introduction/conclusion artificially."
            )

        if errors:
            return False, " ".join(errors)

        essay.word_count = wc
        return True, essay

    return _guardrail
