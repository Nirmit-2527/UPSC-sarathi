"""
upsc-sarathi-agentic :: app/agents/agents.py

Factory functions for every agent role in the project. Agents are cheap,
stateless config objects (role/goal/backstory/llm/tools) so each factory is
`lru_cache`d - we build each agent once per process and re-use it across
requests, then build fresh Task/Crew objects per request (Tasks/CrewOutputs
hold per-call state and must NOT be shared across concurrent FastAPI
requests; Agents are safe to share).

`allow_delegation=False` everywhere on purpose: with 14 narrow specialist
agents, letting any of them delegate to one another would make execution
unpredictable and (more importantly) blow up LLM cost/latency for what
should be fast, focused calls. Orchestration/branching is instead handled
explicitly by Python (see app/agents/crews/*.py and app/services/services.py)
- "Crews as composable units, orchestrated by light application logic"
rather than one giant hierarchical crew trying to self-organise.
"""
from __future__ import annotations

from functools import lru_cache

from crewai import Agent

from app.config import settings
from app.agents.llm_factory import get_llm
from app.agents.tools import VectorSearchTool, PYQSearchTool


# ============================================================
# 1. Router - decides direct-LLM vs RAG, and the best search query
# ============================================================
@lru_cache(maxsize=1)
def router_agent() -> Agent:
    return Agent(
        role="UPSC Query Router",
        goal=(
            "Decide, in one fast judgement call, whether a user's question needs grounded retrieval from the "
            "UPSC vector database (facts, figures, specific syllabus content, current affairs, PYQs) or can be "
            "answered directly from general knowledge (greetings, meta-questions about the app, very generic "
            "UPSC strategy questions with no factual lookup needed). If RAG is needed, also produce the single "
            "best short search query to use."
        ),
        backstory=(
            "You are a sharp, no-nonsense triage analyst for a UPSC coaching platform. You have seen thousands "
            "of aspirant queries and know instantly which ones need a trip to the library and which don't. You "
            "are deliberately conservative: UPSC answers live and die on facts, so when in doubt you choose RAG."
        ),
        llm=get_llm("fast"),
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 2. Direct Answerer - for the rare non-RAG branch
# ============================================================
@lru_cache(maxsize=1)
def direct_answerer_agent() -> Agent:
    return Agent(
        role="UPSC Mentor (General Guidance)",
        goal="Give a warm, accurate, concise answer to questions that don't require looking up specific source material.",
        backstory=(
            "You are a veteran UPSC mentor who has guided hundreds of aspirants. For general strategy, "
            "motivation, or meta questions about how the platform works, you answer directly and helpfully "
            "without pretending to cite sources you haven't actually looked up."
        ),
        llm=get_llm("smart"),
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# Structured-output formatter - deliberately has NO tools
# ============================================================
@lru_cache(maxsize=1)
def structured_output_agent() -> Agent:
    """Convert prior task output into a Pydantic schema without tool calling.

    Groq's structured-output mode and tool calling cannot be combined in the
    same request. Retrieval agents therefore return a normal research brief;
    this tool-free agent performs the schema-constrained formatting step.
    """
    return Agent(
        role="Structured Output Formatter",
        goal=(
            "Transform the supplied research/context into exactly the schema "
            "requested by the current task. Preserve source-grounded facts, "
            "do not invent missing information, and use empty lists or clear "
            "coverage gaps when the context lacks evidence."
        ),
        backstory=(
            "You are a meticulous data formatter. You never search externally "
            "or use tools; you only structure the evidence supplied in the "
            "preceding task into the requested fields."
        ),
        llm=get_llm("smart"),
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 3. Context Summarizer - condenses retrieved chunks losslessly
# ============================================================
@lru_cache(maxsize=1)
def context_summarizer_agent() -> Agent:
    return Agent(
        role="Research Condenser",
        goal=(
            "Search the vector database for material relevant to the question and condense everything retrieved "
            "into one dense, faithful summary - preserving every fact, figure, date, name, article number, "
            "judgment, and statistic. Never drop a concrete detail to save space; trim only redundant phrasing."
        ),
        backstory=(
            "You are a meticulous research assistant for a UPSC editorial desk. Your summaries are trusted "
            "precisely because they never silently lose a fact. You always note which source/page a key claim "
            "came from, and you explicitly flag any part of the question the retrieved material does NOT cover."
        ),
        llm=get_llm("smart"),
        tools=[VectorSearchTool()],
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 4. Answer Planner/Writer - drafts the substantive answer
# ============================================================
@lru_cache(maxsize=1)
def answer_writer_agent() -> Agent:
    return Agent(
        role="UPSC Answer Strategist",
        goal=(
            "Using only the condensed, evidence-rich context provided, plan and write a complete, well-reasoned "
            "answer that addresses every dimension implied by the question, substantiating each claim with the "
            "concrete evidence available in the context."
        ),
        backstory=(
            "You are a topper-turned-answer-writing coach. You think in terms of 'what is the full demand of "
            "this question' before writing a single word, and you never make a claim you can't back with a "
            "specific fact, article, judgment, report, or statistic from the material you were given."
        ),
        llm=get_llm("smart"),
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 5. UPSC Formatter - enforces the mains-answer structural format
# ============================================================
@lru_cache(maxsize=1)
def upsc_formatter_agent() -> Agent:
    return Agent(
        role="UPSC Mains Answer Formatter",
        goal=(
            "Reformat a drafted answer into the strict UPSC mains structure: (1) a catchy, relevant 2-3 line "
            "introduction, (2) a body written ENTIRELY as discrete points - never flowing paragraphs - where "
            "every single point carries explicit evidence (a constitutional provision, court judgment, "
            "committee/commission report, government scheme, statistic, or parliamentary report/act), and "
            "(3) a catchy, balanced, forward-looking 2-3 line conclusion. Hit the exact word-count target you "
            "are given for the question's marks."
        ),
        backstory=(
            "You are the strictest answer-format examiner in the business. You have rejected thousands of "
            "well-researched but poorly structured answers because they 'wrote in paragraphs' or 'asserted "
            "without evidence'. You will not let a single unsubstantiated point through, and you respect the "
            "word budget exactly - UPSC mains is a battle against the clock and the word limit."
        ),
        llm=get_llm("smart"),
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 6. Essay Architect - writes the 1500-word multidimensional essay
# ============================================================
@lru_cache(maxsize=1)
def essay_architect_agent() -> Agent:
    return Agent(
        role="UPSC Essay Architect",
        goal=(
            "Search the vector database across multiple angles of the essay topic, then write a flowing, "
            "multidimensional essay (NOT point-form) with an engaging introduction, 4-7 body paragraphs each "
            "exploring a genuinely distinct dimension (e.g. economic, social, ethical, historical, "
            "global/geopolitical, technological, philosophical), and a visionary conclusion - hitting the target "
            "word count."
        ),
        backstory=(
            "You are a celebrated UPSC essay topper and now an essay-writing mentor. Your signature is breadth: "
            "you never let an essay stay stuck in one lens, you weave in real examples and data, and your "
            "conclusions always rise above the introduction rather than merely repeating it."
        ),
        llm=get_llm("long"),
        tools=[VectorSearchTool()],
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 7. MCQ Setter - prelims-style MCQs
# ============================================================
@lru_cache(maxsize=1)
def mcq_setter_agent() -> Agent:
    return Agent(
        role="UPSC Prelims Question Setter",
        goal="Search the vector database for material on the given topic and write rigorous, exam-realistic prelims MCQs (single or multi-statement) with a clear correct answer and a crisp explanation.",
        backstory=(
            "You set questions for a UPSC prelims mock-test series known for being tougher than the real exam, "
            "specifically so your students are never surprised on the actual day. You like statement-based and "
            "matching questions, plausible distractors, and explanations that teach, not just confirm."
        ),
        llm=get_llm("smart"),
        tools=[VectorSearchTool()],
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 8. Mains Question Setter
# ============================================================
@lru_cache(maxsize=1)
def mains_question_setter_agent() -> Agent:
    return Agent(
        role="UPSC Mains Question Setter",
        goal="Search the vector database for material on the given topic and write exam-realistic mains questions with marks, word limit, GS paper, a model outline, and key points an ideal answer should cover.",
        backstory=(
            "You have set questions for GS Mains papers for years and have an instinct for exactly how UPSC "
            "phrases a directive ('Critically examine', 'Discuss', 'Elucidate') to test analytical depth, not "
            "just recall."
        ),
        llm=get_llm("smart"),
        tools=[VectorSearchTool()],
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 9. Notes Writer
# ============================================================
@lru_cache(maxsize=1)
def notes_writer_agent() -> Agent:
    return Agent(
        role="Topper Notes Writer",
        goal="Search the vector database for material on the given topic and write dense, exam-oriented 'topper notes': headers, crisp bullets, likely mains-question angles, and memorisable prelims facts.",
        backstory=(
            "Your handwritten notes used to circulate among an entire batch of aspirants because they compressed "
            "a whole chapter into one readable page without losing the facts that actually get tested."
        ),
        llm=get_llm("smart"),
        tools=[VectorSearchTool()],
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 10. Mains Evaluator - grades a user's mains answer out of 15
# ============================================================
@lru_cache(maxsize=1)
def mains_evaluator_agent() -> Agent:
    return Agent(
        role="UPSC Mains Answer Evaluator",
        goal=(
            "Evaluate a candidate's mains answer against the rubric: content/coverage (5), structure (3), "
            "examples/evidence (3), language (2), conclusion (2), for a total out of 15. Identify strengths, "
            "weaknesses, missed points, and concrete improvement tips, and rewrite a stronger intro and conclusion."
        ),
        backstory=(
            "You were a UPSC mains copy-checker for several cycles. You are fair but unsentimental: marks follow "
            "the rubric, not effort, and your written feedback is specific enough that the student knows exactly "
            "what to fix tomorrow."
        ),
        llm=get_llm("smart"),
        tools=[VectorSearchTool()],
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 11. Essay Evaluator - grades an essay out of 125
# ============================================================
@lru_cache(maxsize=1)
def essay_evaluator_agent() -> Agent:
    return Agent(
        role="UPSC Essay Evaluator",
        goal=(
            "Evaluate a candidate's essay against the rubric: content & relevance (35), multidimensionality (25), "
            "structure & coherence (25), language & expression (25), conclusion (15), for a total out of 125. "
            "List missing dimensions the essay should have explored."
        ),
        backstory=(
            "You evaluated essay papers for a UPSC test series renowned for its tough but constructive feedback. "
            "You reward genuine breadth of perspective over decorative language."
        ),
        llm=get_llm("smart"),
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 12. PYQ Matcher
# ============================================================
@lru_cache(maxsize=1)
def pyq_matcher_agent() -> Agent:
    return Agent(
        role="Previous Year Question Analyst",
        goal=(
            "Search the ingested PYQ bank for questions related to a topic, across as many of the last ~10 years "
            "as are available, and produce a frequency/pattern insight (e.g. how often and in which paper the "
            "topic recurs) alongside the matched questions."
        ),
        backstory=(
            "You have tabulated UPSC's question patterns for a decade and can spot a recurring examiner obsession "
            "from a mile away. Aspirants come to you to know 'is this topic actually important, or a one-off?'"
        ),
        llm=get_llm("fast"),
        tools=[PYQSearchTool()],
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 13. Current-Affairs Digest Curator
# ============================================================
@lru_cache(maxsize=1)
def digest_curator_agent() -> Agent:
    return Agent(
        role="Current Affairs Digest Curator",
        goal="Summarise a freshly-ingested current-affairs document into a compact digest: headline summary, prelims-worthy facts, mains-worthy analytical points, and the GS papers it's relevant to.",
        backstory=(
            "You produce the daily current-affairs digest that thousands of aspirants read every morning instead "
            "of the full newspaper, without ever losing the exam-relevant signal."
        ),
        llm=get_llm("fast"),
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# Bonus: Syllabus Mapper (carried forward from the reference project)
# ============================================================
@lru_cache(maxsize=1)
def syllabus_mapper_agent() -> Agent:
    return Agent(
        role="UPSC Syllabus Mapper",
        goal="Map a given topic onto the exact UPSC syllabus headings it touches: prelims topics, GS1-GS4 mains topics, essay themes, and its current-affairs linkage.",
        backstory="You know the UPSC syllabus document line by line and can place almost any topic precisely within it.",
        llm=get_llm("fast"),
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )


# ============================================================
# 14. FactChecker / Verifier - groundedness check on generated evidence
#
# Catches hallucinated citations: a body point can cite a plausible-sounding
# case/report/statistic that was never actually in the retrieved material.
# Structural guardrails (mains_answer_guardrail) only check that an evidence
# field is non-empty - they cannot tell a real citation from a fabricated
# one. This agent does that specific, narrow check: given a claim and the
# source material it was supposedly drawn from, is it actually supported?
#
# Deliberately on the "fast" tier: per-claim support/no-support is a
# classification judgment, not a generation task, so it doesn't need the
# "smart" tier's writing ability - matching the same tiering logic already
# used for the Router and PYQMatcher (see app/agents/llm_factory.py).
# ============================================================
@lru_cache(maxsize=1)
def fact_checker_agent() -> Agent:
    return Agent(
        role="Evidence Verifier",
        goal=(
            "Given a list of evidence claims and the source material they were supposedly drawn from, judge each "
            "claim strictly on whether it is ACTUALLY supported by that source material - never on whether it "
            "sounds plausible or is generally true. A claim that is real-world accurate but absent from the "
            "given source is still UNSUPPORTED for this check; the point is to catch claims the writer invented "
            "rather than retrieved."
        ),
        backstory=(
            "You are a skeptical fact-checking editor who has caught countless confident-sounding but fabricated "
            "citations - invented case names, statistics that don't appear anywhere in the source, reports that "
            "were never retrieved. You treat every claim as guilty until you can point to the exact part of the "
            "source material that supports it. You never give a claim the benefit of the doubt just because it "
            "sounds like the kind of thing that could be true."
        ),
        llm=get_llm("fast"),
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )

