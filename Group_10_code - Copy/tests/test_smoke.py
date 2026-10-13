"""
upsc-sarathi-agentic :: tests/test_smoke.py

Smoke tests - verify the package imports cleanly and core logic behaves
correctly, without needing a live LLM (Ollama/Bedrock) or vector store.

Run with: pytest tests/ -v

These tests were exercised during development against a real installed
crewai==1.14.7 to confirm every Agent/Task/Crew construction, guardrail
callable, and the full FastAPI route table actually work - not just that
the files parse. The guardrail tests below specifically reproduce that
verification: a TaskOutput is mocked with a hand-built MainsAnswer to
confirm mains_answer_guardrail() correctly accepts well-formed, in-range
answers and rejects malformed ones with an actionable error message.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Safe defaults so config can load without a .env file present.
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("APP_MODE", "local")
os.environ.setdefault("VECTOR_STORE", "chroma")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDINGS_PROVIDER", "ollama")


def test_config_loads():
    from app.config import get_settings
    s = get_settings()
    assert s.aws_region == "us-east-1"
    assert s.vector_store in ("chroma", "opensearch")
    assert s.llm_provider in ("bedrock", "openai", "ollama", "groq")


def test_agent_schemas_validate():
    from app.models.agent_schemas import RouteDecision, MainsAnswer, BodyPoint, EssayAnswer, EssayParagraph, ClaimVerification, VerificationResult
    RouteDecision(use_rag=True, reasoning="needs facts", search_query="Article 370")
    BodyPoint(point="x", evidence="Article 21", evidence_type="Constitutional Provision")
    EssayParagraph(dimension="Economic", content="...")
    ClaimVerification(claim="x", supported=True, matching_source="y")
    VerificationResult(verified_claims=[], all_supported=True, unsupported_claims=[])


def test_api_schemas_validate():
    from app.models.schemas import QARequest, MCQRequest, MainsWriteRequest, EssayWriteRequest
    QARequest(question="What is Article 370?")
    MCQRequest(topic="Indian Polity", n=5)
    MainsWriteRequest(question="Discuss judicial review", marks=15)
    EssayWriteRequest(topic="Technology and humanity")


def test_word_count_helpers():
    from app.utils.word_count import count_words, marks_to_word_range, essay_word_range
    assert count_words("one two three") == 3
    assert marks_to_word_range(15) == (240, 300)
    assert marks_to_word_range(10) == (130, 170)
    lo, hi = essay_word_range()
    assert lo < 1500 < hi


def test_llm_factory_builds_llm():
    """Doesn't make a network call - crewai.LLM construction is local."""
    from app.agents.llm_factory import get_llm
    llm = get_llm("fast")
    assert llm is not None


def test_all_agents_construct():
    from app.agents import agents as A
    factories = [
        A.router_agent, A.direct_answerer_agent, A.context_summarizer_agent, A.answer_writer_agent,
        A.upsc_formatter_agent, A.essay_architect_agent, A.mcq_setter_agent, A.mains_question_setter_agent,
        A.notes_writer_agent, A.mains_evaluator_agent, A.essay_evaluator_agent, A.pyq_matcher_agent,
        A.digest_curator_agent, A.syllabus_mapper_agent, A.fact_checker_agent,
    ]
    for fn in factories:
        agent = fn()
        assert agent.role
        assert agent.allow_delegation is False


def test_all_crew_modules_import():
    import importlib
    for mod in [
        "app.agents.crews.qa_crew",
        "app.agents.crews.mains_answer_crew",
        "app.agents.crews.essay_crew",
        "app.agents.crews.question_gen_crew",
        "app.agents.crews.notes_crew",
        "app.agents.crews.evaluator_crew",
        "app.agents.crews.pyq_matcher_crew",
        "app.agents.crews.digest_crew",
        "app.agents.crews.syllabus_crew",
    ]:
        importlib.import_module(mod)


def _mock_task_output(pydantic_obj):
    from crewai.tasks.task_output import TaskOutput
    return TaskOutput(description="d", expected_output="e", raw="{}", agent="TestAgent", pydantic=pydantic_obj)


def test_mains_answer_guardrail_rejects_too_few_points():
    from app.agents.guardrails import mains_answer_guardrail
    from app.models.agent_schemas import MainsAnswer, BodyPoint

    bad = MainsAnswer(
        introduction="Short intro here about the topic at hand for context.",
        body_points=[BodyPoint(point="one point", evidence="Article 21 of the Constitution", evidence_type="Constitutional Provision")],
        conclusion="Short conclusion wrapping up the discussion nicely indeed.",
        word_count=0,
        marks=15,
    )
    ok, msg = mains_answer_guardrail(15)(_mock_task_output(bad))
    assert ok is False
    assert "point" in msg.lower()


def test_mains_answer_guardrail_accepts_well_formed_answer():
    from app.agents.guardrails import mains_answer_guardrail
    from app.models.agent_schemas import MainsAnswer, BodyPoint

    intro = ("India's federal architecture balances national unity with deep regional diversity, "
             "a design tested repeatedly since the Constitution came into force in 1950.")
    points = [
        BodyPoint(
            point="Constitutional design enshrines a quasi-federal Union of States with a strong "
                  "centralising bias built in deliberately for emergencies and national integrity concerns",
            evidence="Article 1 and the Seventh Schedule divide legislative subjects into Union, State and Concurrent lists",
            evidence_type="Constitutional Provision",
        ),
        BodyPoint(
            point="The judiciary has repeatedly reinforced federalism as a basic feature of the Constitution "
                  "that cannot be amended away by a simple parliamentary majority alone",
            evidence="S.R. Bommai vs Union of India (1994) held that federalism is part of the basic structure",
            evidence_type="Supreme Court / High Court Judgment",
        ),
        BodyPoint(
            point="Fiscal federalism has been strengthened through formal periodic devolution mechanisms "
                  "recommended by constitutional bodies set up for this exact redistributive purpose",
            evidence="The Fifteenth Finance Commission recommended 41 percent tax devolution to states for 2021-26",
            evidence_type="Committee or Commission Report",
        ),
        BodyPoint(
            point="Cooperative federalism institutions now provide a continuous standing bargaining table "
                  "between the centre and the states on major economic policy questions",
            evidence="The GST Council institutionalised joint centre-state decision making on indirect taxation from 2017",
            evidence_type="Government Scheme or Policy",
        ),
        BodyPoint(
            point="Yet centralising tendencies persist and periodically strain the federal compact in practice "
                  "despite the formal cooperative constitutional design described above",
            evidence="Comptroller and Auditor General reports have flagged delays in GST compensation transfers to states",
            evidence_type="Parliamentary Report or Act",
        ),
    ]
    concl = ("A resilient federal balance, anchored by consistent judicial oversight and institutionalised "
             "fiscal cooperation mechanisms, remains essential to sustaining India's pluralistic democratic "
             "project going forward into the future.")
    good = MainsAnswer(introduction=intro, body_points=points, conclusion=concl, word_count=0, marks=15)

    ok, result = mains_answer_guardrail(15)(_mock_task_output(good))
    assert ok is True
    assert 240 <= result.word_count <= 300


def test_combine_guardrails_aggregates_failures():
    from app.agents.guardrails import combine_guardrails

    def g1(output):
        return False, "error one"

    def g2(output):
        return True, "result two"

    def g3(output):
        return False, "error three"

    ok, msg = combine_guardrails(g1, g2, g3)(None)
    assert ok is False
    assert "error one" in msg and "error three" in msg


def test_combine_guardrails_passes_when_all_pass():
    from app.agents.guardrails import combine_guardrails

    def g1(output):
        return True, "r1"

    def g2(output):
        return True, "r2"

    ok, result = combine_guardrails(g1, g2)(None)
    assert ok is True
    assert result == "r2"


def _mock_structured_llm_response(parsed_obj):
    """Patches the exact internal seam crewai's OpenAICompletion uses for
    structured (`output_pydantic=...`) calls - client.beta.chat.completions.parse
    - so that ANY task built around an Ollama/OpenAI-style LLM (including a
    nested Crew().kickoff() invoked from inside a guardrail callable)
    returns `parsed_obj` without making a real network call. This is the
    same mocking seam used to verify, before wiring groundedness_guardrail
    into mains_answer_crew.py, that a nested Crew().kickoff() called from
    inside another task's guardrail does not conflict with the outer task's
    own guardrail-retry loop.
    """
    from unittest.mock import patch, MagicMock

    msg = MagicMock()
    msg.parsed = parsed_obj
    msg.refusal = None
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.id = "fake-id"
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    resp.model = "fake-model"

    client = MagicMock()
    client.beta.chat.completions.parse.return_value = resp
    return patch(
        "crewai.llms.providers.openai.completion.OpenAICompletion._get_sync_client",
        return_value=client,
    )


class _FakeContextTask:
    """Minimal stand-in for the live CrewAI Task object groundedness_guardrail
    expects - exposes only the `.output.pydantic.summary` attribute path it
    actually reads, so these tests don't need to run the full Summarizer task."""
    def __init__(self, summary: str):
        from unittest.mock import MagicMock
        self.output = MagicMock()
        self.output.pydantic = MagicMock(summary=summary)


def test_groundedness_guardrail_rejects_unsupported_claim():
    from app.agents.guardrails import groundedness_guardrail
    from app.models.agent_schemas import MainsAnswer, BodyPoint, VerificationResult, ClaimVerification

    answer = MainsAnswer(
        introduction="A reasonably long introduction sentence about federalism in India for context purposes.",
        body_points=[
            BodyPoint(
                point="A fabricated-looking judicial precedent is cited here as if it were real and grounded",
                evidence="Fabricated case ABC vs XYZ (2099) held that this is true",
                evidence_type="Supreme Court / High Court Judgment",
            )
        ],
        conclusion="A reasonably long concluding sentence wrapping up the discussion for context purposes.",
        word_count=100,
        marks=15,
    )
    fake_verification = VerificationResult(
        verified_claims=[
            ClaimVerification(
                claim="Fabricated case ABC vs XYZ (2099) held that this is true",
                supported=False,
                matching_source="Not found anywhere in the provided source material",
            )
        ],
        all_supported=False,
        unsupported_claims=["Fabricated case ABC vs XYZ (2099) held that this is true"],
    )
    context_task = _FakeContextTask(summary="Some real retrieved material about Indian federalism and the Constitution.")

    with _mock_structured_llm_response(fake_verification):
        ok, msg = groundedness_guardrail(context_task)(_mock_task_output(answer))

    assert ok is False
    assert "Fabricated case ABC vs XYZ" in msg


def test_groundedness_guardrail_accepts_supported_claims():
    from app.agents.guardrails import groundedness_guardrail
    from app.models.agent_schemas import MainsAnswer, BodyPoint, VerificationResult, ClaimVerification

    answer = MainsAnswer(
        introduction="A reasonably long introduction sentence about fundamental rights for context purposes.",
        body_points=[
            BodyPoint(
                point="Article 21 forms the constitutional bedrock of the right to life and personal liberty",
                evidence="Article 21 guarantees the right to life and personal liberty",
                evidence_type="Constitutional Provision",
            )
        ],
        conclusion="A reasonably long concluding sentence wrapping up the discussion for context purposes.",
        word_count=100,
        marks=15,
    )
    fake_verification = VerificationResult(
        verified_claims=[
            ClaimVerification(
                claim="Article 21 guarantees the right to life and personal liberty",
                supported=True,
                matching_source="Article 21 of the Constitution guarantees the right to life and personal liberty.",
            )
        ],
        all_supported=True,
        unsupported_claims=[],
    )
    context_task = _FakeContextTask(
        summary="Article 21 of the Constitution guarantees the right to life and personal liberty."
    )

    with _mock_structured_llm_response(fake_verification):
        ok, result = groundedness_guardrail(context_task)(_mock_task_output(answer))

    assert ok is True


def test_revision_sm2():
    from app.services import revision
    c = revision.add_card("Polity", "What is Article 21?", "Right to life and personal liberty")
    c2 = revision.review(c.card_id, quality=5)
    assert c2.repetitions == 1
    assert c2.interval_days >= 1
    c3 = revision.review(c.card_id, quality=1)
    assert c3.repetitions == 0


def test_fastapi_app_factory_and_routes():
    from app.main import create_app
    app = create_app()
    assert app.title == "UPSC Sarathi Agentic"
    spec = app.openapi()
    paths = set(spec["paths"].keys())
    expected = {
        "/api/v1/health", "/api/v1/qa", "/api/v1/mains/write", "/api/v1/essay/write",
        "/api/v1/generate/mcq", "/api/v1/generate/mains", "/api/v1/generate/notes",
        "/api/v1/test/prelims", "/api/v1/test/mains", "/api/v1/evaluate", "/api/v1/evaluate/essay",
        "/api/v1/pyq/match", "/api/v1/syllabus/map", "/api/v1/ingest/upload", "/api/v1/ingest/pyq",
        "/api/v1/ingest/current-affairs",
    }
    assert expected.issubset(paths), f"Missing routes: {expected - paths}"


def test_sr_endpoints_accept_json_body_not_query_params():
    """Regression test for a real, previously-shipped bug: /sr/add and
    /sr/review used bare scalar function parameters, which FastAPI reads as
    query parameters - but every caller (the Streamlit frontend included)
    sends a JSON body like every other endpoint in this API. That mismatch
    made both endpoints fail with a 422 on every real call. Fixed by giving
    them proper Pydantic request models (schemas.SRAddRequest/SRReviewRequest)
    consistent with the rest of the API. This test sends exactly the JSON
    shape the frontend sends and asserts it now succeeds end-to-end,
    including a real round-trip through the SM-2 scheduler."""
    from fastapi.testclient import TestClient
    from app.main import create_app

    client = TestClient(create_app())

    add_resp = client.post("/api/v1/sr/add", json={"topic": "Polity", "front": "What is Article 21?", "back": "Right to life and personal liberty"})
    assert add_resp.status_code == 200, add_resp.text
    card = add_resp.json()
    assert card["topic"] == "Polity"
    card_id = card["card_id"]

    review_resp = client.post("/api/v1/sr/review", json={"card_id": card_id, "quality": 5})
    assert review_resp.status_code == 200, review_resp.text
    reviewed = review_resp.json()
    assert reviewed["repetitions"] == 1

    # Unknown card -> a clean 404, not a raw unhandled 500
    missing_resp = client.post("/api/v1/sr/review", json={"card_id": "does-not-exist", "quality": 5})
    assert missing_resp.status_code == 404

    # /sr/due should also work cleanly (it never had the query/body bug, but
    # confirm it round-trips through the same TestClient without error).
    due_resp = client.get("/api/v1/sr/due", params={"limit": 5})
    assert due_resp.status_code == 200
    assert isinstance(due_resp.json(), list)
