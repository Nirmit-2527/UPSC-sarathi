# UPSC Sarathi Agentic — A Complete, Beginner-Friendly Walkthrough

*A component-by-component explanation of the project, with the underlying Generative AI and CrewAI concepts explained alongside each piece.*

---

## How to read this document

This guide is organized so you can read it top to bottom as a course, or jump to any section as a reference:

1. **Part 1 — What is this project?** (the product view, no jargon)
2. **Part 2 — Gen AI fundamentals** you need before anything else makes sense
3. **Part 3 — CrewAI fundamentals** (the framework this project is built on)
4. **Part 4 — The project's folder structure**, at a glance
5. **Part 5 — Deep dive**, component by component, with real code from the project
6. **Part 6 — Two full end-to-end walkthroughs** (a simple question, and the flagship "write my mains answer" feature)
7. **Part 7 — The big design decisions**, explained in plain language
8. **Part 8 — What's honestly still rough** (the project's own audit notes)
9. **Part 9 — Glossary**

---

# Part 1 — What is this project, in plain English?

**UPSC Sarathi Agentic** is a study assistant for people preparing for India's UPSC Civil Services Examination. Think of it as a small team of specialist AI tutors, each good at exactly one job, working together behind a single chat-like web app.

A student can:
- Upload their study PDFs, previous-year question papers, and daily newspaper/current-affairs PDFs.
- Ask any question and get an answer that's *grounded* in those uploaded documents (not just the AI's general knowledge).
- Ask the app to **write a model mains answer** in the exact format UPSC examiners expect (short intro, bullet-point body where every point has real evidence, short conclusion, exact word count).
- Ask for a full **~1500 word essay**, **MCQs**, **mains-style questions**, **topper notes**, a **PYQ (Previous Year Question) pattern report**, or an **evaluation** of an answer they wrote themselves (marked out of 15, like a real examiner would).
- Use a **spaced-repetition flashcard system** to revise facts just before they'd otherwise forget them.

The interesting engineering part — the reason this is called "agentic" — is *how* the answers get produced. Instead of one big AI model trying to do everything with one giant instruction, the system is broken into **14 narrow specialist AI agents**, each with one job, that hand work to each other in a fixed pipeline. A special mechanism called a **guardrail** mechanically checks the output (e.g. "is this really 250–300 words? does every point actually have evidence?") and automatically asks the agent to try again if it doesn't pass — rather than just hoping the AI followed instructions.

---

# Part 2 — Gen AI fundamentals (read this first if you're new)

You don't need a machine learning background to follow this project, but a few core ideas will make everything below click into place.

### 2.1 What is an LLM?

A **Large Language Model (LLM)** — like GPT-4, Claude, or Llama — is a program trained on huge amounts of text that has learned to predict "what text comes next." You give it text (a **prompt**), and it generates more text that plausibly follows. That's it, fundamentally. Everything else — chatbots, coding assistants, agents — is built by cleverly *structuring* what you put into the prompt and what you do with what comes out.

Two properties matter a lot for this project:

- **LLMs don't "know" your private documents.** An LLM's knowledge comes only from its training data (which has a cutoff date) and whatever text you put in the prompt. It has never seen the student's uploaded PDFs.
- **LLMs can "hallucinate."** They generate *plausible-sounding* text, not necessarily *true* text. If asked "cite a Supreme Court judgment for this point," an LLM may confidently invent a case name that sounds real but doesn't exist. This single fact is the reason a large part of this project's engineering exists (see the FactChecker agent in Part 5.6).

### 2.2 Prompting

A prompt is just the text you send the model. Well-engineered prompts typically give the model:
- A **role** ("You are a strict UPSC answer-format examiner…")
- A **task** ("Reformat this into introduction / body-points / conclusion…")
- **Constraints** ("Body must be 250–300 words, every point must cite real evidence…")
- Sometimes **examples** of what a good answer looks like

You'll see this exact pattern (role + goal + backstory + task instructions) reused for every one of the 14 agents in this project.

### 2.3 Embeddings — turning text into numbers you can compare

An **embedding model** converts a piece of text into a list of numbers (a **vector**), for example `[0.021, -0.114, 0.87, ...]`, typically with hundreds or thousands of dimensions. The key property: **texts with similar meaning produce vectors that are close together** in this numeric space, even if they don't share exact words. "Fundamental Rights" and "Constitutional protections for citizens" would land close together; "Fundamental Rights" and "Monsoon rainfall patterns" would land far apart.

This is what lets a computer do **semantic search** — search by *meaning*, not just by keyword matching.

"Closeness" between two vectors is usually measured with **cosine similarity**: imagine each vector as an arrow from the origin; cosine similarity looks at the angle between two arrows. Same direction (small angle) = very similar meaning. This project's Chroma vector store is explicitly configured with `"hnsw:space": "cosine"` (see Part 5.2) to use exactly this measure.

### 2.4 Vector databases

A **vector database** stores many pieces of text alongside their embeddings, and can quickly answer: *"given this new query vector, which stored vectors are closest to it?"* This project uses two interchangeable options — **Chroma** (runs locally, no cloud account needed) and **OpenSearch Serverless** (AWS's managed option) — covered in Part 5.2.

### 2.5 RAG — Retrieval-Augmented Generation

This is the single most important concept in the whole project. **RAG** means: before asking the LLM to answer, first **retrieve** relevant chunks of real text from a knowledge base (via embeddings + vector search), and **stuff those chunks into the prompt** as context. The LLM then answers *using that provided material* instead of purely from its own memory.

Why this matters for UPSC prep specifically: exam answers must cite specific, real facts — a constitutional article number, an actual Supreme Court case, an actual scheme name. RAG is what lets the system ground its answers in the aspirant's *actual* uploaded textbooks and PYQs, rather than the LLM improvising facts from its training data (which might be outdated, generic, or simply wrong for this narrow domain).

The RAG loop in this project, in one line:
**question → embed the question → find similar chunks in the vector DB → hand those chunks to an LLM agent → agent writes an answer using only that material.**

### 2.6 Chunking

You can't hand an LLM (or a vector database) an entire 400-page textbook as one unit — it's too large, and a huge block of text embeds into one vague, unhelpful vector. So documents are split into **chunks** — smaller, semantically coherent windows of a few hundred to ~1000 characters, each embedded and stored separately. Later, the vector search retrieves the *specific* chunks relevant to the question, not the whole book. Part 5.3 covers exactly how this project chunks textbooks differently from previous-year-question compilations.

### 2.7 Structured output

Instead of letting an LLM produce free-flowing prose, you can force it to answer in a specific machine-readable shape — typically **JSON** matching a schema you define (in Python, using **Pydantic** models). For example: `{"introduction": "...", "body_points": [{"point": "...", "evidence": "...", "evidence_type": "..."}], "conclusion": "..."}`.

Why bother? Because structured output is what makes **automated checking** possible. You cannot reliably write code that checks "did every point in this answer include real evidence?" against a blob of markdown text — but you *can* trivially check `len(point.evidence.strip()) > 10` for every item in a Python list. This project uses structured output *everywhere* (see `app/models/agent_schemas.py`, Part 5.7), and it's the foundation the whole guardrail system (Part 3.6) is built on.

### 2.8 Why break work into multiple agents at all?

You could, in principle, write one giant prompt: *"You are a UPSC tutor. If the question needs facts, search the database. Write a great answer. Format it strictly. Check your own evidence. Suggest related PYQs."* This is tempting but tends to work poorly in practice, for reasons this project explicitly calls out:

- A single mega-prompt is harder for the model to follow reliably — instructions compete with each other and get lost.
- You can't easily test "does the routing logic work" separately from "does the formatting logic work" if it's all tangled into one call.
- You pay for the LLM's full capability (cost + latency) on every sub-task, even trivial classification ones like "does this need a database lookup at all?"

Instead, this project uses **14 narrow specialist agents**, each with a single, clearly-scoped job (a Router, a Summarizer, a Writer, a Formatter, an Evaluator, a Fact-Checker, etc.), wired together into short pipelines. This is the "multi-agent" or "agentic" part of the name, and it's exactly what the CrewAI framework (Part 3) is designed to make easy.

---

# Part 3 — CrewAI fundamentals

**CrewAI** is a Python framework purpose-built for exactly this pattern: define multiple specialist AI agents, give them tasks, and run them as a coordinated team ("crew"). This project pins `crewai==1.14.7`.

### 3.1 Agent

An `Agent` is a **configuration object**, not a running process. It bundles:
- `role` — a short job title, e.g. `"UPSC Mains Answer Formatter"`
- `goal` — what it's trying to achieve
- `backstory` — a persona/context that shapes tone and judgment (this is prompt engineering — a well-written backstory measurably improves how the model behaves)
- `llm` — which underlying language model powers it
- `tools` — optional capabilities it can invoke (like a search function)
- `allow_delegation` — whether it's allowed to hand off work to other agents

Here's an actual agent from the project (`app/agents/agents.py`):

```python
def upsc_formatter_agent() -> Agent:
    return Agent(
        role="UPSC Mains Answer Formatter",
        goal=(
            "Reformat a drafted answer into the strict UPSC mains structure: (1) a catchy, relevant 2-3 line "
            "introduction, (2) a body written ENTIRELY as discrete points - never flowing paragraphs - where "
            "every single point carries explicit evidence ..."
        ),
        backstory=(
            "You are the strictest answer-format examiner in the business. You have rejected thousands of "
            "well-researched but poorly structured answers because they 'wrote in paragraphs' or 'asserted "
            "without evidence'. ..."
        ),
        llm=get_llm("smart"),
        allow_delegation=False,
        verbose=settings.agents_verbose,
        max_rpm=settings.agent_max_rpm,
    )
```

Notice `allow_delegation=False` on **every single agent** in this project — a deliberate choice explained in Part 7.

### 3.2 Task

A `Task` is one unit of work assigned to an agent: a `description` (the actual instructions/prompt content), an `expected_output` (a hint to the model about the shape of a good answer), optionally `output_pydantic` (forces structured JSON output matching a schema — see Part 2.7), optionally `context=[...]` (feeds in the output of earlier tasks — this is how agents "hand off" work to each other), and optionally `guardrail=...` (see 3.6).

### 3.3 Crew

A `Crew` groups a list of agents and a list of tasks and actually **runs** them (`crew.kickoff()`). This is the "team" that executes the pipeline.

### 3.4 Process — sequential vs. hierarchical

CrewAI supports different execution strategies:
- **`Process.sequential`** (used everywhere in this project): tasks run one after another, in the order given. Simple, predictable, debuggable.
- **`Process.hierarchical`**: a "manager" agent (itself an LLM call) decides, at runtime, what should happen next — more flexible for open-ended agent swarms, but adds an extra unpredictable LLM call just to decide "what next," and is much harder to unit test. This project explicitly chose **not** to use this (see Part 7.2) — instead, "routing" decisions are made by an ordinary `if` statement in Python that reads a structured decision from a tiny one-task crew.

### 3.5 Tools

A `Tool` gives an agent the ability to *do something* beyond generating text — most commonly here, to search the vector database. Tools in CrewAI subclass `BaseTool`, declare their input schema (also a Pydantic model), and implement a `_run(...)` method. The LLM decides *when* to call the tool and with what arguments, based on its description — this is the mechanism (often called "function calling" or "tool calling" in the broader industry) that lets an LLM take actions, not just talk. See Part 5.6 for this project's two tools.

### 3.6 Guardrails — the mechanism that makes "the AI actually followed instructions" checkable

This is the single most distinctive engineering idea in this project, so it's worth understanding precisely.

A `Task` can be given `guardrail=some_function` and `guardrail_max_retries=N`. After the agent produces an output, CrewAI calls your function with that output. Your function returns a tuple: `(success: bool, result)`.

- If `success` is `True`: the task is considered done; `result` becomes the accepted output.
- If `success` is `False`: **CrewAI automatically re-runs the task**, feeding the agent your failure message as feedback (e.g. "your word count was 412, must be 240–300 — trim the body points"), up to `N` times.

This turns "please follow the format" from a hopeful prompt instruction into a **mechanically enforced, self-correcting loop** — the agent doesn't get to submit a wrong-shaped answer and have nobody notice. This project uses guardrails for:
1. Structural correctness (word count, point count, non-empty evidence) — `mains_answer_guardrail`, `essay_guardrail`
2. **Groundedness** — whether cited evidence was actually retrieved, not invented — `groundedness_guardrail` (a second, independent AI agent doing the checking!)

Full code walkthrough is in Part 5.9.

### 3.7 Structured output + guardrails + tools, together

The pattern that recurs throughout this project is:

```
Agent (role, goal, tools=[VectorSearchTool()], llm=get_llm("smart"))
   → Task (description, output_pydantic=SomeSchema, guardrail=some_check_fn)
      → Crew([agent], [task]).kickoff()
         → task.output.pydantic   # the validated, structured result
```

Once you recognize this shape, every crew file in `app/agents/crews/` becomes readable at a glance.

---

# Part 4 — Project folder structure, at a glance

```
upsc-sarathi-agentic/
├── app/
│   ├── agents/          ← the CrewAI layer: agents, tools, guardrails, LLM factory, and crews/
│   │   └── crews/       ← one pipeline-definition module per feature
│   ├── api/routes/      ← FastAPI HTTP endpoints (one per feature)
│   ├── core/            ← embeddings, vector store abstraction, retrieval logic (no LLM agents here)
│   ├── ingestion/       ← PDF loading, chunking, the ingestion CLI
│   ├── models/          ← Pydantic schemas: agent_schemas.py (internal) + schemas.py (API)
│   ├── services/        ← thin glue layer between API routes and crews, + spaced-repetition scheduler
│   ├── utils/           ← small helpers (word counting) used by guardrails
│   ├── config.py        ← all settings, loaded from .env
│   └── main.py          ← FastAPI application entrypoint
├── frontend/
│   └── streamlit_app.py ← the multi-page web UI
├── infrastructure/
│   └── main.tf           ← Terraform: defines the AWS cloud deployment
├── scripts/               ← setup/deploy/teardown shell scripts
├── tests/
│   └── test_smoke.py     ← imports everything + tests the guardrails actually work
├── data/                  ← sample folders for PDFs, PYQs, current affairs
├── Dockerfile, docker-compose*.yml
└── README.md, ARCHITECTURE.md, SETUP_LOCAL.md, DEPLOY_AWS.md, IMPLEMENTATION_GUIDE.md
```

A useful mental model: **`ingestion/` and `core/` are the "librarian"** (get documents in, make them searchable). **`agents/` is the "faculty"** (the actual AI reasoning). **`services/` and `api/` are the "front desk"** (route a request to the right specialist and hand back the result). **`frontend/` is the "classroom"** (what the student actually sees).

---

# Part 5 — Deep dive, component by component

## 5.1 `app/config.py` — one settings object drives two deployment profiles

The whole project can run in two modes without changing any code — only environment variables (`.env` file):

| | **local profile** | **AWS profile** |
|---|---|---|
| LLM | Ollama (open-source models running on your own machine — `llama3.1:8b`, `llama3.2:3b`) | Amazon Bedrock (`Claude 3.5 Sonnet` / `Haiku`) |
| Embeddings | Ollama (`nomic-embed-text`) | Bedrock (`Amazon Titan Text Embeddings v2`) |
| Vector DB | Chroma (a local file-based vector database) | OpenSearch Serverless |
| Progress tracking | in-memory Python dict | DynamoDB |

This is implemented with Pydantic's `BaseSettings` class, which auto-reads a `.env` file and validates types (e.g. `vector_store: Literal["chroma", "opensearch"]` — Python will refuse to start if you typo the value). Every other file in the project imports the single shared `settings` object rather than reading environment variables directly — this is what lets Part 7's "one `.env` change flips the whole deployment" design decision actually be true.

## 5.2 `app/core/embeddings.py` and `app/core/vectorstore.py` — the "librarian's" memory

**`Embeddings`** is a small wrapper class with one real job: `embed(list_of_texts) -> list_of_vectors`. It supports four providers (`bedrock`, `ollama`, `sentence_transformers`, `openai`) behind the same interface, so nothing downstream needs to know which one is active. One nice detail: for Ollama, it doesn't hardcode the embedding vector's dimensionality — it makes one throwaway "dimension probe" call and measures the returned vector's length, because different embedding models produce different-sized vectors and getting this wrong would silently corrupt the vector database.

**`BaseVectorStore`** defines the common interface (`add`, `similarity_search`, `count`, `delete_by_source`), implemented by:
- **`ChromaStore`** — a local, persistent, file-based vector database. Notice the collection name: `f"upsc_sarathi_{settings.embeddings_provider}_{self.embeddings.dim}"`. This is a subtle but important safety detail — if you ever switch embedding providers mid-project (e.g. from Ollama to Bedrock), the two providers produce *numerically incompatible* vectors (different scales, different dimensions, different meaning-geometry). By baking the provider name and vector dimension into the collection name, switching providers automatically starts a *fresh* collection instead of silently mixing incompatible vectors — which would otherwise quietly return garbage search results with no error.
- **`OpenSearchStore`** — talks to AWS's managed OpenSearch Serverless with its vector ("kNN") search feature, authenticating via either an explicit username/password or AWS's native request-signing (`AWSV4SignerAuth`) so it can use the IAM role attached to the deployment instead of hardcoded credentials.

Every stored chunk carries metadata: `source` (filename), `page`, `chunk_id`, `subject` (a heuristically-detected topic tag), `doc_type` (`study_material` / `pyq` / `current_affairs` / `digest`), and for PYQs, `exam_year` / `exam_type` / `gs_paper`. This metadata is what lets a search be *filtered*, not just ranked — e.g., "only search chunks where `doc_type == pyq`" for the PYQ matcher, without needing a separate database.

## 5.3 `app/ingestion/` — getting PDFs into the vector database

**`pdf_loader.py`** extracts raw text (and page numbers) from PDF files, configurable between `pypdf`, `pdfplumber`, or `unstructured` (the latter can also handle scanned/OCR'd PDFs).

**`chunker.py`** implements the generic chunking strategy using LangChain's `RecursiveCharacterTextSplitter`. "Recursive" means: it first *tries* to split on the most meaningful boundary (a custom list tuned for UPSC material — `"\n\nCHAPTER "`, `"\n\nArticle "`, `"\n\nSchedule "`), and only falls back to cruder boundaries (paragraphs, then sentences, then raw characters) if a chunk is still too big. This keeps chunks from awkwardly cutting a legal Article or book chapter in half.

**`pyq_parser.py`** solves a specific real problem: Previous-Year-Question compilation PDFs are lists of many short, independently-meaningful questions (not flowing prose), so the generic chunker would badly cut through them (merging three unrelated questions into one chunk, or splitting one question in half). This module instead:
1. Parses the exam year, `Prelims`/`Mains`, and GS-paper number straight out of the **filename**, using a simple naming convention (e.g. `Mains_GS2_2021.pdf`) — via regex.
2. Splits each page's text at every detected "new numbered question" boundary (`23. `, `Q.23 `, `23) `) using a regex, so each PYQ becomes its own precise, self-contained chunk.
3. Falls back to the generic chunker only for pages where fewer than 2 question boundaries were found (e.g. a cover page).

**`pipeline.py`** is the end-to-end orchestrator: `load → chunk (generic or PYQ-aware, based on doc_type) → tag_subject (a keyword-scoring heuristic: e.g. mentions of "Article", "Lok Sabha", "Fundamental Rights" push a chunk toward the "Polity" tag) → embed → store`. It's exposed both as a Python function callable from the API, and as a command-line tool:
```bash
python -m app.ingestion.pipeline ingest --path data/sample_pdfs
python -m app.ingestion.pipeline ingest --path data/sample_pyqs --doc-type pyq
python -m app.ingestion.pipeline stats
```

## 5.4 `app/core/retrieval.py` — turning a query into good context

The `Retriever` class is the deliberately **non-agentic**, deterministic part of the RAG loop (no LLM call here — just embedding + vector math), kept separate so agent tools can call it as a fast utility.

Its `retrieve()` method does a two-stage search:
1. Fetch a larger pool of candidates (`fetch_k=20` by default) by plain similarity search.
2. **Re-rank with MMR (Maximal Marginal Relevance)** down to the final `k` (default 6).

**Why MMR, in plain terms:** plain similarity search can return 6 chunks that are all near-duplicates of each other (e.g. six paragraphs that all say almost the same thing about Article 21), which wastes the agent's limited context window on redundant information. MMR instead balances two goals when picking each chunk: *how relevant is it to the query* **and** *how different is it from chunks already selected*. The trade-off between these two goals is a tunable `lambda_mult` (here `0.6`, favoring relevance somewhat more than diversity). The result: 6 chunks that together cover more *distinct* ground.

`build_context()` then formats the chosen chunks into one labeled text block the LLM can read and cite:
```
[Chunk 1 | laxmikanth_polity.pdf p.42]
<chunk text>
[Chunk 2 | ...]
...
```

## 5.5 `app/agents/llm_factory.py` — LLM "tiering," not provider-picking

Every agent asks for a model by capability **tier** — `"fast"`, `"smart"`, or `"long"` — never by a specific provider or model name. `get_llm(tier)` resolves the tier into an actual `crewai.LLM` object based on `settings.llm_provider`.

| Tier | Used by | Local model | AWS model | Why this tier |
|---|---|---|---|---|
| `fast` | Router, PYQMatcher, DigestCurator, SyllabusMapper, FactChecker | `llama3.2:3b` | Claude 3.5 Haiku | These are **classification-style** judgments ("does this need RAG?", "is this claim supported?"), not open-ended writing — a small model is not just cheaper, it's *more reliably fast*, and several of these calls can happen per single user request. |
| `smart` | Summarizer, AnswerWriter, Formatter, MCQ/MainsQuestion setters, NotesWriter, both Evaluators | `llama3.1:8b` | Claude 3.5 Sonnet | Real writing, planning, and grading benefits from the stronger model's judgment and prose quality. |
| `long` | EssayArchitect only | same model as `smart`, larger `max_tokens` | same model as `smart`, larger `max_tokens` | The essay writer's bottleneck isn't intelligence, it's **output length budget** (~1500 words) — so it reuses the `smart` model with a bigger token allowance rather than paying for a third model. |

This is a cost/latency-vs-quality engineering decision, made explicit and centralized in one place, rather than left as an accidental side effect of "whatever model I happened to import."

## 5.6 `app/agents/tools.py` — how agents reach into the vector database

Two `BaseTool` subclasses, both wrapping the `Retriever` from 5.4:

- **`VectorSearchTool`** (tool name `vector_search`): general-purpose search, optionally filtered by `subject` or `doc_type`. Given to almost every content-generating agent (Summarizer, MCQSetter, NotesWriter, EssayArchitect, both Evaluators).
- **`PYQSearchTool`** (tool name `pyq_search`): search hard-scoped to `doc_type="pyq"` only. Given to the PYQMatcher agent.

Each tool declares its arguments as a Pydantic model (`VectorSearchInput`, `PYQSearchInput`) — this is what lets the underlying LLM know exactly what parameters it's allowed to pass when it decides to call the tool (this parameter-declaration mechanism is what the industry generally calls "function calling" or "tool use" — see Part 2's mention of it). Both tools return **plain formatted text** (never raw Python objects), because that's the only thing an LLM can actually read.

## 5.7 `app/models/agent_schemas.py` — the shapes every agent must produce

As explained in Part 2.7, forcing agents to answer in structured Pydantic schemas (rather than free markdown) is what makes automated checking possible in the first place. A few of the most important schemas:

```python
class BodyPoint(BaseModel):
    point: str
    evidence: str          # never empty — enforced by a guardrail
    evidence_type: EvidenceType   # one of a fixed set: Constitutional Provision, Judgment, ...

class MainsAnswer(BaseModel):
    introduction: str
    body_points: List[BodyPoint]
    conclusion: str
    word_count: int
    gs_paper: Optional[str] = None
    marks: int = 15
```
Every field here maps directly to something a guardrail can mechanically check (Part 5.9). There's a matching schema for essays (`EssayAnswer`), MCQs (`MCQBatch`), mains questions (`MainsQuestionBatch`), notes (`NotesOutput`), evaluations (`MainsEvaluation`, `EssayEvaluation`), PYQ matches (`PYQMatchResult`), syllabus mapping (`SyllabusMap`), current-affairs digests (`DigestOutput`), routing decisions (`RouteDecision`), and — the interesting one — claim verification (`VerificationResult` / `ClaimVerification`, covered next).

## 5.8 `app/agents/agents.py` — the 14 specialist agents

Every agent factory function is decorated `@lru_cache(maxsize=1)` — meaning it's built **once** per running process and reused for every subsequent request (agents are stateless *configuration*, so sharing them is safe and avoids rebuilding the same object repeatedly). Fresh `Task`/`Crew` objects, by contrast, are built **per request**, because those hold per-call state that must not leak between concurrent users.

| # | Agent | Tier | Tools | One-line job |
|---|---|---|---|---|
| 1 | **Router** | fast | — | Decide RAG vs. direct-answer, and produce the best search query |
| 2 | **DirectAnswerer** | smart | — | Answer generic/meta questions with no factual lookup needed |
| 3 | **ContextSummarizer** | smart | vector_search | Search + condense retrieved material losslessly into one dense summary |
| 4 | **AnswerWriter** | smart | — | Draft a substantive answer using only the condensed context |
| 5 | **UPSCFormatter** | smart | — | Reformat a draft into the strict mains-answer structure |
| 6 | **EssayArchitect** | long | vector_search | Research + write a ~1500-word multidimensional essay |
| 7 | **MCQSetter** | smart | vector_search | Write prelims-style multiple-choice questions |
| 8 | **MainsQuestionSetter** | smart | vector_search | Write mains-style exam questions with model outlines |
| 9 | **NotesWriter** | smart | vector_search | Write topper-style condensed notes |
| 10 | **MainsEvaluator** | smart | vector_search | Grade a student's mains answer out of 15, with feedback |
| 11 | **EssayEvaluator** | smart | — | Grade a student's essay out of 125 |
| 12 | **PYQMatcher** | fast | pyq_search | Find and pattern-analyze relevant previous-year questions |
| 13 | **DigestCurator** | fast | — | Summarize a fresh current-affairs document into a compact digest |
| 14 | **FactChecker** | fast | — | Verify each evidence claim against the source it was supposedly drawn from |
| bonus | **SyllabusMapper** | fast | — | Map a topic onto the official UPSC syllabus headings |

Every one of these has `allow_delegation=False` (see Part 7.1 for why), and every `goal`/`backstory` pair is written as a small, specific piece of prompt engineering — notice how the FactChecker's backstory explicitly instructs suspicion ("*You treat every claim as guilty until you can point to the exact part of the source material that supports it*") rather than helpfulness, because a lenient fact-checker would defeat the entire point of having one.

## 5.9 `app/agents/guardrails.py` — enforcement, not hope

This is where Part 3.6's guardrail mechanism gets put to real use. Two structural guardrails and one much more interesting semantic one.

**`mains_answer_guardrail(marks)`** — parses the agent's `MainsAnswer` output and rejects it, with a specific actionable message, if:
- fewer than 3 body points
- any point's `evidence` field is empty or trivially short (`< 10` characters)
- the introduction or conclusion is missing/too short
- the total word count (computed by flattening the structured answer back into plain text and counting real word-tokens) falls outside the marks-appropriate range (e.g. 240–300 words for a 15-mark question, from `marks_to_word_range()`)

**`essay_guardrail()`** does the analogous thing for essays: at least 4 distinct dimensions (paragraphs must not repeat the same `dimension` label), substantive intro/conclusion, and total word count within `essay_target_words ± essay_word_tolerance`.

**The FactChecker + `groundedness_guardrail` — catching *fabricated* evidence, not just *missing* evidence.** This is the project's most sophisticated piece, so it's worth walking through slowly:

The structural guardrail above can confirm a body point's `evidence` field is non-empty and reasonably long — but it has no way to know whether "*S.R. Bommai vs Union of India (1994) held X*" is a real citation that was actually retrieved from the student's uploaded material, or a plausible-sounding fabrication the LLM invented (remember Part 2.1 — this is exactly the hallucination risk LLMs have). Closing that gap needs an **independent check against the source**, not a longer regex.

So `groundedness_guardrail(context_task)` does something clever: it captures a *reference to the live Summarizer `Task` object* from earlier in the same pipeline (not its output value — the object itself), before the crew even starts running. Because the crew runs its tasks strictly in sequence, by the time this guardrail fires (after the Formatter task, which runs later), `context_task.output` is already populated with the Summarizer's condensed context. The guardrail then:

1. Pulls out every `(point, evidence)` pair from the just-formatted `MainsAnswer`.
2. Spins up a **second, nested, one-task `Crew`** — built around the FactChecker agent — and hands it the claims plus the Summarizer's condensed text, asking: *"for each claim, is it genuinely supported by this specific source material — never by whether it merely sounds true?"*
3. Gets back a `VerificationResult` with one `ClaimVerification{claim, supported, matching_source}` per point.
4. If any claim comes back unsupported, the guardrail fails — and, crucially, the retry feedback names the *specific* fabricated-looking claims, so the Formatter's next attempt knows exactly what to fix, rather than just being told "try again."

Because a `Task` in CrewAI only accepts one `guardrail=` callable, and the Formatter task already needed the structural check, `combine_guardrails()` composes the structural guardrail and the groundedness guardrail into a single function that runs *both*, collects *all* problems found, and reports them together — so one retry can fix a wrong word count **and** an unsupported claim at once, instead of the system burning through its retry budget discovering problems one at a time.

One engineering detail worth calling out for anyone learning to build with agent frameworks: running a whole second `Crew().kickoff()` from inside a guardrail — synchronously, mid-execution of the outer task — is not something CrewAI's documentation explicitly promises is safe. The project's own architecture notes describe verifying this directly (mocking the LLM client at the exact internal seam CrewAI uses) before shipping it, rather than assuming it would just work. That's a good habit worth learning from: when you're relying on undocumented behavior of a library, test it in isolation before building a feature on top of it.

This whole check can be switched off via `ENABLE_GROUNDEDNESS_CHECK=false` in `.env` (useful if you're iterating quickly on a slow, CPU-only local Ollama setup and don't want the extra LLM call on every attempt).

## 5.10 `app/agents/crews/` — nine pipelines, one per feature

Each file here wires a small number of agents + tasks into a `Crew` and exposes one plain Python function. This is the layer where Part 3's building blocks (Agent, Task, Crew, Process, guardrail, tools) actually compose into features.

**`qa_crew.py`** — the general "ask anything" flow, and the one that best demonstrates the routing design decision from Part 7.2:
```
_route(question)                       # 1-task crew: Router agent → RouteDecision (use_rag?, search_query)
        │
   plain Python `if decision.use_rag:`
        │
   ┌────┴─────┐
   │           │
False        True
   │           │
_direct_answer()   _rag_answer()
(DirectAnswerer)   (Summarizer → AnswerWriter [→ optional PYQMatcher])
```
Notice: the *branch itself* is an ordinary, unit-testable `if` statement in Python — not a manager-agent LLM call — even though everything on either side of the branch is still "real" CrewAI agents collaborating via `Task(context=[...])` chaining.

**`mains_answer_crew.py`** (the flagship feature) — `Summarizer → AnswerWriter → Formatter (guardrail: structure + word count + groundedness) [→ optional PYQMatcher]`. Full trace in Part 6.2.

**`essay_crew.py`** — a single agent (`EssayArchitect`) holding the search tool directly, rather than a separate Summarizer step, because essay research benefits from *several distinct searches* (one per candidate dimension: economic, ethical, historical...) issued adaptively as the agent writes, rather than one upfront condensation.

**`question_gen_crew.py`** — four related entry points sharing two agents: `run_mcq_batch`, `run_mains_question_batch` (single-topic), `run_prelims_paper`, `run_mains_paper` (multi-question mock papers, tracking `subject_distribution`).

**`notes_crew.py`** — `Summarizer → NotesWriter`.

**`evaluator_crew.py`** — `run_mains_evaluation` and `run_essay_evaluation`; the evaluator agent holds `vector_search` directly so it can fact-check the *student's* claims and surface reference material the answer should have cited but didn't.

**`pyq_matcher_crew.py`** — a single-task crew wrapping the PYQMatcher agent, exposed standalone (feature: "is this topic actually important, or a one-off?").

**`digest_crew.py`** — runs at ingestion time on a freshly-uploaded current-affairs PDF; works directly off raw page text (not `vector_search`) since the document isn't guaranteed to be searchable yet within the same request.

**`syllabus_crew.py`** — a single fast-tier classification task mapping any text onto official UPSC syllabus headings.

## 5.11 `app/services/` — the "front desk"

**`services.py`** is deliberately thin: every function here (1) calls the relevant crew, (2) does small bits of plain-Python post-processing the UI needs (e.g. building a separate, deterministic source list for display — independent of whatever the agent's own tool calls internally searched for), and (3) returns plain Python dicts, never framework-specific response objects — which keeps this module usable from the API, the Streamlit app, and the tests without any of them depending on each other.

**`trace.py`** turns a `CrewOutput` into a simple list of `{agent, task, output_preview}` dicts — purely for showing the student (or an interviewer!) *what each agent actually did*, in order. It never affects the real answer.

**`revision.py`** implements **SM-2**, a well-known spaced-repetition scheduling algorithm (the same family used by Anki). This part has *no* LLM or agent involvement at all — it's pure deterministic scheduling math, worth understanding on its own:
- Every flashcard tracks an `easiness factor (ef)` (starts at 2.5), a current `interval_days`, and a `repetitions` count.
- After a review, the student rates recall quality `0`–`5` (0 = total blackout, 5 = perfect recall).
- If quality `< 3`: the card "resets" — `repetitions = 0`, next review in 1 day (they clearly need to see it again soon).
- If quality `≥ 3`: the interval grows — 1 day → 6 days → then `interval × ef` each time — spacing reviews further and further apart as the student demonstrates they remember it, which is the entire point of spaced repetition (review right before you'd forget, not needlessly often).
- The easiness factor itself adjusts based on how *easy* the recall was, so cards the student finds hard get reviewed more frequently over time than cards they find easy.

## 5.12 `app/api/routes/api.py` and `app/main.py` — the HTTP layer

A standard **FastAPI** app: `app/main.py` builds the `FastAPI()` instance, adds CORS middleware, and mounts one `APIRouter` (from `api.py`) under `/api/v1`. Every endpoint follows the same shape: parse a typed Pydantic request body (`app/models/schemas.py`), call the matching `services.py` function, wrap it in `try/except` (logging the full exception server-side via `logger.exception`, but returning a clean generic `HTTPException(500, ...)` to the client rather than leaking a stack trace), and return a JSON-serializable dict. File-upload endpoints (`/ingest/*`) save uploaded files to a temp directory, ingest them, and always clean up the temp directory in a `finally` block even if ingestion fails.

## 5.13 `frontend/streamlit_app.py` — the classroom

**Streamlit** is a Python framework for building simple, interactive data/web apps without writing any HTML/JS — you write normal Python, and it renders as a browser UI. This project's frontend is a multi-page app (sidebar navigation) with one page per feature: Ask a Question, Write Mains Answer, Write Essay, Generate MCQs, Generate Mains Questions, Generate Notes, Mock Prelims/Mains, Evaluate Mains Answer, Evaluate Essay, PYQ Matcher, Syllabus Mapper, Upload PDFs, Spaced-Repetition Revision, and a Service Status page — each one just calling the corresponding FastAPI endpoint and rendering the structured JSON response as readable widgets (metrics for word count/marks, badges for evidence type, an expandable "agent trace" panel showing what each agent did).

## 5.14 Deployment: `Dockerfile`, `docker-compose*.yml`, `infrastructure/main.tf`, `scripts/`

The whole app (FastAPI on port 8000 + Streamlit on port 8501) ships as **one Docker image** running both processes in the background inside a single container — the simplest topology that still satisfies "open one browser URL" on AWS.

`infrastructure/main.tf` is **Terraform** (infrastructure-as-code — you describe the cloud resources you want in a config file, and Terraform creates/updates/destroys them to match). It provisions: S3 (document storage), ECR (Docker image registry), OpenSearch Serverless (the AWS vector database), ECS Fargate (runs the container without managing servers yourself), and an Application Load Balancer with two routing rules — the default route goes to Streamlit, and an `/api/*` rule routes to FastAPI, both hitting the same running container.

`scripts/` provides one command per lifecycle step: `setup_local.sh` (venv + deps + `.env` + pull Ollama models + sample ingest + run, all with one `--all` flag), `pull_ollama_models.sh`, `ingest_docs.sh`, `deploy_aws.sh` (runs Terraform + prints/opens the resulting public URL), and `teardown.sh` (tears the AWS resources back down, to avoid ongoing cloud costs).

## 5.15 `tests/test_smoke.py`

A **smoke test suite** — not exhaustive correctness testing, but "does the basic machinery even work" testing. It imports every agent factory and every crew module (catching import-time errors before they'd surface at 2am in production), constructs the full FastAPI route table, and — most interestingly — runs `mains_answer_guardrail` directly against both a deliberately malformed `MainsAnswer` (too few points, missing evidence, wrong word count) and a well-formed one, asserting the guardrail actually rejects the first and accepts the second. This is a meaningfully stronger test than "the code runs without crashing" — it verifies the *enforcement logic itself* does its job, independent of whether any real LLM is even called.

---

# Part 6 — Two full end-to-end walkthroughs

## 6.1 Walkthrough: "What are the key features of Article 21?"

1. Student submits the question via the Streamlit "Ask any UPSC question" page → `POST /api/v1/qa`.
2. FastAPI parses it into a `QARequest`, calls `services.qa(...)`.
3. `services.qa()` calls `run_qa()` in `qa_crew.py`, which first runs the **Router** (fast-tier, one task, no tools): given the question, it must decide `use_rag` (almost certainly `True` here — this needs specific constitutional facts) and produce a good `search_query`, returning a structured `RouteDecision`.
4. Since `use_rag=True`, plain Python calls `_rag_answer()`:
   - **Summarizer** agent calls the `vector_search` tool with the router's query, gets back the top MMR-reranked chunks from the vector DB (Part 5.4), and condenses everything into one dense `CondensedContext` (preserving every fact/citation, and explicitly flagging any part of the question it *couldn't* find material for).
   - **AnswerWriter** agent receives that condensed context (via `Task(context=[summarizer_task])`) and writes the actual prose answer, substantiating claims only with what's in the context.
   - (Optionally) **PYQMatcher** runs in parallel within the same crew to surface related previous-year questions.
5. `services.qa()` also runs one extra, purely-for-display, non-agentic retrieval call (via `Retriever` directly) to build a clean "Sources" list for the UI — independent of whatever the agent internally searched for.
6. The full response — answer text, sources, whether RAG was used, the routing reasoning, the agent trace — flows back to the student's browser.

## 6.2 Walkthrough: "Write a 15-mark mains answer on judicial review in India" (with a guardrail retry)

1. `POST /api/v1/mains/write` → `services.mains_write()` → `run_mains_answer()` in `mains_answer_crew.py`.
2. `marks_to_word_range(15)` resolves the target to **240–300 words**.
3. **Summarizer**: searches broadly (`k=8-10`, to cover every angle the question could demand), produces a `CondensedContext`.
4. **AnswerWriter**: drafts a full prose answer identifying "the full demand of the question" and backing every claim with something from the context.
5. **UPSCFormatter**: reformats the draft into the strict `MainsAnswer` schema (introduction / body_points / conclusion), targeting 240–300 words. Its task carries the **combined guardrail** — structural + groundedness (Part 5.9).
6. **Guardrail runs.** Suppose, on the first attempt, one body point cites "*a 2022 NITI Aayog report showing 85% pendency reduction*" that the FactChecker cannot find anywhere in the Summarizer's condensed context. `combine_guardrails()` returns `(False, "...unsupported claims: ['a 2022 NITI Aayog report showing 85% pendency reduction']... Revise these points to use only evidence actually present in the retrieved context...")`.
7. **CrewAI automatically re-runs the Formatter task**, feeding it that exact feedback. The agent now knows precisely which point is the problem and rewrites just that evidence, grounding it in something the Summarizer actually retrieved.
8. This can repeat up to `guardrail_max_retries` (default 4) times. If it still fails after all retries, CrewAI surfaces the failure rather than silently shipping a bad answer.
9. Once accepted, `mains_answer.word_count` is set by the guardrail itself (computed once, at acceptance time — not trusted from the LLM's own claim), plus `marks` and `gs_paper` are attached.
10. (Optionally) **PYQMatcher** runs alongside, so the student also sees "this exact topic has come up in GS-II in 3 of the last 10 years" next to their model answer.
11. Response includes the structured answer, PYQ matches, the full agent trace (showing every task including the guardrail retry), and a `groundedness_checked: true` flag.

---

# Part 7 — The big design decisions, explained simply

### 7.1 Why 14 *narrow* agents, all with `allow_delegation=False`?

CrewAI agents *can* be allowed to delegate work to each other dynamically (another agent decides who does what, at runtime). This project deliberately disables that everywhere. With 14 specialists, letting any of them freely hand off work to any other would make execution unpredictable (you can no longer easily reason about "what will this request cost/how long will it take") and risks runaway chains of LLM calls. Instead, **all branching and hand-off is explicit, ordinary Python code** (Part 5.10's crew files, Part 5.11's services layer) — "crews as composable units, orchestrated by light application logic," in the project's own words.

### 7.2 Why is routing a Python `if`, not CrewAI's `Process.hierarchical`?

Covered conceptually in Part 3.4. In concrete terms: `Process.hierarchical` would mean an LLM call decides "use RAG or not" *every single time*, adding latency and cost, and — critically for anyone building production software — making that decision **impossible to unit-test** the normal way (you can't assert `assert route(question) == "rag"` against a nondeterministic manager-agent the way you can against a plain function). The chosen approach keeps a tiny, testable *AI* judgment call (the Router agent — deciding *whether RAG is needed* genuinely does require some judgment) but keeps the *branching logic itself* in ordinary, deterministic Python.

### 7.3 Why does the vector-DB collection name encode the embeddings provider + dimension?

Covered in 5.2. This is a small but genuinely important lesson in robustness: two different embedding models produce vectors that are numerically incompatible even if stored in "the same" database — comparing a Bedrock-Titan vector against an Ollama-nomic vector with cosine similarity produces meaningless numbers, silently. Baking the provider+dimension into the collection name turns a *silent data-corruption bug* into an *automatic fresh start*.

### 7.4 Why does the FactChecker exist as a separate agent, rather than asking the Formatter to "double check itself"?

Because an LLM checking its own output for accuracy tends to rubber-stamp itself — it already "believes" what it wrote. A genuinely useful check needs a **different framing and a different (even if same-model) pass**: the FactChecker is given *only* the claim and the source text, explicitly told plausibility doesn't count, and asked to be skeptical by default. Separating this into its own agent, its own task, its own (fast-tier, cheaper) LLM call, keeps that skepticism intact rather than diluting it into "please also double-check your answer" buried inside a longer writing prompt.

---

# Part 8 — What's honestly still rough (from the project's own audit notes)

Good engineering documentation says what *isn't* solved yet, not just what is. Straight from `ARCHITECTURE.md`, in plain terms:

- **Evaluator sub-scores aren't guardrailed to sum to the total.** The prompt tells the LLM "make sure content(5) + structure(3) + ... adds up to `total`," but nothing in code currently checks this arithmetic — an easy, cheap fix (no extra LLM call needed) that just hasn't been wired in yet.
- **No guardrails on MCQ/mains-question batches** — nothing currently checks the returned question count matches what was requested, or that an MCQ's stated `answer` is actually one of its own `options`.
- **MMR reranking re-embeds every candidate chunk on every single search** (Part 5.4), even though the vector store already computes those embeddings internally during the initial search — it's just not returned for reuse. This is real, measurable added latency/cost on every retrieval.
- **Groundedness checking (the FactChecker) only covers the mains-answer flow** — the plain Q&A flow and the essay flow currently have no equivalent fact-checking pass.
- **The OpenSearch Serverless path (the AWS vector DB) has never actually been exercised against a real AWS deployment** from within the project's own development — the code follows standard OpenSearch syntax, but hasn't been confirmed with a real `terraform apply` + real data.

Reading a project's own honest gap list like this is a genuinely valuable habit — it tells you exactly where to focus if you extend the project, and it's a good template for documenting your own future projects.

---

# Part 9 — Glossary

| Term | Plain-English meaning |
|---|---|
| **LLM** | A language model (e.g. GPT, Claude, Llama) that generates text by predicting what comes next. |
| **Prompt** | The text you feed an LLM to get it to do something specific. |
| **Embedding** | A list of numbers representing a piece of text's *meaning*, so meaning can be compared mathematically. |
| **Vector database** | A database optimized for storing embeddings and finding the closest ones to a query. |
| **Cosine similarity** | A way of measuring how "similar in direction" two vectors are — the standard way to compare embeddings. |
| **RAG (Retrieval-Augmented Generation)** | Look up relevant real text first, then have the LLM answer using that text as grounding, instead of relying purely on its trained-in memory. |
| **Chunking** | Splitting a long document into smaller, semantically coherent pieces before embedding/storing it. |
| **MMR (Maximal Marginal Relevance)** | A re-ranking technique that picks search results balancing relevance *and* diversity, avoiding near-duplicate results. |
| **Hallucination** | An LLM confidently generating false or invented information that sounds plausible. |
| **Structured output** | Forcing an LLM to answer in a specific machine-readable schema (usually JSON/Pydantic) instead of free text. |
| **Agent (CrewAI)** | A configured "specialist" (role, goal, backstory, LLM, tools) that can be assigned tasks. |
| **Task (CrewAI)** | One unit of work given to an agent, with instructions and an expected output shape. |
| **Crew (CrewAI)** | A team of agents + tasks that actually runs (executes) together. |
| **Process (sequential/hierarchical)** | The strategy for how a crew's tasks execute — in fixed order, or dynamically decided by a manager-agent. |
| **Tool (CrewAI)** | A function an agent can choose to call (e.g. "search the database") in addition to just generating text. |
| **Guardrail (CrewAI)** | A validation function attached to a task; on failure, CrewAI automatically retries the task with the failure fed back as feedback. |
| **Tiering (LLM)** | Assigning cheaper/faster models to simple tasks and stronger models to hard tasks, based on actual task difficulty. |
| **SM-2** | A spaced-repetition scheduling algorithm that spaces out flashcard reviews based on how well you remembered them last time. |
| **Terraform** | A tool for describing cloud infrastructure (servers, databases, networking) as code, so it can be created/updated repeatably. |
| **FastAPI / Streamlit** | FastAPI: a Python framework for building HTTP APIs. Streamlit: a Python framework for building simple interactive web UIs, without writing HTML/JS. |

---

*This document was generated by reading the actual source code of the uploaded `upsc-sarathi-agentic-v4.zip` project — every code snippet above is taken verbatim from the project's files, not reconstructed from memory.*
