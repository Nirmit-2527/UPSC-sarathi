"""
upsc-sarathi-agentic :: frontend/streamlit_app.py

Multi-page Streamlit UI that calls the FastAPI backend. Extends the
non-agentic reference UI with: a model mains-answer writer, an essay
writer, an essay evaluator, a standalone PYQ matcher, dedicated
current-affairs/PYQ upload tabs, and an "agent trace" expander on every
agentic page (shows which agent did what, in order - great for demos).

Run:
    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations
import os
import json
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(page_title="UPSC Sarathi Agentic", page_icon="📚", layout="wide")


# ----- sidebar -----
with st.sidebar:
    st.title("📚 UPSC Sarathi Agentic")
    st.caption("Agentic RAG (CrewAI) CSE prep assistant")
    page = st.radio(
        "Feature",
        [
            "💬 Ask a Question",
            "✍️ Write Mains Answer",
            "📜 Write Essay",
            "📝 Generate MCQs (Prelims)",
            "🧾 Generate Mains Questions",
            "📑 Generate Notes",
            "🎯 Mock Test - Prelims",
            "🎯 Mock Test - Mains",
            "🧪 Evaluate Mains Answer",
            "🧐 Evaluate Essay",
            "🔎 PYQ Matcher",
            "🗂️ Syllabus Mapper",
            "📤 Upload PDFs",
            "🔁 Spaced Revision",
            "ℹ️ Status",
        ],
    )
    st.divider()
    api_base = st.text_input("API base", API_BASE)
    st.session_state["api"] = api_base


def post(path: str, payload: dict | None = None, files=None):
    url = f"{st.session_state['api']}{path}"
    try:
        if files:
            r = requests.post(url, files=files, timeout=600)
        else:
            r = requests.post(url, json=payload or {}, timeout=600)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def get(path: str):
    try:
        r = requests.get(f"{st.session_state['api']}{path}", timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def render_trace(trace: list | None):
    if not trace:
        return
    with st.expander(f"🕵️ Agent trace ({len(trace)} step(s))", expanded=False):
        for i, step in enumerate(trace, 1):
            st.markdown(f"**{i}. {step.get('agent', '?')}** — {step.get('task', '')}")
            st.caption(step.get("output_preview", ""))
            st.divider()


def render_pyq_matches(pyq):
    if not pyq:
        return
    with st.expander("📚 Related Previous Year Questions", expanded=False):
        st.info(pyq.get("frequency_insight", ""))
        for m in pyq.get("matches", []):
            tag = " ".join(x for x in [m.get("year") and str(m["year"]), m.get("exam_type"), m.get("paper")] if x)
            st.markdown(f"**[{tag}]** {m.get('question','')}")
            if m.get("note"):
                st.caption(m["note"])
            st.divider()


EVIDENCE_EMOJI = {
    "Constitutional Provision": "📜",
    "Supreme Court / High Court Judgment": "⚖️",
    "Committee or Commission Report": "📋",
    "Government Scheme or Policy": "🏛️",
    "Statistical Data": "📊",
    "Parliamentary Report or Act": "🏦",
    "Expert Opinion / Other": "💬",
}


def render_mains_answer(ans: dict):
    st.markdown("### 📖 Introduction")
    st.write(ans["introduction"])
    st.markdown("### 🧩 Body")
    for bp in ans["body_points"]:
        emoji = EVIDENCE_EMOJI.get(bp["evidence_type"], "•")
        st.markdown(f"- **{bp['point']}**")
        st.caption(f"{emoji} {bp['evidence_type']}: {bp['evidence']}")
    st.markdown("### 🏁 Conclusion")
    st.write(ans["conclusion"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Word count", ans.get("word_count", "-"))
    c2.metric("Marks", ans.get("marks", "-"))
    c3.metric("GS Paper", ans.get("gs_paper") or "-")


def render_essay(essay: dict):
    st.markdown(f"## {essay.get('title','')}")
    st.write(essay["introduction"])
    for p in essay["body_paragraphs"]:
        st.markdown(f"**[{p['dimension']}]**")
        st.write(p["content"])
    st.markdown("### Conclusion")
    st.write(essay["conclusion"])
    st.metric("Word count", essay.get("word_count", "-"))


# ============================================================
if page == "💬 Ask a Question":
    st.header("💬 Ask any UPSC question")
    q = st.text_area("Your question", height=110,
                     placeholder="e.g. Explain the doctrine of Basic Structure with relevant cases.")
    col1, col2, col3 = st.columns([1, 1, 1])
    subj = col1.selectbox("Subject filter (optional)",
                          ["", "Polity", "History", "Geography", "Economy",
                           "Environment", "Science", "Ethics", "IR"])
    k = col2.slider("Retrieval depth (k)", 3, 12, 6)
    include_pyq = col3.checkbox("Include related PYQs", value=True)
    if st.button("Ask Sarathi 🚀", type="primary", use_container_width=True) and q.strip():
        with st.spinner("Routing, retrieving, and reasoning..."):
            res = post("/qa", {"question": q, "k": k, "subject": subj or None, "include_pyq": include_pyq})
        if res:
            st.caption(f"{'🔎 Used RAG' if res['used_rag'] else '⚡ Answered directly'} — {res['route_reason']}")
            st.markdown("### 📖 Answer")
            st.markdown(res["answer"])
            if res.get("sources"):
                with st.expander("📚 Sources & retrieved chunks", expanded=False):
                    for s in res["sources"]:
                        st.markdown(f"**{s['source']}** — p.{s['page']} (score={s['score']:.3f})")
                        st.caption(s["snippet"])
                        st.divider()
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "✍️ Write Mains Answer":
    st.header("✍️ Model Mains Answer Writer")
    st.caption("Catchy intro · point-format body, every point with evidence · catchy conclusion · word-count enforced · evidence fact-checked")
    q = st.text_area("Mains question", height=80,
                     placeholder="e.g. Critically examine the relevance of Directive Principles in 21st century India.")
    c1, c2 = st.columns(2)
    marks = c1.select_slider("Marks", options=[10, 15, 20], value=15)
    gs = c2.selectbox("GS Paper (optional)", ["", "GS-I", "GS-II", "GS-III", "GS-IV"])
    if st.button("Write model answer", type="primary") and q.strip():
        with st.spinner("Summarizing, drafting, formatting, and fact-checking evidence (guardrails may auto-retry)..."):
            res = post("/mains/write", {"question": q, "marks": marks, "gs_paper": gs or None})
        if res and res.get("mains_answer"):
            if res.get("groundedness_checked"):
                st.caption("✅ Every evidence point was independently fact-checked against the retrieved source material by the FactChecker agent.")
            render_mains_answer(res["mains_answer"])
            render_pyq_matches(res.get("pyq_matches"))
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "📜 Write Essay":
    st.header("📜 UPSC Essay Writer")
    st.caption("Multidimensional, flowing prose, target ~1500 words")
    topic = st.text_input("Essay topic", "Technology cannot replace manpower but can advance its efficiency")
    if st.button("Write essay", type="primary") and topic.strip():
        with st.spinner("Researching multiple angles and writing (this can take a couple of minutes)..."):
            res = post("/essay/write", {"topic": topic})
        if res and res.get("essay"):
            render_essay(res["essay"])
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "📝 Generate MCQs (Prelims)":
    st.header("📝 Generate Prelims MCQs")
    topic = st.text_input("Topic", "Fundamental Rights")
    c1, c2 = st.columns(2)
    n = c1.slider("Number of MCQs", 5, 30, 10)
    subj = c2.selectbox("Subject filter (optional)", ["", "Polity", "History", "Geography",
                                                       "Economy", "Environment", "Science", "Ethics", "IR"])
    if st.button("Generate MCQs", type="primary"):
        with st.spinner("Generating UPSC-style MCQs..."):
            res = post("/generate/mcq", {"topic": topic, "n": n, "subject": subj or None})
        batch = res.get("mcq_batch") if res else None
        if batch:
            for q in batch["questions"]:
                with st.expander(f"Q{q['id']}. {q['stem'][:120]}..."):
                    st.markdown(f"**{q['stem']}**")
                    for k, v in q["options"].items():
                        st.markdown(f"- **{k}.** {v}")
                    st.success(f"Answer: {q['answer']}  |  Difficulty: {q.get('difficulty','-')}")
                    st.info(f"**Explanation:** {q['explanation']}")
                    st.caption(f"Syllabus: {q.get('syllabus_tag','-')}  ·  Source: {q.get('source_ref','-')}")
            if batch.get("sources"):
                st.caption(f"Generated from: {', '.join(batch['sources'])}")
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "🧾 Generate Mains Questions":
    st.header("🧾 Generate Mains Questions")
    topic = st.text_input("Topic", "Federalism in India")
    c1, c2 = st.columns(2)
    n = c1.slider("Number of questions", 1, 10, 5)
    gs = c2.selectbox("GS Paper", ["", "GS-I", "GS-II", "GS-III", "GS-IV", "Essay"])
    if st.button("Generate", type="primary"):
        with st.spinner("Crafting mains questions..."):
            res = post("/generate/mains", {"topic": topic, "n": n, "gs_paper": gs or None})
        batch = res.get("mains_question_batch") if res else None
        if batch:
            for q in batch["questions"]:
                with st.expander(f"Q{q['id']}. ({q.get('marks','?')} marks) {q['question'][:80]}..."):
                    st.markdown(f"**{q['question']}**")
                    st.caption(f"GS: {q.get('gs_paper','-')} · Words: {q.get('word_limit','-')} · Tag: {q.get('syllabus_tag','-')}")
                    st.markdown("**Model outline:**")
                    for o in q.get("model_outline", []):
                        st.markdown(f"- {o}")
                    st.markdown("**Key points:**")
                    for p in q.get("key_points", []):
                        st.markdown(f"- {p}")
                    if q.get("recent_examples"):
                        st.markdown("**Recent examples:**")
                        for r in q["recent_examples"]:
                            st.markdown(f"- {r}")
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "📑 Generate Notes":
    st.header("📑 Generate UPSC Notes")
    topic = st.text_input("Topic", "Indian Federalism")
    subj = st.selectbox("Subject (optional)",
                        ["", "Polity", "History", "Geography", "Economy",
                         "Environment", "Science", "Ethics", "IR"])
    if st.button("Make notes", type="primary"):
        with st.spinner("Compiling topper-style notes..."):
            res = post("/generate/notes", {"topic": topic, "subject": subj or None})
        notes = res.get("notes") if res else None
        if notes:
            st.markdown(notes["notes_markdown"])
            st.download_button(
                "⬇️ Download notes (.md)",
                data=notes["notes_markdown"],
                file_name=f"{topic.replace(' ', '_')}_notes.md",
                mime="text/markdown",
            )
            if notes.get("mains_angle"):
                st.markdown("**Likely mains-question angles:**")
                for a in notes["mains_angle"]:
                    st.markdown(f"- {a}")
            if notes.get("prelims_facts"):
                st.markdown("**Prelims facts:**")
                for f in notes["prelims_facts"]:
                    st.markdown(f"- {f}")
            st.caption(f"Sources: {', '.join(notes.get('sources', []))}")
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "🎯 Mock Test - Prelims":
    st.header("🎯 Mock Prelims Mini-Paper")
    topics = st.text_input("Topics (comma-separated)",
                           "Polity, Modern History, Economy, Environment, Geography").split(",")
    topics = [t.strip() for t in topics if t.strip()]
    n = st.slider("Number of MCQs", 10, 100, 25)
    if st.button("Build paper", type="primary"):
        with st.spinner("Generating paper... this can take ~1-2 minutes"):
            res = post("/test/prelims", {"topics": topics, "n": n})
        batch = res.get("mcq_batch") if res else None
        if batch:
            if batch.get("subject_distribution"):
                st.json(batch["subject_distribution"])
            for q in batch.get("questions", []):
                with st.expander(f"Q{q['id']}. {q['stem'][:80]}..."):
                    st.markdown(f"**{q['stem']}**")
                    for k, v in q["options"].items():
                        st.markdown(f"- **{k}.** {v}")
                    st.success(f"Answer: {q['answer']}")
                    st.info(q["explanation"])
            st.download_button("⬇️ Download paper as JSON",
                               data=json.dumps(batch, indent=2),
                               file_name="prelims_mock.json")
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "🎯 Mock Test - Mains":
    st.header("🎯 Mock Mains Section")
    topic = st.text_input("Topic", "Indian Polity & Governance")
    gs = st.selectbox("GS Paper", ["GS-I", "GS-II", "GS-III", "GS-IV"], index=1)
    n = st.slider("Number of questions", 1, 10, 5)
    if st.button("Build paper", type="primary"):
        with st.spinner("Generating..."):
            res = post("/test/mains", {"topic": topic, "gs_paper": gs, "n": n})
        batch = res.get("mains_question_batch") if res else None
        if batch:
            st.metric("Total marks", batch.get("total_marks", "?"))
            for q in batch.get("questions", []):
                st.markdown(f"**Q{q['id']}. ({q.get('marks')} marks, {q.get('word_limit')} words)** {q['question']}")
                with st.expander("Model outline"):
                    for o in q.get("model_outline", []):
                        st.markdown(f"- {o}")
                st.divider()
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "🧪 Evaluate Mains Answer":
    st.header("🧪 Mains Answer Evaluator")
    q = st.text_area("Question", height=80,
                     placeholder="e.g. Critically examine the relevance of Directive Principles in 21st century India.")
    a = st.text_area("Your answer", height=300)
    if st.button("Evaluate (15 marks)", type="primary") and q and a:
        with st.spinner("UPSC evaluator scoring..."):
            res = post("/evaluate", {"question": q, "answer": a})
        s = res.get("evaluation") if res else None
        if s:
            cols = st.columns(6)
            cols[0].metric("Content /5", s["content"])
            cols[1].metric("Structure /3", s["structure"])
            cols[2].metric("Examples /3", s["examples"])
            cols[3].metric("Language /2", s["language"])
            cols[4].metric("Conclusion /2", s["conclusion"])
            cols[5].metric("Total /15", s["total"])

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("✅ Strengths")
                for x in s.get("strengths", []):
                    st.markdown(f"- {x}")
                st.subheader("🔴 Weaknesses")
                for x in s.get("weaknesses", []):
                    st.markdown(f"- {x}")
            with c2:
                st.subheader("🎯 Missed key points")
                for x in s.get("missed_points", []):
                    st.markdown(f"- {x}")
                st.subheader("💡 Improvement tips")
                for x in s.get("improvement_tips", []):
                    st.markdown(f"- {x}")
            st.subheader("✍️ Improved intro")
            st.write(s.get("improved_intro", ""))
            st.subheader("✍️ Improved conclusion")
            st.write(s.get("improved_conclusion", ""))
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "🧐 Evaluate Essay":
    st.header("🧐 Essay Evaluator")
    topic = st.text_input("Essay topic")
    essay_text = st.text_area("Your essay", height=400)
    if st.button("Evaluate (125 marks)", type="primary") and topic and essay_text:
        with st.spinner("UPSC essay evaluator scoring..."):
            res = post("/evaluate/essay", {"topic": topic, "essay_text": essay_text})
        s = res.get("evaluation") if res else None
        if s:
            cols = st.columns(5)
            cols[0].metric("Content /35", s["content_relevance"])
            cols[1].metric("Multidim. /25", s["multidimensionality"])
            cols[2].metric("Structure /25", s["structure_coherence"])
            cols[3].metric("Language /25", s["language_expression"])
            cols[4].metric("Conclusion /15", s["conclusion"])
            st.metric("Total /125", s["total"])
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("✅ Strengths")
                for x in s.get("strengths", []):
                    st.markdown(f"- {x}")
                st.subheader("🔴 Weaknesses")
                for x in s.get("weaknesses", []):
                    st.markdown(f"- {x}")
            with c2:
                st.subheader("💡 Improvement tips")
                for x in s.get("improvement_tips", []):
                    st.markdown(f"- {x}")
                st.subheader("🧭 Missing dimensions")
                for x in s.get("missing_dimensions", []):
                    st.markdown(f"- {x}")
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "🔎 PYQ Matcher":
    st.header("🔎 Previous Year Question Matcher")
    st.caption("Semantic search over the ingested PYQ bank (last ~10 years, as uploaded)")
    topic = st.text_input("Topic", "Judicial review")
    k = st.slider("Number of matches", 3, 30, 10)
    if st.button("Find PYQs", type="primary") and topic.strip():
        with st.spinner("Matching against the PYQ bank..."):
            res = post("/pyq/match", {"topic": topic, "k": k})
        pyq = res.get("pyq_result") if res else None
        if pyq:
            render_pyq_matches(pyq)
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "🗂️ Syllabus Mapper":
    st.header("🗂️ Map text to UPSC syllabus")
    text = st.text_area("Paste a current-affairs snippet, news article, or topic description",
                        height=240)
    if st.button("Map", type="primary") and text:
        with st.spinner("Mapping..."):
            res = post("/syllabus/map", {"text": text})
        sm = res.get("syllabus_map") if res else None
        if sm:
            st.json(sm)
            render_trace(res.get("agent_trace"))


# ============================================================
elif page == "📤 Upload PDFs":
    st.header("📤 Upload UPSC PDFs to the Vector DB")
    tab1, tab2, tab3 = st.tabs(["📘 Study material", "🗓️ Current affairs", "📋 Previous Year Questions"])

    with tab1:
        st.caption("Standard notes, books, syllabus PDFs - chunked generically (chapter/section aware).")
        files = st.file_uploader("Select PDFs", type=["pdf"], accept_multiple_files=True, key="study")
        if files and st.button("Ingest study material", type="primary"):
            with st.spinner(f"Ingesting {len(files)} file(s)..."):
                multi_files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
                res = post("/ingest/upload", files=multi_files)
            if res:
                st.success(f"Added {res.get('total_chunks_added', 0)} chunks. Total in store: {res.get('store_size', '?')}")
                st.json(res)

    with tab2:
        st.caption("Daily current-affairs PDFs - tagged with today's date; a digest agent summarises each one.")
        files = st.file_uploader("Select PDFs", type=["pdf"], accept_multiple_files=True, key="ca")
        if files and st.button("Ingest current affairs", type="primary"):
            with st.spinner(f"Ingesting and digesting {len(files)} file(s)..."):
                multi_files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
                res = post("/ingest/current-affairs", files=multi_files)
            if res:
                st.success(f"Added {res.get('total_chunks_added', 0)} chunks. Total in store: {res.get('store_size', '?')}")
                for d in res.get("digests", []):
                    with st.expander(f"📰 Digest: {d['source_file']}"):
                        st.write(d["digest"]["headline_summary"])
                        for f in d["digest"].get("prelims_facts", []):
                            st.markdown(f"- {f}")

    with tab3:
        st.caption("PYQ compilations - chunked per-question; name files like 'Mains_GS2_2022.pdf' for best metadata extraction.")
        files = st.file_uploader("Select PDFs", type=["pdf"], accept_multiple_files=True, key="pyq")
        if files and st.button("Ingest PYQs", type="primary"):
            with st.spinner(f"Ingesting {len(files)} file(s)..."):
                multi_files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
                res = post("/ingest/pyq", files=multi_files)
            if res:
                st.success(f"Added {res.get('total_chunks_added', 0)} chunks. Total in store: {res.get('store_size', '?')}")
                st.json(res)


# ============================================================
elif page == "🔁 Spaced Revision":
    st.header("🔁 Spaced-Repetition Revision (SM-2)")
    st.caption("Add cards from your study, then come back daily to review what's due.")
    with st.expander("Add a new revision card"):
        t = st.text_input("Topic", "Polity")
        front = st.text_area("Front (question / prompt)")
        back = st.text_area("Back (answer / explanation)")
        if st.button("Add card") and front and back:
            res = post("/sr/add", {"topic": t, "front": front, "back": back})
            if res:
                st.success(f"Added card: {res.get('card_id','?')[:8]}…")

    st.subheader("Due today")
    due = get("/sr/due?limit=20") or []
    for c in due:
        with st.expander(f"[{c['topic']}] {c['front'][:80]}"):
            st.markdown(c["front"])
            if st.button(f"Show answer {c['card_id']}", key=f"show_{c['card_id']}"):
                st.info(c["back"])
            cols = st.columns(6)
            for q in range(6):
                if cols[q].button(str(q), key=f"rev_{c['card_id']}_{q}"):
                    post("/sr/review", {"card_id": c["card_id"], "quality": q})
                    st.rerun()


# ============================================================
elif page == "ℹ️ Status":
    st.header("ℹ️ Service status")
    h = get("/health")
    if h:
        st.json(h)
    st.divider()
    s = get("/stats")
    if s:
        st.metric("Vectors stored", s.get("vector_store_count", -1))
