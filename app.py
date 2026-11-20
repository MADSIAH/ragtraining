"""RAG Tutor: Streamlit chat app (Phase 8).

Run:  .venv/bin/streamlit run app.py   (binds to localhost via .streamlit/config.toml; Ollama must be running)
Default engine = Local (D22): qwen3:14b + local reranker, $0, works without an API key.
The Claude engine (Opus + Claude rerank, D19) and agent mode need the API key in .env.
"""
import pandas as pd
import streamlit as st

from ragtraining.agent.loop import agent_answer
from ragtraining.config import CLAUDE_MODEL, LOCAL_MODEL
from ragtraining.generate.tutor import Tutor
from ragtraining.llm import cost_so_far

st.set_page_config(page_title="RAG Tutor", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading indexes…")
def get_tutors() -> tuple[Tutor, Tutor]:
    local = Tutor()                                                  # D22 default: answers on this Mac
    return local, Tutor(retriever=local.retriever, model=CLAUDE_MODEL)  # same indexes, answers via the API


local_tutor, claude_tutor = get_tutors()
courses = sorted(local_tutor.retriever.chunks.course.unique())

with st.sidebar:
    st.header("Settings")
    engine = st.radio("Engine", ["Local (free, offline)", "Claude (API, best quality)"], index=0,
                      help=f"Local = {LOCAL_MODEL.split(':', 1)[1]} via Ollama + local reranker: $0, no API key, "
                           "slower and a bit less precise · Claude = Opus 5.5 + Claude rerank, ~$0.04/question")
    local = engine.startswith("Local")
    mode = st.radio("Mode", ["Tutor (single search)", "Agent (plan → search → grade → retry)"], disabled=local,
                    help="Agent mode uses Claude (Haiku) to plan, so it's only available with the Claude engine")
    if local:
        mode = "Tutor (single search)"
    method = st.selectbox("Retrieval", ["hybrid", "cosine", "bm25", "cos_euclid", "euclid"], index=0,
                          help="hybrid = Qwen3 dense + BM25 fused with RRF (best in Phase 4)")
    rerank = st.selectbox("Reranker", ["local", "none"] if local else ["claude", "local", "none"], index=0,
                          help="claude = Haiku 5.5 reranker (best in Phase 4, ~$0.0016/question) · local = free")
    course_filter = st.multiselect("Only these courses (optional)", courses)
    st.caption(f"Claude spend this session: ${cost_so_far():.3f}")

st.title("🎓 RAG Tutor")
st.caption("Answers only from your course material, with citations. Rules: GUIDELINES.md")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])

question = st.chat_input("Ask about your courses…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your material and writing the answer…"):
            if mode.startswith("Agent"):
                res = agent_answer(question, claude_tutor)
            else:
                kw = dict(method=method, how="rrf") if method == "hybrid" else dict(method=method)
                kw["rerank"] = None if rerank == "none" else rerank
                if rerank == "claude":
                    kw["rerank_model"] = "claude-haiku-5-5"
                if course_filter:
                    kw["courses"] = course_filter
                res = (local_tutor if local else claude_tutor).answer(question, **kw)
        st.markdown(res["answer"])
        if res.get("blocked"):
            st.info(f"Guardrail: {res['blocked']} (top score {res.get('top_score', 0):.3f})")
        if res.get("citation_problems"):
            st.warning("Citation check: " + "; ".join(res["citation_problems"])
                       + (" (still missing after one retry)" if res.get("citation_retry") else ""))
        elif res.get("citation_retry"):
            st.caption("Citation check: the first answer had no [S#] citations, re-asked once (P-53).")
        with st.expander("Sources"):
            for s in res.get("sources", []):
                st.write(s)
        if res.get("contexts"):
            with st.expander("Retrieved chunks (what the model saw)"):
                for i, c in enumerate(res["contexts"], 1):
                    st.markdown(f"**S{i}**")
                    st.text(c[:800])
        if res.get("trace"):
            with st.expander("Agent trace"):
                st.dataframe(pd.DataFrame(res["trace"]).astype(str))
    st.session_state.history.append(dict(question=question, answer=res["answer"]))
