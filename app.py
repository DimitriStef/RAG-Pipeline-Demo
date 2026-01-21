import streamlit as st
from rag.ingest import run_ingestion
from rag.llm import load_llm
from rag.chain import build_rag_chain
from rag.retrieval_gate import parse_gate_decision
from utils.config import CHATBOT_SYSTEM, CHATBOT_PROMPT


st.set_page_config(page_title="RAG Demo", layout="wide", page_icon=":mag_right:")

CSS = """
/* Simple chat bubbles and source card styles */
.chat {
  display: block;
  padding: 14px 18px;
  border-radius: 12px;
  margin-bottom: 8px;
}
.user { background: #e6f0ff; color: #022c64; text-align: right }
.assistant { background: #f7f7f9; color: #111; text-align: left }
.source-card { border: 1px solid #eee; padding: 12px; border-radius: 8px; margin-bottom: 8px; background: #fff }
.muted { color: #666; font-size: 12px }
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.header("RAG Demo")
    st.markdown("Use the left panel to adjust settings. Results show on submit.")

# with col2:
#     st.image("link", width=120)


# Cache resources to avoid reloading on every interaction
@st.cache_resource
def cached_corpus():
    run_ingestion()

@st.cache_resource
def cached_llm():
    return load_llm()

@st.cache_resource
def cached_rag_chain(_llm, k, use_mmr, use_bm25, use_rerank, system, prompt):
    return build_rag_chain(_llm, k, use_mmr, use_bm25, use_rerank, system, prompt)


# Sidebar controls
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Max sources to return (top_k)", 1, 10, 6)
use_mmr = st.sidebar.checkbox("Use MMR retrieval", value=True)
use_bm25 = st.sidebar.checkbox("Use BM25 hybrid retrieval", value=True)
use_rerank = st.sidebar.checkbox("Use cross-encoder reranking", value=True)

if st.sidebar.button("Re-run ingestion"):
    with st.spinner("Re-running ingestion..."):
        cached_corpus.clear()
        cached_corpus()
        st.success("Ingestion complete — caches cleared.")

# Build the RAG gate and chatbot chains
cached_corpus()
llm = cached_llm()
rag_chain = cached_rag_chain(llm, top_k, use_mmr, use_bm25, use_rerank, CHATBOT_SYSTEM, CHATBOT_PROMPT)
 
with st.container():
    query = st.chat_input(
        "Ask a question",
        key="query_input"
    )

    if query and query.strip():

        with st.spinner("Evaluating context sufficiency..."):
            result = rag_chain.invoke({
                "input": query,
                "top_k": top_k
            })
            parse_gate_decision(query, result.get("context"))

        with st.spinner("Retrieving context and generating answer..."):
            result = rag_chain.invoke({
                "input": query,
                "top_k": top_k
            })

        answer = result.get("answer")

        left, right = st.columns([3, 1])

        with left:
            st.subheader("Answer")
            st.write(answer)

            st.download_button("Download answer", answer, file_name="answer.txt")

            meta_info = []
            if isinstance(result, dict):
                meta_info.append(f"Sources: {len(result.get('source', []))}")

        with right:
            st.subheader("Context Documents")

            context_docs = result.get("context")

            if not context_docs:
                st.write("No documents returned.")
            else:
                for i, doc in enumerate(context_docs, 1):
                    meta = getattr(doc, 'metadata', {}) or {}
                    src = meta.get('source')
                    content = getattr(doc, 'page_content', str(doc))
                    
                    with st.expander(f"[{i}/{len(context_docs)}] {src}"):
                        if meta.get('start_index') is not None:
                            st.markdown(f"**Index:** {meta.get('start_index')}")
                        preview = content[:800].replace("\n", " ")
                        st.write(preview)
                        st.download_button(f"Download doc {i}", preview, file_name=f"doc_{i}.txt")

