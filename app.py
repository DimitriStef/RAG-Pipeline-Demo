import streamlit as st
from rag.ingest import run_ingestion
from rag.llm import load_llm
from rag.chain import build_rag_chain


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
    st.markdown("Use the left panel to ask questions. Results show on submit.")

# with col2:
#     st.image("link", width=120)


# Cache resources to avoid reloading on every interaction
@st.cache_resource
def cached_corpus():
    run_ingestion("data/urls.txt")

@st.cache_resource
def cached_llm():
    return load_llm()

@st.cache_resource
def cached_rag_chain(_llm, k, use_mmr):
    return build_rag_chain(_llm, k, use_mmr)


# Sidebar controls
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Max sources to return (top_k)", 1, 20, 4)
use_mmr = st.sidebar.checkbox("Use MMR retrieval", value=True)

if st.sidebar.button("Re-run ingestion"):
    with st.spinner("Re-running ingestion..."):
        cached_corpus.clear()
        cached_corpus()
        st.success("Ingestion complete — caches cleared.")

# Build the RAG chain
cached_corpus()
llm = cached_llm()
rag_chain = cached_rag_chain(llm, top_k, use_mmr)
 
with st.container():

    query = st.chat_input(
        "Ask a question",
        key="query_input"
    )

    if query and query.strip():
        with st.spinner("Thinking — retrieving context and generating answer..."):
            result = rag_chain.invoke({
                "input": query,
                "top_k": top_k
            })

        answer = result.get("answer") or result.get("result") or result

        left, right = st.columns([3, 1])

        with left:
            st.subheader("Answer")
            st.write(answer)

            st.download_button("Download answer", answer, file_name="answer.txt")

            meta_info = []
            if isinstance(result, dict):
                meta_info.append(f"Sources: {len(result.get('source_documents', []) or result.get('context') or [])}")

        with right:
            st.subheader("Context Documents")

            context_docs = (
                result.get("context")
                or result.get("source_documents")
                or []
            )

            if not context_docs:
                st.write("No documents returned.")
            else:
                for i, doc in enumerate(context_docs, 1):
                    meta = getattr(doc, 'metadata', {}) or {}
                    src = meta.get('source') or meta.get('url') or f"Document {i}"
                    content = getattr(doc, 'page_content', str(doc))
                    
                    with st.expander(f"[{i}/{len(context_docs)}] {src}"):
                        st.markdown(f"**Source:** {src}")
                        if meta.get('start_index') is not None:
                            st.markdown(f"**Index:** {meta.get('start_index')}")
                        preview = content[:800].replace("\n", " ")
                        st.write(preview)
                        st.download_button(f"Download doc {i}", preview, file_name=f"doc_{i}.txt")

