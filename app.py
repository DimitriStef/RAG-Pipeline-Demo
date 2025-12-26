import streamlit as st
from rag.ingest import run_ingestion
from rag.llm import load_llm
from rag.chain import build_rag_chain
from rag.retriever import build_retriever


st.title("Custom RAG Demo")

# Cache resources to avoid reloading on every interaction
@st.cache_resource
def cached_corpus():
    run_ingestion("data/urls.txt")
    return True

@st.cache_resource
def cached_retriever():
    cached_corpus()
    return build_retriever()

@st.cache_resource
def cached_llm():
    return load_llm()

@st.cache_resource
def cached_rag_chain():
    llm = cached_llm()
    retriever = cached_retriever()
    return build_rag_chain(llm, retriever)

# Build the RAG chain
rag_chain = cached_rag_chain()

query = st.text_input("Ask a question:")
if query:
    result = rag_chain.invoke({"input": query})

    st.subheader("Answer")
    st.write(result.get("answer") or result.get("result") or result)

    context_docs = (
        result.get("context")
        or result.get("source_documents")
        or []
    )

    st.subheader("Context Documents")

    if not context_docs:
        st.write("No documents returned.")
    else:
        for i, doc in enumerate(context_docs, 1):
            meta = doc.metadata

            with st.expander(f"Document {i}"):
                st.markdown(f"**Source:** {meta.get('source')}")
                st.markdown(f"**Index:** {meta.get('start_index')}")

                preview = doc.page_content[:300].replace("\n", " ") + " ..."
                st.markdown("**Content Preview:**")
                st.write(preview)
