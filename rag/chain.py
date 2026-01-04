from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
import textwrap
from rag.retriever import build_retriever

# The stuff-documents chain injects retrieved docs into {context} and formats the full
# TinyLlama prompt before calling the LLM. The retrieval chain then couples this with the
# retriever to produce a complete RAG pipeline (query → retrieve → format prompt → generate).
def build_rag_chain(llm, k, use_mmr=True, use_bm25=True, use_rerank=True):
    SYSTEM = (
        "You are a summariser. Answer using only the provided context." \
        "Do not repeat yourself, or the question, in your answer." \
        "Keep your sentences short, and your answer concise and complete."
    )

    TINYLLAMA_PROMPT = textwrap.dedent("""\
    <|system|>
    {system}</s>

    <|user|>
    <context>
    Provided context (might be cut-off/incomplete): {context}
    </context>

    Question: {input}</s>
    <|assistant|>
    """)

    prompt = ChatPromptTemplate.from_template(TINYLLAMA_PROMPT).partial(system=SYSTEM)
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = build_retriever(k=k, use_mmr=use_mmr, use_bm25=use_bm25, use_rerank=use_rerank)

    return create_retrieval_chain(retriever, doc_chain)
