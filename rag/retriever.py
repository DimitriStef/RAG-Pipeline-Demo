from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from utils.config import EMBED_MODEL, DB_DIR, RERANK_MODEL


def _base_retriever(vectordb, k, use_mmr):
    if use_mmr:
        # MMR: trades relevance for diversity; reduces near-duplicate chunks
        return vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 2, "lambda_mult": 0.7},
        )
    # Threshold mode: higher cutoff = cleaner, but can miss borderline-relevant chunks
    return vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": 0.75},
    )


def _bm25_retriever_from_chroma(vectordb, k):
    # Pull stored texts+metadata from Chroma to build BM25 index.
    raw = vectordb.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(raw["documents"], raw["metadatas"])
    ]
    return BM25Retriever.from_documents(docs, k=k)


def build_retriever(k = 6, use_mmr = True, use_bm25 = True, use_rerank = True):
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embed)

    base = _base_retriever(vectordb, k=k, use_mmr=use_mmr)
    retriever = base

    if use_bm25:
        bm25 = _bm25_retriever_from_chroma(vectordb, k=k)
        # Hybrid: BM25 stabilizes keyword hits; dense handles paraphrase/semantic drift
        retriever = EnsembleRetriever(retrievers=[bm25, base], weights=[0.3, 0.7])

    if use_rerank:
        reranker = HuggingFaceCrossEncoder(model_name=RERANK_MODEL, model_kwargs={"device": "cpu"})
        compressor = CrossEncoderReranker(model=reranker, top_n=k)

        retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=compressor,
        )

    return retriever
