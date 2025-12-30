from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from utils.config import EMBED_MODEL, DB_DIR

# Build a retriever with contextual compression
# k: number of documents to retrieve
def build_retriever(k=3, use_mmr=True):
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embed)
    
    if use_mmr:
        # MMR (Maximal Marginal Relevance) retriever.
        # `fetch_k`: initial candidate pool size for diversity selection (larger = more diverse).
        # `lambda_mult`: diversity-relevance trade-off (0.0 = max diversity, 1.0 = max relevance).
        # Trade-off: higher lambda prioritizes relevance but may return redundant docs
        return vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 4,
                "lambda_mult": 0.8
            }
        )
    else:
        # `score_threshold`: cosine-similarity cutoff used to filter retrieved documents.
        # Trade-off: higher thresholds reduce noise (lower recall), lower thresholds increase recall but add irrelevant documents.
        return vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": 0.5
            }
        )

