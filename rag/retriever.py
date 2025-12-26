from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from utils.config import EMBED_MODEL, DB_DIR

# Build a retriever with contextual compression
# k: number of documents to retrieve
def build_retriever(k=3):

    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embed)

    base_retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # `similarity_threshold`: cosine-similarity cutoff used to filter retrieved documents.
    # Scale: cosine similarity ranges from -1.0 (opposite) to 1.0 (identical); higher = more similar.
    # Trade-off: higher thresholds reduce noise (lower recall), lower thresholds increase recall but add irrelevant documents.
    emb_filter = EmbeddingsFilter(
        embeddings=embed,
        similarity_threshold=0.4
    )

    return ContextualCompressionRetriever(
        base_compressor=emb_filter,
        base_retriever=base_retriever
    )
