from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from utils.config import EMBED_MODEL, DB_DIR


def build_retriever(k=3):

    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embed)

    base_retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    emb_filter = EmbeddingsFilter(
        embeddings=embed,
        similarity_threshold=0.5
    )

    return ContextualCompressionRetriever(
        base_compressor=emb_filter,
        base_retriever=base_retriever
    )
