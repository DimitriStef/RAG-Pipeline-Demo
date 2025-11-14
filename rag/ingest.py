from utils.crawler import crawl_from_txt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from utils.config import DB_DIR, EMBED_MODEL

def run_ingestion(txt_path: str):
    docs = crawl_from_txt(txt_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        add_start_index=True
    )

    chunks = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed,
        persist_directory=DB_DIR
    )

    return vectordb
