from utils.crawler import crawl_from_txt
from utils.config import DB_DIR, EMBED_MODEL

from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def chunk_fragments(fragments):
    splitter_base = dict(
        max_chunk_size=600,
        separators=["\n\n", "\n", ". ", "! ", "? "],
        elements_to_preserve=["table", "ul", "ol", "pre", "code"],
        denylist_tags=["script", "style", "head", "sup"],
        stopword_removal=False,
        normalize_text=False,
        headers_to_split_on=[],
        keep_separator="end",
    )

    splitter_section = HTMLSemanticPreservingSplitter(chunk_overlap=0, **splitter_base)
    splitter_subsection = HTMLSemanticPreservingSplitter(chunk_overlap=60, **splitter_base)

    chunks = []
    for frag in fragments:
        has_subsection = bool(frag.metadata.get("has_subsection", False))
        splitter = splitter_subsection if has_subsection else splitter_section

        out_docs = splitter.split_text(frag.page_content)

        upstream_meta = dict(frag.metadata)
        for d in out_docs:
            d.metadata.update(upstream_meta)

        chunks.extend(out_docs)

    return chunks


def run_ingestion():
    fragments = crawl_from_txt()

    if not fragments:
        print("No new documents to ingest.")
        return

    chunks = chunk_fragments(fragments)
    if not chunks:
        print("No chunks produced from extracted fragments.")
        return

    embed = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embed,
        persist_directory=DB_DIR
    )
