import re
import unicodedata
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader, UnstructuredURLLoader, PyPDFLoader
from langchain_core.documents import Document
import requests
import hashlib
import json
from utils.config import CORPUS_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / CORPUS_DIR
CORPUS_DIR.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    patterns_html = [
        r"<[^>]+>",
        r"(?i)javascript:void.*?;",
        r"(?i)cookies? settings.*",
    ]
    for p in patterns_html:
        text = re.sub(p, " ", text)

    patterns_boilerplate = [
        r"© ?\d{4}.*",
        r"All rights reserved.*",
        r"Privacy Policy.*",
        r"Terms of Use.*",
        r"Home > .*",
        r"^\s*(Share|Tweet|Print)\s*$",
    ]
    for p in patterns_boilerplate:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)

    def is_symbol_line(line: str) -> bool:
        non_alnum = sum(1 for c in line if not c.isalnum())
        return non_alnum > len(line) * 0.7

    text = "\n".join([l for l in text.splitlines() if not is_symbol_line(l)])

    def informative(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if len(s) < 15:
            return any(c.isalpha() for c in s)
        return True

    text = "\n".join([l for l in text.splitlines() if informative(l)])
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _url_to_cache_path(url: str) -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CORPUS_DIR / f"{h}.json"


def load_from_url(url):
    cache_path = _url_to_cache_path(url)

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_list = json.load(f)

            print(f"Loaded cached: {url}")

            restored_docs = []
            for item in cached_list:
                meta = item.get("metadata") or {"source": url}
                doc = Document(page_content=item.get("text", ""), metadata=meta)
                restored_docs.append(doc)

            return restored_docs

        except Exception:
            print(f"Cache corrupted for {url}, re-downloading...")

    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        content_type = response.headers.get("content-type", "")

        if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
            loader = PyPDFLoader(url)
        else:
            try:
                loader = WebBaseLoader(url)
            except Exception:
                loader = UnstructuredURLLoader(urls=[url])

        docs = loader.load()

        # Clean text & apply metadata for langchain
        cleaned_cache = []
        for d in docs:
            d.page_content = clean_text(d.page_content)
            d.metadata.setdefault("source", url)

            cleaned_cache.append({
                "text": d.page_content,
                "metadata": d.metadata,
            })

        # Save to cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_cache, f, ensure_ascii=False, indent=2)

        print(f"Saved to cache: {url}")
        return docs

    except Exception as e:
        print(f"Error loading {url}: {e}")
        return []


def crawl_from_txt(file_path):
    path = BASE_DIR / file_path
    print(f"Reading URLs from {path}...")

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    with open(path, encoding="utf-8") as f:
        urls = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]

    docs = []
    for url in urls:
        print(f"Fetching: {url}")
        docs.extend(load_from_url(url))

    print(f"Loaded {len(docs)} documents from {len(urls)} URLs.")
    return docs
