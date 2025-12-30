from pathlib import Path
import re
import json
import hashlib
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from utils.config import CORPUS_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / CORPUS_DIR
CORPUS_PATH.mkdir(parents=True, exist_ok=True)

CITATION_RE = re.compile(r"\[\d+\]")


def extract_wikipedia_content(soup: BeautifulSoup) -> dict:
    """Extract title and main paragraph content from a Wikipedia page."""
    title_tag = soup.select_one("h1#firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

    content_div = soup.select_one("div#mw-content-text")
    if not content_div:
        return {"title": title, "content": ""}

    # Remove elements that often pollute the paragraph text
    for tag in content_div.select(
        "script, style, sup, .navbox, .infobox, .sidebar, .metadata"
    ):
        tag.decompose()

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in content_div.find_all("p")
        if p.get_text(strip=True)
    ]

    content = "\n\n".join(paragraphs)
    content = CITATION_RE.sub("", content)
    content = re.sub(r"[ \t]+", " ", content).strip()  # normalize intra-line spaces

    return {"title": title, "content": content}


def _url_to_cache_path(url: str) -> Path:
    """Generate cache file path from URL hash."""
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CORPUS_PATH / f"{h}.json"


def _cache_is_valid(cache_path: Path, url: str) -> bool:
    """
    Return True if cache exists and is valid JSON.
    If corrupted, delete it and return False (so caller refetches).
    """
    if not cache_path.exists():
        return False

    try:
        json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Loaded cached: {url}")
        return True
    except Exception as e:
        print(f"Cache corrupted for {url} ({e}), deleting and refetching.")
        cache_path.unlink(missing_ok=True)
        return False


def load_from_url(url: str) -> list[Document]:
    """Load Wikipedia article from URL with caching."""
    cache_path = _url_to_cache_path(url)

    # If document is cached, return empty list to avoid duplicates.
    if _cache_is_valid(cache_path, url):
        return []

    # Fetch and parse
    try:
        print(f"Fetching: {url}")
        loader = WebBaseLoader(url)
        soup = loader.scrape()

        extracted = extract_wikipedia_content(soup)
        if not extracted["content"]:
            print(f"Warning: No content extracted from {url}")
            return []

        doc = Document(
            page_content=extracted["content"],
            metadata={"source": url, "title": extracted["title"]},
        )

        cache_path.write_text(
            json.dumps(
                {"content": doc.page_content, "metadata": doc.metadata},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved to cache: {url}")
        return [doc]

    except Exception as e:
        print(f"Error loading {url}: {e}")
        return []


def crawl_from_txt(file_path: str) -> list[Document]:
    """Load multiple Wikipedia articles from a text file of URLs."""
    p = Path(file_path)
    path = p if p.is_absolute() else (BASE_DIR / p)
    print(f"Reading URLs from {path}...")

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)

    docs: list[Document] = []
    for url in urls:
        docs.extend(load_from_url(url))

    print(f"Loaded {len(docs)} documents from {len(urls)} URLs.")
    return docs
