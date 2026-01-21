from pathlib import Path
import json
import hashlib

from bs4 import Tag
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader

from utils.config import DATA_DIR, CORPUS_DIR

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / DATA_DIR
DATA_PATH.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = BASE_DIR / CORPUS_DIR
CORPUS_PATH.mkdir(parents=True, exist_ok=True)

# Wikipedia-specific noise (templates / navigation / references scaffolding)
NOISE_SELECTORS = (
    "script",
    "style",
    "sup.reference",
    ".mw-editsection",
    ".navbox",
    ".infobox",
    ".sidebar",
    ".metadata",
    "#toc",
    ".toc",
    ".mw-references-wrap",
    ".reflist",
    ".catlinks",
    ".hatnote",
    ".mainarticle",
    ".dablink",
    ".rellink",
)


def extract_wikipedia_fragments(soup):
    title_tag = soup.select_one("h1#firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

    root = soup.select_one("div#mw-content-text div.mw-parser-output") or soup.select_one("div#mw-content-text")
    if not root:
        return title, []

    for sel in NOISE_SELECTORS:
        for tag in root.select(sel):
            tag.decompose()

    fragments = []
    section = "Lead"
    subsection = None
    current = []

    for node in root.children:
        if not isinstance(node, Tag):
            continue

        if node.name == "h2" or node.name == "h3":
            if current:
                body_html = "".join(str(x) for x in current).strip()
                if body_html:
                    fragments.append((body_html, section, subsection))
                current = []

            header_text = node.get_text(" ", strip=True)
            if node.name == "h2":
                section = header_text or "Untitled section"
                subsection = None
            else:
                subsection = header_text or "Untitled subsection"

            continue

        current.append(node)

    if current:
        body_html = "".join(str(x) for x in current).strip()
        if body_html:
            fragments.append((body_html, section, subsection))

    return title, fragments


def _url_to_cache_path(url) -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CORPUS_PATH / f"{h}.json"


def _cache_is_valid(cache_path, url) -> bool:
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


def load_from_url(url) -> list[Document]:
    cache_path = _url_to_cache_path(url)

    if _cache_is_valid(cache_path, url):
        return []

    try:
        print(f"Fetching: {url}")
        soup = WebBaseLoader(url).scrape()

        title, fragments = extract_wikipedia_fragments(soup)
        if not fragments:
            print(f"Warning: No content extracted from {url}")
            return []

        docs: list[Document] = []
        for i, (html, section, subsection) in enumerate(fragments):
            docs.append(
                Document(
                    page_content=html,
                    metadata={
                        "source": url,
                        "title": title,
                        "section": section,
                        "subsection": subsection,
                        "has_subsection": bool(subsection),
                        "fragment_index": i,
                    },
                )
            )

        cache_path.write_text(
            json.dumps(
                [{"content": d.page_content, "metadata": d.metadata} for d in docs],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved to cache: {url}")
        return docs

    except Exception as e:
        print(f"Error loading {url}: {e}")
        return []


def crawl_from_txt():
    path = DATA_PATH / "urls.txt"
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

    print(f"Loaded {len(docs)} extracted fragments from {len(urls)} URLs.")
    return docs