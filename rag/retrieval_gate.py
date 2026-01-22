from pathlib import Path
from rag.topic_parser import generate_wikipedia_entities
from transformers import pipeline
from utils.config import DATA_DIR

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / DATA_DIR
DATA_PATH.mkdir(parents=True, exist_ok=True)


qa = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2",
        handle_impossible_answer=True,
    )


def to_context_text(ctx):
    if not ctx:
        return ""

    # Single Document-like
    if hasattr(ctx, "page_content"):
        return ctx.page_content

    # List/tuple of Document-like
    if isinstance(ctx, (list, tuple)):
        return " ".join(
            (d.page_content or "")
            for d in ctx
            if d is not None and hasattr(d, "page_content") and (d.page_content)
        )

    # Fallback
    return str(ctx)


def is_context_sufficient(question, context, margin = 0.01):
    r = qa(question=question, context=context, top_k=2)

    # Expect list[dict] for top_k
    if not r or not isinstance(r, list):
        return False

    best = r[0]
    best_ans = (best.get("answer") or "").strip()
    if best_ans == "":
        return False  # model prefers "no answer"

    # If we didn't actually get 2 candidates, accept (non-empty best)
    if len(r) < 2:
        return True

    second = r[1]
    second_ans = (second.get("answer") or "").strip()

    # If runner-up is empty, accept
    if second_ans == "":
        return True

    s0 = float(best.get("score", 0.0))
    s1 = float(second.get("score", 0.0))
    return (s0 - s1) >= margin


def write_urls(urls):
    output = DATA_PATH / "urls.txt"
    output.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if output.exists():
        existing = set(output.read_text(encoding="utf-8").splitlines())

    new_lines = [link for link in urls if link not in existing]
    if new_lines:
        with output.open("a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(new_lines) + "\n")


def parse_gate_decision(query, context):

    context_string = to_context_text(context)
    sufficient = is_context_sufficient(query, context=context_string)
    
    if not sufficient:
        topics = generate_wikipedia_entities(query)
        print(f"Identified topics for web crawling: {topics}")

        write_urls(topics)
                
        return False

    return True