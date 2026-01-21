from rag.topic_parser import generate_wikipedia_entities
from transformers import pipeline


def to_context_text(ctx, max_chars: int = 8000) -> str:
    """Coerce ctx (str | Document | list[Document] | dict) into a single string."""
    if ctx is None:
        return ""

    # If a dict was passed, try to pull docs out of it
    if isinstance(ctx, dict):
        ctx = ctx.get("context") or ctx.get("source_documents") or ctx.get("documents") or ""

    # Already a string
    if isinstance(ctx, str):
        return ctx

    # Single Document-like
    if hasattr(ctx, "page_content"):
        return str(ctx.page_content or "")

    # List/tuple of Document-like or strings
    if isinstance(ctx, (list, tuple)):
        parts = []
        total = 0
        for item in ctx:
            if item is None:
                continue
            if isinstance(item, str):
                text = item
            elif hasattr(item, "page_content"):
                text = item.page_content
            else:
                text = str(item)

            if not text:
                continue

            # simple cap to reduce truncation
            if total + len(text) > max_chars:
                text = text[: max(0, max_chars - total)]
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return "\n\n".join(parts)

    # Fallback
    return str(ctx)


def is_context_sufficient(question: str, context: str, thr: float = 0.5) -> bool:

    qa = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2", handle_impossible_answer=True)

    res = qa(question=question, context=context)
    score = float(res.get("score", 0.0))
    # Weird behavior: lower scores are given to correct context
    return (score < thr)


def parse_gate_decision(query, context):

    context_string = to_context_text(context, max_chars=8000)
    sufficient = is_context_sufficient(query, context=context_string)
    
    if not sufficient:
        topics = generate_wikipedia_entities(query)
        print(f"Identified topics for web crawling: {topics}")
        return False
        # TODO: if context is insufficient, trigger topic parser and webcrawler 


    return True