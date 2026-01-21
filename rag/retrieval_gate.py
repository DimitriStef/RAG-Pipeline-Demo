from rag.topic_parser import generate_wikipedia_entities
from transformers import pipeline


def to_context_text(ctx) -> str:
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


def is_context_sufficient(question: str, context: str, thr: float = 0.5) -> bool:

    qa = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2", handle_impossible_answer=True)

    res = qa(question=question, context=context)
    score = float(res.get("score", 0.0))
    # Weird behavior: lower scores are given to correct context
    return (score < thr)


def parse_gate_decision(query, context):

    context_string = to_context_text(context)
    sufficient = is_context_sufficient(query, context=context_string)
    
    if not sufficient:
        topics = generate_wikipedia_entities(query)
        print(f"Identified topics for web crawling: {topics}")
        return False
        # TODO: if context is insufficient, trigger topic parser and webcrawler 


    return True