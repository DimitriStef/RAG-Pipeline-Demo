from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from utils.config import MODEL_NAME


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda",
        dtype="auto"
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=False, # deterministic output
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.06,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=gen)
