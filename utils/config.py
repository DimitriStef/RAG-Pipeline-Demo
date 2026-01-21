# Paths and model configurations
DATA_DIR = "data"
DB_DIR = "data/chroma"
CORPUS_DIR = "data/corpus"

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOPIC_MODEL = "facebook/genre-kilt"

# Prompts
CHATBOT_SYSTEM = "You are a summariser. Answer using only the provided context. " \
                "Do not repeat yourself, or the question, in your answer. " \
                "Keep your sentences short, and your answer concise and complete, in nice paragraphs."

CHATBOT_PROMPT = """ \
    <|system|>
    {system}</s>

    <|user|>
    <context>
    Provided context (might be cut-off/incomplete): {context}
    </context>

    Question: {input}</s>
    <|assistant|>
    """