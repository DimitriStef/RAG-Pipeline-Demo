from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
import textwrap

# The stuff-documents chain injects retrieved docs into {context} and formats the full
# TinyLlama prompt before calling the LLM. The retrieval chain then couples this with the
# retriever to produce a complete RAG pipeline (query → retrieve → format prompt → generate).
def build_rag_chain(llm, retriever):
    # Prompt engineering based on TinyLlama's recommended prompt, from Ollama's website
    SYSTEM = "You are a helpful AI assistant that summarises information."

    TINYLLAMA_PROMPT = textwrap.dedent("""<|system|>
    {system}</s>
    <|user|>
    {input}

    Context:
    {context}

    Instructions:
    Answer in a few sentences. Answer only based on the context. 
    Do not repeat the question or yourself. Do not include your thought process.</s>
    <|assistant|>
    """)


    prompt = ChatPromptTemplate.from_template(TINYLLAMA_PROMPT).partial(system=SYSTEM)

    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    return create_retrieval_chain(retriever, doc_chain)
