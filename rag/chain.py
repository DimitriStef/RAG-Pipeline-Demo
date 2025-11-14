from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

def build_rag_chain(llm, retriever):
    SYSTEM = "You are a summariser, not a chatbot."

    TINYLLAMA_PROMPT = """<|system|>
        {system}</s>
        <|user|>
        ### Question
        {input}

        ### Context
        {context}

        ### Instructions
        Answer in a few sentences. Do not repeat the question.
        You do not interact beyond answering the question based on the provided context.</s>
        <|assistant|>
        """

    prompt = ChatPromptTemplate.from_template(TINYLLAMA_PROMPT).partial(system=SYSTEM)

    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    return create_retrieval_chain(retriever, doc_chain)
