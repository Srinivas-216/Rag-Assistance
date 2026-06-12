import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.tools import tool
from rag_engine import get_rag_chain, vector_store


@tool
def search_documents(query: str) -> str:
    """Search the uploaded PDF documents for relevant information."""
    chain = get_rag_chain()
    return chain.invoke(query)


@tool
def list_sources(query: str) -> str:
    """List the source documents available in the vector store."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(query)
    sources = list(set(
        f"{d.metadata.get('source')} - Page {d.metadata.get('page')}"
        for d in docs
    ))
    return "\n".join(sources) if sources else "No documents loaded yet."


@tool
def summarize_topic(topic: str) -> str:
    """Summarize everything known about a specific topic from the documents."""
    chain = get_rag_chain()
    return chain.invoke(f"Give a detailed summary about: {topic}")