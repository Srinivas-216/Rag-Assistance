import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Module-level store — replaced (not appended) on each new upload
vector_store: InMemoryVectorStore = InMemoryVectorStore(embedding=embeddings)
_current_pdf: str | None = None


def load_pdf(file_path: str) -> int:
    global vector_store, _current_pdf

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # ✅ FIX: Replace the store completely instead of appending to it.
    # This ensures old PDF chunks are never mixed with the new PDF.
    vector_store = InMemoryVectorStore(embedding=embeddings)
    _current_pdf = os.path.basename(file_path)

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    for page in pages:
        page.metadata["source"] = _current_pdf

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    vector_store.add_documents(chunks)
    return len(chunks)


def get_current_pdf() -> str | None:
    """Returns the filename of the currently loaded PDF, or None."""
    return _current_pdf


def get_rag_chain():
    if _current_pdf is None:
        raise ValueError("No PDF has been uploaded yet.")

    # ✅ FIX: Retriever is built from the current (fresh) vector_store,
    # not a stale reference captured at module load time.
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt_template = os.getenv("RAG_PROMPT_TEMPLATE")
    if not prompt_template:
        raise ValueError("RAG_PROMPT_TEMPLATE is not set in the .env file")

    groq_model = os.getenv("GROQ_MODEL")
    if not groq_model:
        raise ValueError("GROQ_MODEL is not set in the .env file")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in the .env file")

    RAG_PROMPT = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatGroq(
        model=groq_model,
        max_tokens=512,
        api_key=groq_api_key,
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', '?')} | "
            f"Page: {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )