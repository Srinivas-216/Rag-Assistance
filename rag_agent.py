import os
import sys
import time
import threading
from dotenv import load_dotenv
load_dotenv()
required_keys = ["GROQ_API_KEY", "GROQ_MODEL", "RAG_PROMPT_TEMPLATE"]
for key in required_keys:
    if not os.getenv(key):
        print(f"Missing required environment variable: {key}")
        sys.exit(1)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class ThinkingSpinner:
    FRAMES = ["-", "\\", "|", "/"]

    def __init__(self, message="Thinking"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        while not self._stop_event.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            print(f"\r{frame}  {self.message}...", end="", flush=True)
            time.sleep(0.08)
            i += 1

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        print("\r" + " " * 40 + "\r", end="", flush=True)


def stream_print(text, delay=0.02):
    words = text.split(" ")
    for i, word in enumerate(words):
        print(word, end="", flush=True)
        if i < len(words) - 1:
            print(" ", end="", flush=True)
        time.sleep(delay)
    print()


PDF_PATHS = [
    "data/AGL_V_BonfireBlaze.pdf",
    "data/AGL_V_WesternBlaze.pdf",
]

all_docs = []
for path in PDF_PATHS:
    if not os.path.exists(path):
        print(f"File not found: {path} — skipping")
        continue
    loader = PyPDFLoader(path)
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = os.path.basename(path)
    all_docs.extend(pages)

if not all_docs:
    print("No documents loaded. Place the PDFs inside the data/ folder.")
    sys.exit(1)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

model_name = os.getenv("GROQ_MODEL")
prompt_template = os.getenv("RAG_PROMPT_TEMPLATE")

RAG_PROMPT = ChatPromptTemplate.from_template(prompt_template)

llm = ChatGroq(
    model=model_name,
    max_tokens=512,
)

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source','?')} | Page: {d.metadata.get('page','?')}]\n"
        f"{d.page_content}"
        for d in docs
    )

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

spinner = ThinkingSpinner()

print("\nAmerican Gas Log Assistant - Ready!")
print("Type your question below. Type 'quit' to exit.")
print("-" * 60)

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if user_input.lower() in ("quit", "exit", "q", ""):
        print("Goodbye!")
        break

    spinner.start()
    answer = rag_chain.invoke(user_input)
    spinner.stop()

    print("Agent: ", end="")
    stream_print(answer, delay=0.02)