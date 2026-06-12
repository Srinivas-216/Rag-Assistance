import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from rag_engine import load_pdf, get_rag_chain, get_current_pdf
from agent_tools import search_documents, list_sources, summarize_topic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

class QueryRequest(BaseModel):
    question: str

class AgentRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "running", "current_pdf": get_current_pdf()}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # ✅ FIX: Validate that the uploaded file is actually a PDF
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are supported."}
        )

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # ✅ FIX: Delete ALL previously uploaded files before saving the new one
        # so the uploads folder never accumulates stale PDFs
        for old_file in os.listdir(UPLOAD_DIR):
            old_path = os.path.join(UPLOAD_DIR, old_file)
            if os.path.isfile(old_path):
                os.remove(old_path)

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # load_pdf now resets the vector store before indexing
        chunks = load_pdf(file_path)

        return {
            "message": f"'{file.filename}' uploaded and indexed successfully.",
            "chunks": chunks,
            "filename": file.filename,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask")
async def ask_question(request: QueryRequest):
    # ✅ FIX: Return a clear error if no PDF has been uploaded yet
    if get_current_pdf() is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No PDF uploaded. Please upload a PDF before asking questions."}
        )

    try:
        chain = get_rag_chain()
        answer = chain.invoke(request.question)
        return {"answer": answer, "source": get_current_pdf()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/agent")
async def run_agent(request: AgentRequest):
    if get_current_pdf() is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No PDF uploaded. Please upload a PDF before using the agent."}
        )

    try:
        tools = [search_documents, list_sources, summarize_topic]
        llm = ChatGroq(model=os.getenv("GROQ_MODEL"), max_tokens=1024)
        agent = create_react_agent(llm, tools)
        result = agent.invoke({"messages": [{"role": "user", "content": request.question}]})
        return {"answer": result["messages"][-1].content}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})