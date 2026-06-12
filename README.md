#  RAG Assistant

A PDF Question-Answering Chatbot built using **Retrieval Augmented Generation (RAG)**.

Upload any PDF document and ask questions in natural language. The assistant retrieves relevant information from the uploaded document and generates answers strictly based on the document content.

---

#  Features

* Upload and process PDF documents
* Intelligent document chunking
* Vector-based semantic search using embeddings
* Context-aware question answering
* FastAPI backend
* Interactive web-based chat interface
* LangChain-powered RAG pipeline
* Groq LLM integration for fast inference
* Automatic document replacement when a new PDF is uploaded
* Chat reset functionality

---

#  Technology Stack

### Backend

* Python
* FastAPI
* Uvicorn

### RAG Components

* LangChain
* LangGraph
* FAISS Vector Store
* HuggingFace Embeddings
* Sentence Transformers

### LLM

* Groq API
* Llama 3 Model

### Frontend

* HTML
* CSS
* JavaScript

---

#  Prerequisites

Before running the project, ensure you have:

* Python 3.10 or above
* pip (Python Package Manager)
* Groq API Key
* HuggingFace Access Token

### Get API Credentials

**Groq API Key**
https://console.groq.com

**HuggingFace Token**
https://huggingface.co/settings/tokens

---

#  Installation

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd RAG-Assistant
```

---

## Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

---

## Step 3: Install Dependencies

### Option 1: Install Individually

```bash
pip install fastapi
pip install uvicorn
pip install python-multipart
pip install langchain
pip install langchain-community
pip install langchain-core
pip install langchain-groq
pip install langchain-huggingface
pip install langchain-text-splitters
pip install langgraph
pip install pypdf
pip install sentence-transformers
pip install huggingface-hub
pip install python-dotenv
pip install groq
```

### Option 2: Install from Requirements File

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create a `.env` file in the project root directory and add the following configuration:

```env
GROQ_API_KEY=your_groq_api_key_here

GROQ_MODEL=llama3-8b-8192

RAG_PROMPT_TEMPLATE=Use the following context to answer the question.

Context: {context}

Question: {question}

HF_TOKEN=your_huggingface_token_here
```

---

#  Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Successful startup output:

```text
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

---

#  Access the Application

Open your browser and navigate to:

```text
http://127.0.0.1:8000
```

---

#  How to Use

### Upload a PDF

1. Click **Choose PDF**
2. Select a PDF file from your system
3. Click **Upload PDF**
4. Wait for the upload confirmation

### Ask Questions

1. Type a question in the chat input field
2. Press **Enter** or click **Send**
3. Receive answers generated from the uploaded document

### Switch Documents

* Upload a new PDF
* Previous document embeddings are automatically cleared
* New document becomes the active knowledge source

### Start a New Chat

* Click **New Chat**
* Chat history will be cleared

---

#  RAG Workflow

1. User uploads a PDF
2. PDF content is extracted
3. Text is split into chunks
4. Embeddings are generated using HuggingFace models
5. Chunks are stored in a FAISS vector database
6. User submits a question
7. Relevant chunks are retrieved
8. Context is sent to the Groq LLM
9. Response is generated based on retrieved document content
10. Answer is displayed in the chat interface

---

# Project Structure

```text
RAG-Assistant/
│
├── main.py                 # FastAPI application and API routes
├── rag_engine.py           # PDF processing, embeddings, vector store, RAG pipeline
├── agent_tools.py          # LangGraph agent tools and utilities
│
├── frontend/
│   └── index.html          # User Interface
│
├── uploads/                # Uploaded PDF files (auto-created)
│
├── .env                    # Environment variables (Do not share)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

#  Security Notes

* Never commit the `.env` file to GitHub.
* Keep your API keys private.
* Add `.env` to `.gitignore`.

Example:

```gitignore
.env
uploads/
__pycache__/
venv/
```

---

#  Future Enhancements

* Multi-PDF support
* Persistent vector database
* Source citation display
* Chat history storage
* User authentication
* Streaming responses
* Advanced document retrieval techniques
* Hybrid search (Keyword + Semantic Search)
