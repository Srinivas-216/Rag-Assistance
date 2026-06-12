RAG Assistant

A PDF question-answering chatbot. Upload any PDF and ask questions — answers come strictly from your document using Retrieval Augmented Generation (RAG).


#Requirements


Python 3.10 or above
pip (comes with Python)
A Groq API Key — https://console.groq.com
A HuggingFace Token — https://huggingface.co/settings/tokens



Step 1 — Create a Virtual Environment

Open a terminal inside the project folder and run:

bashpython -m venv venv

Activate it:

bash# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate


Step 2 — Install Dependencies

bashpip install fastapi
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

Or install everything at once:

bashpip install -r requirements.txt


Step 3 — Create the .env File

Create a file named .env in the root project folder and add:

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
RAG_PROMPT_TEMPLATE=Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}
HF_TOKEN=your_huggingface_token_here


Step 4 — Run the Server

bashuvicorn main:app --reload

You should see:

INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.


Step 5 — Open the App

Open your browser and go to:

http://127.0.0.1:8000


Step 6 — Using the App


Click Choose PDF in the left sidebar
Select a PDF from your computer
Click Upload PDF and wait for the confirmation message
Type your question in the input bar and press Send or Enter
To switch documents, upload a new PDF — old data clears automatically
Click New Chat to clear the chat area



#Project Structure

Backend/
├── main.py               # FastAPI routes
├── rag_engine.py         # PDF loading, chunking, embeddings, RAG chain
├── agent_tools.py        # LangGraph agent tools
├── frontend/
│   └── index.html        # Chat UI
├── uploads/              # Auto created when a PDF is uploaded
├── .env                  # Your API keys — do not share this file
├── requirements.txt      # Python dependencies
└── README.md
