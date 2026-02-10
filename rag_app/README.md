🧠 RAG App – Local Embeddings with ChromaDB

A simple Retrieval-Augmented Generation (RAG) application built using local embeddings, a persistent vector database, and GitHub Models (Azure Inference) for answer generation.

This project demonstrates how to build a cost-efficient, offline-friendly RAG system without relying on paid embedding APIs.

🚀 Features

📚 Retrieval-Augmented Generation (RAG)

🔎 Semantic search using vector similarity

🧠 Local embeddings with SentenceTransformers (no OpenAI embeddings)

🗄 Persistent vector storage using ChromaDB

🌐 Web page ingestion via urls.txt

📄 Text file ingestion from local documents

🧩 Chunking with overlap for better context retrieval

🎯 Similarity thresholding to reduce hallucinations

📌 Source attribution for every answer

🧪 Clean CLI workflow (--index mode)

🏗 Architecture Overview
Documents / URLs
        ↓
Text Cleaning (HTML → Plain Text)
        ↓
Chunking (with overlap)
        ↓
Local Embeddings (SentenceTransformers)
        ↓
ChromaDB (Persistent Vector Store)
        ↓
Semantic Search
        ↓
LLM Answer Generation (GitHub Models)

📁 Project Structure
rag_app/
├── app.py                 # Main RAG application
├── requirements.txt       # Python dependencies
├── chroma_db/             # Persistent vector database
├── documents/
│   ├── urls.txt           # List of URLs to index
│   └── *.txt              # Optional local text files
└── .env                   # Environment variables

⚙️ Prerequisites

Python 3.10+

GitHub account with GitHub Models access

Internet access (for URL ingestion)

🔐 Environment Variables

Create a .env file in the project root:

GITHUB_TOKEN=your_github_models_token_here


This token is used to call GitHub Models (Azure Inference) for chat generation.

📦 Installation
1️⃣ Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

2️⃣ Upgrade pip
python -m pip install --upgrade pip

3️⃣ Install dependencies
python -m pip install -r requirements.txt

4️⃣ Install PyTorch (CPU-only)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

📄 Adding Documents
➤ Local Text Files

Place .txt files inside:

documents/


Each file will be indexed automatically.

➤ Web URLs (Recommended)

Create a file:

documents/urls.txt


Example:

# RAG fundamentals
https://en.wikipedia.org/wiki/Retrieval-augmented_generation
https://docs.trychroma.com/

# Vector databases
https://www.pinecone.io/learn/vector-database/


Rules:

One URL per line

Blank lines are allowed

Lines starting with # are ignored

🧱 Indexing the Data

Before asking questions, you must index documents:

python app.py --index


This will:

Load documents & URLs

Chunk the text

Generate embeddings locally

Store vectors in ChromaDB

❓ Asking Questions

After indexing:

python app.py


You’ll see:

RAG App – Semantic Search (local embeddings)
Type 'quit' to exit
------------------------------------------------------------
Question:


Example:

Question: What is retrieval augmented generation?

🧠 How It Avoids Hallucinations

Answers are generated only from retrieved context

A similarity threshold (MIN_SIMILARITY = 0.40) filters weak matches

If context is insufficient, the model replies:

"I don't have enough information to answer that."

🔍 Key Configuration
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K = 5
MIN_SIMILARITY = 0.40


These values balance context quality, recall, and performance.

🛠 Tech Stack

Python

ChromaDB – Vector database

SentenceTransformers – Local embeddings

PyTorch (CPU)

GitHub Models (Azure Inference) – LLM responses

BeautifulSoup – HTML parsing

Requests – Web fetching
