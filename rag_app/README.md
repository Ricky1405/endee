# 🧠 Local-Embedding RAG System

A privacy-first, production-grade Retrieval-Augmented Generation (RAG) system built using local embeddings, a persistent vector database, and GitHub Models (Azure Inference) for grounded answer generation.

This project demonstrates how to build a cost-efficient, scalable, and offline-friendly RAG pipeline without relying on paid embedding APIs.

---

## 🚀 Features

- 📚 Retrieval-Augmented Generation (RAG)
- 🔎 Semantic search using vector similarity
- 🧠 100% local embeddings (SentenceTransformers – no OpenAI embedding cost)
- 🗄 Persistent vector storage using ChromaDB
- 🌐 Web page ingestion via `urls.txt`
- 📄 Local document ingestion (.txt files)
- 🧩 Smart chunking with overlap
- ♻️ Deterministic duplicate-safe indexing (SHA-256 hashing)
- 🎯 Similarity threshold gating (reduces hallucinations)
- 📌 Source attribution for answers
- 🖥 Multiple UI options (CLI / Tkinter / Streamlit)

---

## 🏗 System Architecture

### 1️⃣ Data Ingestion Layer
- Local `.txt` file loader
- Web scraping (Requests + BeautifulSoup4)
- HTML cleaning (removes `<script>`, `<style>`, `<noscript>`)
- URL retry logic

### 2️⃣ Processing & Embedding Layer
- Chunk size: **200**
- Overlap: **50**
- Embedding model: **all-MiniLM-L6-v2**
- Framework: **PyTorch (CPU compatible)**

### 3️⃣ Vector Storage Layer
- **ChromaDB (Persistent)**
- Cosine similarity search
- On-disk storage
- Incremental indexing support

### 4️⃣ Retrieval & Generation Layer
- Top-K semantic retrieval
- Similarity threshold: **0.40**
- Context-only prompting
- LLM: **GPT-4o-mini (GitHub Models via Azure Inference SDK)**

---

## 📁 Project Structure

rag_app/
├── app.py
├── ui.py
├── streamlit_app.py
├── requirements.txt
├── chroma_db/
├── documents/
│ ├── urls.txt
│ └── *.txt
└── .env


---

## ⚙️ Prerequisites

- Python 3.10+
- GitHub account with GitHub Models access
- Internet (for URL ingestion only)

---

## 🔐 Environment Setup

Create a `.env` file:

GITHUB_TOKEN=your_github_personal_access_token


Get token from:
https://github.com/marketplace/models

---

## 📦 Installation

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt

pip install torch --index-url https://download.pytorch.org/whl/cpu

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install CPU PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

📄 Adding Documents

Place .txt files inside:

documents/


Add URLs inside:

documents/urls.txt


Rules:

One URL per line

Blank lines allowed

Lines starting with # ignored

🧱 Index Data
python app.py --index


This:

Loads documents

Chunks text

Generates embeddings

Stores vectors in ChromaDB

❓ Ask Questions
CLI
python app.py

Tkinter UI
python ui.py

Streamlit UI
streamlit run streamlit_app.py

🧠 Hallucination Control

Similarity threshold filtering

Strict context-only prompting

If insufficient context:

"I don't have enough information to answer that."

📊 Performance Characteristics

⚡ Fast local embedding generation

💰 Zero embedding API cost

📈 Scalable retrieval via persistent vector DB

🔒 No document data sent externally during indexing

🛠 Tech Stack

Python 3.10+

ChromaDB

SentenceTransformers

PyTorch

GitHub Models (Azure Inference)

BeautifulSoup4

Requests

Tkinter

Streamlit


---

# ✅ Now Finish the Rebase

Run:

```bash
git add rag_app/README.md
git rebase --continue
git push origin dev


This keeps:

Your professional architecture description

Your UI updates

Your earlier structured explanations

Clean formatting

No conflict markers
