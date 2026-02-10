"""
RAG (Retrieval Augmented Generation) App – Local Embeddings
-----------------------------------------------------------
Uses:
- ChromaDB (persistent vector DB)
- SentenceTransformers (PyTorch) for local embeddings
- GitHub Models (Azure Inference) for chat generation
"""

import os
import argparse
import hashlib
from pathlib import Path
import time

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import requests
from bs4 import BeautifulSoup

# Optional built-in URLs. If you prefer managing links via `documents/urls.txt`,
# you can leave this empty.
URLS: list[str] = []

DEFAULT_HEADERS = {
    # A basic UA reduces blocking vs python-requests default.
    "User-Agent": "rag_app/1.0 (+https://github.com/) python-requests",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def load_from_url(url: str, timeout: int = 15, max_retries: int = 2) -> tuple[str, str]:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout, headers=DEFAULT_HEADERS)
            response.raise_for_status()
            html = response.text
            break
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                raise
            time.sleep(0.5 * (attempt + 1))
    else:
        # Should never happen, but keeps type-checkers happy.
        raise RuntimeError(f"Failed to fetch {url}: {last_err}")

    soup = BeautifulSoup(html, "html.parser")

    # remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(soup.get_text().split())
    return url, text




# ===============================
# 1. Load Environment Variables
# ===============================
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN missing from environment")


# ===============================
# 2. Configuration
# ===============================
DOCUMENTS_FOLDER = "documents"
CHROMA_PERSIST_DIR = "./chroma_db"
# Use a new collection name so we don't conflict with any
# previous OpenAI-based embedding configuration in Chroma.
CHROMA_COLLECTION_NAME = "rag_docs_local"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K = 5
MIN_SIMILARITY = 0.40# confidence gate

CHAT_MODEL_NAME = "openai/gpt-4o-mini"
ENDPOINT = "https://models.github.ai/inference"


# ===============================
# 3. Clients
# ===============================
chat_client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_embedding_function():
    """
    Use local SentenceTransformers embeddings (PyTorch).
    No OpenAI / external embedding API is used.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


# ===============================
# 4. Document Loading
# ===============================
def read_urls_file(path: Path) -> list[str]:
    """
    Reads URLs from a text file.
    - Extracts only http(s):// links (so headings/notes are allowed)
    - Supports blank lines and comments starting with '#'
    """
    if not path.exists():
        return []

    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Allow notes like: "https://example.com — description"
        # Only keep actual http(s) links.
        if line.startswith("http://") or line.startswith("https://"):
            urls.append(line.split()[0])
    return urls


def load_documents(folder_path: str = DOCUMENTS_FOLDER) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []

    # Load .txt files
    folder = Path(folder_path)
    folder.mkdir(exist_ok=True)
    for file in sorted(folder.glob("*.txt")):
        # This file is for URL input, not a knowledge document.
        if file.name.lower() == "urls.txt":
            continue
        try:
            content = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file.read_text(encoding="utf-8", errors="replace")
        if content.strip():
            docs.append((file.name, content))

    # Load URLs (optional): documents/urls.txt overrides/extends hardcoded list
    urls_from_file = read_urls_file(folder / "urls.txt")
    all_urls = list(dict.fromkeys(urls_from_file + URLS))  # de-dupe, keep order

    for url in all_urls:
        try:
            source, text = load_from_url(url)
            # Skip pages that extract to almost nothing (common for JS-rendered sites)
            if len(text) < 400:
                print(f"Skipping URL (too little text): {url}")
                continue
            docs.append((source, text))
        except Exception as e:
            print(f"Failed to load {url}: {e}")

    if not docs:
        print(
            f"No documents loaded. Add .txt files to '{folder_path}/' "
            f"and/or URLs to '{folder_path}/urls.txt'."
        )

    return docs



# ===============================
# 5. Chunking
# ===============================
def split_into_chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap

    return chunks


def make_chunk_id(filename, chunk):
    digest = hashlib.sha256(chunk.encode()).hexdigest()[:12]
    return f"{filename}__{digest}"


# ===============================
# 6. Indexing (Offline / Explicit)
# ===============================
def build_or_update_index(documents, chroma_client, embedding_fn):
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    existing_ids = set(collection.get()["ids"])
    new_ids, metadatas, chunks = [], [], []

    for filename, content in documents:
        for chunk in split_into_chunks(content):
            cid = make_chunk_id(filename, chunk)
            if cid not in existing_ids:
                new_ids.append(cid)
                metadatas.append({"source": filename})
                chunks.append(chunk)

    if new_ids:
        collection.add(
            ids=new_ids,
            documents=chunks,
            metadatas=metadatas,
        )

    return len(new_ids)


# ===============================
# 7. Semantic Search
# ===============================
def semantic_search(question, chroma_client, embedding_fn):
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    results = collection.query(
        query_texts=[question],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist
        if similarity >= MIN_SIMILARITY:
            hits.append((doc, meta, similarity))

    return hits


# ===============================
# 8. Answer Generation
# ===============================
def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a retrieval-based QA system.
You must answer ONLY using the provided context.
Do not use prior knowledge.
If the context is insufficient, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}
"""

    response = chat_client.complete(
        model=CHAT_MODEL_NAME,
        messages=[
            SystemMessage("Strictly follow retrieval context."),
            UserMessage(prompt),
        ],
        temperature=0.2,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()


# ===============================
# 9. Main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Index documents only")
    args = parser.parse_args()

    chroma_client = get_chroma_client()
    embedding_fn = get_embedding_function()

    if args.index:
        print("Indexing documents (local embeddings)...")
        docs = load_documents()
        added = build_or_update_index(docs, chroma_client, embedding_fn)
        print(f"New chunks indexed: {added}")
        return

    print("RAG App – Semantic Search (local embeddings)")
    print("Type 'quit' to exit")
    print("-" * 60)

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break

        results = semantic_search(question, chroma_client, embedding_fn)
        if not results:
            print("Answer: I don't have enough information.\n")
            continue

        context = [doc for doc, _, _ in results]
        answer = generate_answer(question, context)

        print("\nAnswer:\n", answer)
        print("\nSources:")
        for _, meta, score in results:
            print(f"- {meta['source']} (similarity: {score:.3f})")
        print("-" * 60)


if __name__ == "__main__":
    main()
