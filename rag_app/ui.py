import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# Import from your app.py
from app import (
    load_documents,
    build_or_update_index,
    semantic_search,
    generate_answer,
    get_chroma_client,
    get_embedding_function,
)

# Initialize backend clients
chroma_client = get_chroma_client()
embedding_fn = get_embedding_function()


class RAGApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("RAG Semantic Search – Local Embeddings")
        self.geometry("950x720")
        self.minsize(800, 600)

        self.configure(padx=15, pady=15)

        self.create_widgets()

    # ===============================
    # UI Layout
    # ===============================
    def create_widgets(self):
        title = ttk.Label(
            self,
            text="RAG App (Chroma + Local Embeddings)",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(pady=10)

        # Index Button
        self.index_button = ttk.Button(
            self,
            text="Index Documents",
            command=self.run_indexing,
        )
        self.index_button.pack(pady=5)

        # Question Section
        ttk.Label(self, text="Ask a Question:", font=("Segoe UI", 12)).pack(
            anchor="w", pady=(20, 5)
        )

        self.question_entry = ttk.Entry(self, font=("Segoe UI", 12))
        self.question_entry.pack(fill="x", pady=5)

        self.ask_button = ttk.Button(
            self,
            text="Ask",
            command=self.run_query,
        )
        self.ask_button.pack(pady=10)

        # Answer Section
        ttk.Label(self, text="Answer:", font=("Segoe UI", 12)).pack(
            anchor="w", pady=(15, 5)
        )

        self.answer_box = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            height=14,
        )
        self.answer_box.pack(fill="both", expand=True)

        # Sources Section
        ttk.Label(self, text="Sources:", font=("Segoe UI", 12)).pack(
            anchor="w", pady=(15, 5)
        )

        self.sources_box = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            height=6,
        )
        self.sources_box.pack(fill="x")

    # ===============================
    # Indexing Logic
    # ===============================
    def run_indexing(self):
        threading.Thread(target=self.index_documents, daemon=True).start()

    def index_documents(self):
        try:
            self.index_button.config(state="disabled")
            self.answer_box.insert(tk.END, "Indexing documents...\n")
            self.answer_box.see(tk.END)

            docs = load_documents()
            added = build_or_update_index(docs, chroma_client, embedding_fn)

            messagebox.showinfo(
                "Index Complete",
                f"New chunks indexed: {added}",
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

        finally:
            self.index_button.config(state="normal")

    # ===============================
    # Query Logic
    # ===============================
    def run_query(self):
        threading.Thread(target=self.ask_question, daemon=True).start()

    def ask_question(self):
        question = self.question_entry.get().strip()

        if not question:
            messagebox.showwarning("Missing Question", "Please enter a question.")
            return

        self.ask_button.config(state="disabled")
        self.answer_box.delete("1.0", tk.END)
        self.sources_box.delete("1.0", tk.END)

        try:
            results = semantic_search(question, chroma_client, embedding_fn)

            if not results:
                self.answer_box.insert(
                    tk.END,
                    "I don't have enough information."
                )
                return

            context = [doc for doc, _, _ in results]
            answer = generate_answer(question, context)

            # Show answer
            self.answer_box.insert(tk.END, answer)

            # Show sources
            for _, meta, score in results:
                line = f"{meta['source']} (similarity: {score:.3f})\n"
                self.sources_box.insert(tk.END, line)

        except Exception as e:
            messagebox.showerror("Error", str(e))

        finally:
            self.ask_button.config(state="normal")


if __name__ == "__main__":
    app = RAGApp()
    app.mainloop()
