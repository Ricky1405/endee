import streamlit as st

from app import (
    load_documents,
    build_or_update_index,
    semantic_search,
    generate_answer,
    get_chroma_client,
    get_embedding_function,
)

@st.cache_resource
def init_backend():
    return get_chroma_client(), get_embedding_function()

chroma_client, embedding_fn = init_backend()

st.set_page_config(page_title="Local RAG System", layout="wide")

st.title("🔎 Local-Embedding RAG System")
st.markdown("Privacy-centric document intelligence")

# Sidebar
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("Index Documents"):
    with st.spinner("Indexing..."):
        docs = load_documents()
        added = build_or_update_index(docs, chroma_client, embedding_fn)
    st.sidebar.success(f"Indexed {added} new chunks")

# Question input
question = st.text_input("Ask a question about your documents:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching..."):
            results = semantic_search(question, chroma_client, embedding_fn)

        if not results:
            st.error("I don't have enough information.")
        else:
            context = [doc for doc, _, _ in results]

            with st.spinner("Generating answer..."):
                answer = generate_answer(question, context)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for _, meta, score in results:
                st.write(f"- {meta['source']} (similarity: {score:.3f})")
