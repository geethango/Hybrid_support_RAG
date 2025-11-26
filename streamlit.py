import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import json

# -------------------------------
# CONFIG
# -------------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "manual"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = r"D:\BOT_Rag\Llama-3.2-1B-Instruct-IQ3_M.gguf"

# -------------------------------
# LOAD MODELS (CACHED)
# -------------------------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_llm():
    return Llama(
        model_path=LLAMA_MODEL_PATH,
        n_ctx=4096,
        temperature=0.0,
        verbose=False
    )

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)

# -------------------------------
# QUERY FUNCTION
# -------------------------------

def rag_query(question, metadata_filter=None, top_k=5):
    embedder = load_embedder()
    llm = load_llm()
    collection = load_chroma()

    st.write("🔍 **Embedding query...**")
    q_emb = embedder.encode(question).tolist()

    st.write("📚 **Retrieving context from ChromaDB...**")
    if metadata_filter:
        st.json({"metadata_filter_used": metadata_filter})

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=metadata_filter
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    st.write("📄 **Pages Retrieved:**", [m["page"] for m in metas])

    context = "\n\n---\n\n".join(
        f"Page {m.get('page')}:\n{d}"
        for d, m in zip(docs, metas)
    )

    # Build LLM prompt
    prompt = f"""
Answer ONLY from context. If answer is not found, say "I don't know".

QUESTION:
{question}

CONTEXT:
{context}

Include the page number.
"""

    st.write("🤖 **Generating answer using Llama...**")
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )

    return output["choices"][0]["message"]["content"]

# -------------------------------
# STREAMLIT UI
# -------------------------------

st.title("📘 PDF RAG Assistant (Chroma + Llama.cpp + Streamlit)")
st.write("Ask questions about your PDF manual.")

question = st.text_input("Enter your question:")
top_k = st.slider("Top K Results", 1, 10, 5)

# Metadata filter UI
with st.expander("📌 Metadata Filters (Optional)"):
    enable_filter = st.checkbox("Enable Page Filter")
    page_lt = st.number_input("Page < x", min_value=1, value=100)

metadata_filter = {"page": {"$lt": int(page_lt)}} if enable_filter else None

if st.button("Run Query"):
    if not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Running RAG pipeline..."):
            answer = rag_query(question, metadata_filter, top_k)
        st.success("Answer:")
        st.write(answer)
