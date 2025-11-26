# query_api.py  (HYBRID RAG VERSION)
import time
import chromadb
import re
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import argparse

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "manual"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

LLAMA_MODEL_PATH = r"D:\BOT_Rag\Llama-3.2-1B-Instruct-IQ3_M.gguf"

PATTERN = re.compile(
    r"in (the )?(?P<chapter>[\w \-&']{3,60}) (chapter|section)",
    re.I
)

def run_query(q, top_k=5, alpha=0.5):
    """
    alpha = 1.0 → pure dense (vector)
    alpha = 0.0 → pure sparse (keyword/BM25)
    alpha = 0.5 → 50/50 Hybrid RAG
    """

    print(f"[INFO] Running Hybrid RAG search (alpha={alpha})...")

    # Load embedding model
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # Chroma client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(COLLECTION_NAME)

    # Hybrid search (text + embedding mix)
    results = col.query(
        query_texts=[q],          # Required for sparse/BM25
        query_embeddings=[embedder.encode(q).tolist()],
        n_results=top_k,
        search_type="hybrid",     # <-- NEW
        alpha=alpha               # <-- blend dense + sparse
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Create context
    context = "\n\n---\n\n".join(
        f"Page {m.get('page')}\n{d}" for d, m in zip(docs, metas)
    )

    # Load Llama
    llm = Llama(
        model_path=LLAMA_MODEL_PATH,
        n_ctx=4096,
        temperature=0.0
    )

    prompt = f"""
Answer ONLY from context. If answer is not found, say "I don't know".

QUESTION:
{q}

CONTEXT:
{context}

Include the page number.
"""

    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )

    return out["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Query Llama with Hybrid RAG.")
    parser.add_argument("-q", "--query", required=True, help="Query string to ask.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid ratio: 1=dense, 0=sparse, 0.5=hybrid")
    args = parser.parse_args()

    answer = run_query(args.query, alpha=args.alpha)
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
