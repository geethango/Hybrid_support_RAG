import time
import chromadb
import argparse
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "manual"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = r"D:\BOT_Rag\Llama-3.2-1B-Instruct-IQ3_M.gguf"

# -----------------------------------------------------------------------------------
# LOAD MODELS ONLY ONCE
# -----------------------------------------------------------------------------------

print("[INFO] Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

print("[INFO] Loading Llama model...")
llm = Llama(
    model_path=LLAMA_MODEL_PATH,
    n_ctx=4096,
    temperature=0.0,
)

print("[INFO] Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_collection(COLLECTION_NAME)


def run_query(q, top_k=5):
    print(f"[INFO] Running vector search for query: {q}")

    # -----------------------------------------
    # RETRIEVAL LATENCY LOGGING
    # -----------------------------------------
    t0 = time.time()

    results = col.query(
        query_embeddings=[embedder.encode(q).tolist()],
        n_results=top_k
    )

    retrieval_latency = time.time() - t0
    print(f"[INFO] Retrieval time: {retrieval_latency:.4f} sec")

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n---\n\n".join(
        f"Page {m.get('page')}\n{d}" for d, m in zip(docs, metas)
    )

    prompt = f"""
You are a helpful assistant. Use ONLY the provided context.
Rules:
- Do NOT say "I don't know"
- Do NOT guess
- Answer strictly from context
- Include page numbers

QUESTION:
{q}

CONTEXT:
{context}

Provide the BEST answer using ONLY the given context.
"""

    # -----------------------------------------
    # LLM GENERATION LATENCY LOGGING
    # -----------------------------------------
    t1 = time.time()

    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )

    generation_latency = time.time() - t1
    print(f"[INFO] Generation time: {generation_latency:.4f} sec")

    return out["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="RAG Query Engine With Latency Logs")
    parser.add_argument("-q", "--query", required=True)
    args = parser.parse_args()

    answer = run_query(args.query)
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
