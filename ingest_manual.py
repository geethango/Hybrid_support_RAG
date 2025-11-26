# ingest_manual.py
import os
import time
import fitz  # PyMuPDF
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

PDF_PATH = "manual.pdf"
COLLECTION_NAME = "manual"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_db"


def load_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
    return pages


def main():
    assert os.path.exists(PDF_PATH), f"{PDF_PATH} not found."

    print("Loading PDF pages…")
    pages = load_pdf_pages(PDF_PATH)

    print(f"Total pages: {len(pages)}")

    print("Loading embedding model…")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print("Connecting to Chroma (new API)…")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("Processing pages and inserting into DB…")
    for i, text in enumerate(pages):
        emb = embedder.encode(text).tolist()

        collection.add(
            ids=[f"page_{i+1}"],
            embeddings=[emb],
            documents=[text],
            metadatas=[{"page": i + 1}]
        )

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
