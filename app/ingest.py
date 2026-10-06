"""Build the vector database: load -> chunk -> embed -> store.
Run again whenever the documents change:  python ingest.py
"""
import shutil
from collections import Counter

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from chunk_documents import chunk_documents
from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

load_dotenv()  # picks up HF_TOKEN if you added one


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def ingest():
    # 1. Start fresh so re-running never creates duplicates
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    # 2. Load and chunk (department/source metadata comes along)
    chunks = chunk_documents()
    print(f"Embedding {len(chunks)} chunks...")

    # 3 + 4. Embed every chunk and save vector + text + metadata to disk
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )


if __name__ == "__main__":
    vectorstore = ingest()

    stored = vectorstore.get()  # everything in the collection: ids, documents, metadatas
    print(f"Stored {len(stored['ids'])} chunks in {CHROMA_DIR}\n")

    counts = Counter(meta["department"] for meta in stored["metadatas"])
    for department, count in sorted(counts.items()):
        print(f"  {department}: {count}")