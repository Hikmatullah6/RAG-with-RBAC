"""Search the vector database (no LLM yet).
Run:  python retrieve.py
"""
from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION_NAME
from ingest import get_embeddings

TOP_K = 5


def get_vectorstore():
    """Open the EXISTING database built by ingest.py (does not rebuild it)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),  # same model used to embed the chunks
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(vectorstore, question: str, k: int = TOP_K):
    """Return (document, distance) pairs. Lower distance = closer match."""
    return vectorstore.similarity_search_with_score(question, k=k)


def print_results(results):
    for rank, (doc, score) in enumerate(results, start=1):
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"\n[{rank}] score={score:.3f} | {doc.metadata['department']} | {doc.metadata['source']}")
        print(f"    {preview}...")


if __name__ == "__main__":
    if not CHROMA_DIR.exists():
        raise SystemExit("No database found. Run `python ingest.py` first.")

    vectorstore = get_vectorstore()
    print(f"Loaded {len(vectorstore.get()['ids'])} chunks. Type a question (or 'quit').")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            print_results(retrieve(vectorstore, question))