import sys
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from documents import load_documents

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_documents(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Load all documents and split them into chunks. Metadata is copied onto every chunk."""
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


if __name__ == "__main__":
    # Optional: pass a chunk size to experiment, e.g.  python chunk_documents.py 300
    size = int(sys.argv[1]) if len(sys.argv) > 1 else CHUNK_SIZE
    chunks = chunk_documents(chunk_size=size)

    print(f"chunk_size={size}, overlap={CHUNK_OVERLAP}")
    print(f"Total chunks: {len(chunks)}\n")

    counts = Counter(chunk.metadata["department"] for chunk in chunks)
    for department, count in sorted(counts.items()):
        print(f"  {department}: {count}")

    # Show the first chunk from each department so you can judge chunk quality
    print("\nSample chunks (first one from each department):")
    seen = set()
    for chunk in chunks:
        department = chunk.metadata["department"]
        if department in seen:
            continue
        seen.add(department)
        print("\n" + "-" * 60)
        print(f"{chunk.metadata}  |  {len(chunk.page_content)} characters")
        print("-" * 60)
        print(chunk.page_content)