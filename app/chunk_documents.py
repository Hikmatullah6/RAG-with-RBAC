"""Split documents into chunks that remember which section they came from.

Markdown files: split by headings first, then by size. Each chunk's text starts
with its heading path, e.g.
    [Quarterly Financial Report > Q2 - April to June 2024 > Quarterly Expense Breakdown]
so "Q2" is part of what gets embedded and what the LLM sees.

CSV rows: already one small document per row, so they pass through unchanged.

Run:  python chunk_documents.py        (or: python chunk_documents.py 400)
"""
import sys
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from documents import load_documents

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Which heading levels to split on, and the metadata key each one is stored under
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def split_markdown(doc: Document, size_splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    # 1. Split into sections at each heading. Each section's metadata holds its headings,
    #    e.g. {"h1": "Quarterly Financial Report", "h2": "Q2 - April to June 2024", ...}
    sections = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS).split_text(doc.page_content)

    chunks = []
    for section in sections:
        # 2. Build the heading path: "Report > Q2 - April to June 2024 > Expense Breakdown"
        heading_path = " > ".join(
            section.metadata[key] for _, key in HEADERS if key in section.metadata
        )

        # 3. Long sections still get cut to size; every piece gets the heading path added
        for piece in size_splitter.split_text(section.page_content):
            text = f"[{heading_path}]\n{piece}" if heading_path else piece
            chunks.append(Document(
                page_content=text,
                metadata={**doc.metadata, "section": heading_path},  # keep department + source
            ))
    return chunks


def chunk_documents(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for doc in load_documents():
        if doc.metadata["source"].endswith(".md"):
            chunks.extend(split_markdown(doc, size_splitter))
        else:
            chunks.append(doc)  # CSV row: already small, keep as is
    return chunks


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else CHUNK_SIZE
    chunks = chunk_documents(chunk_size=size)

    print(f"chunk_size={size}, overlap={CHUNK_OVERLAP}")
    print(f"Total chunks: {len(chunks)}\n")

    counts = Counter(chunk.metadata["department"] for chunk in chunks)
    for department, count in sorted(counts.items()):
        print(f"  {department}: {count}")

    # Show the first chunk from each department so you can check the heading paths
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