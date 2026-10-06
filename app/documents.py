from collections import Counter
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

# Build the path from this file's location, so it works no matter
# which folder you run the script from.
# app/load_documents.py -> app/ -> RAG-with-RBAC/ -> resources/data
DATA_DIR = Path(__file__).resolve().parent.parent / "resources" / "data"


def load_documents() -> list[Document]:
    documents = []

    # Each subfolder of data/ is a department (engineering, finance, hr, ...)
    for dept_folder in sorted(DATA_DIR.iterdir()):
        if not dept_folder.is_dir():
            continue
        department = dept_folder.name.lower()

        for file in sorted(dept_folder.iterdir()):
            suffix = file.suffix.lower()

            if suffix == ".md":
                # Whole markdown file = one document (we'll chunk it in Step 3)
                documents.append(Document(
                    page_content=file.read_text(encoding="utf-8"),
                    metadata={"department": department, "source": file.name},
                ))

            elif suffix == ".csv":
                # One document per row, with column names kept in the text
                df = pd.read_csv(file)
                for i, row in df.iterrows():
                    text = "; ".join(f"{col}: {row[col]}" for col in df.columns)
                    documents.append(Document(
                        page_content=text,
                        metadata={"department": department, "source": f"{file.name} (row {i + 1})"},
                    ))

    return documents


# Only runs when you execute this file directly, not when another file imports it
if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No documents found in {DATA_DIR} - check the path.")
    print(f"Loaded {len(docs)} documents from {DATA_DIR}\n")

    counts = Counter(doc.metadata["department"] for doc in docs)
    for department, count in sorted(counts.items()):
        print(f"  {department}: {count}")

    print("\nExample document:")
    print(docs[0].metadata)
    print(docs[0].page_content[:200])