"""Shared settings. Both ingest.py and the retrieval code import these,
so they always point at the same database."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CHROMA_DIR = APP_DIR.parent / "chroma_db"   # RAG-with-RBAC/chroma_db
COLLECTION_NAME = "company_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"