"""Shared setup for the tests. pytest loads this automatically."""
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Let tests import app modules (rag, retrieve, rbac, ...) from the parent folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from config import CHROMA_DIR  # noqa: E402


@pytest.fixture(scope="session")
def vectorstore():
    if not CHROMA_DIR.exists():
        pytest.skip("No database found. Run `python ingest.py` first.")
    from retrieve import get_vectorstore
    return get_vectorstore()  # loaded once for the whole test run


@pytest.fixture(scope="session")
def classifier():
    import os
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    from guardrails import get_injection_classifier
    return get_injection_classifier()