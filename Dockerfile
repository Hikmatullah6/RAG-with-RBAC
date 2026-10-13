# ---- FinSolve RAG chatbot: Streamlit + ChromaDB + Groq ----
    FROM python:3.12-slim

    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        HF_HOME=/opt/hf-cache
    
    WORKDIR /srv
    
    # 1. CPU-only PyTorch. The default Linux build bundles ~2 GB of GPU libraries we don't need.
    RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
    
    # 2. App dependencies (copied first so Docker can reuse this layer when only code changes)
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    # 3. Code and documents
    COPY app/ app/
    COPY resources/ resources/
    
    # 4. Build the vector database inside the image.
    #    This also downloads the embedding model into HF_HOME, so the container
    #    starts without downloading anything.
    RUN python app/ingest.py
    
    # 5. Run as a normal user, not root
    RUN useradd --create-home appuser \
        && mkdir -p logs \
        && chown -R appuser /srv /opt/hf-cache
    USER appuser
    
    # Everything is already downloaded: don't contact Hugging Face at runtime
    ENV HF_HUB_OFFLINE=1
    
    EXPOSE 8501
    CMD ["streamlit", "run", "app/app.py", \
         "--server.port=8501", "--server.address=0.0.0.0", \
         "--server.headless=true", "--browser.gatherUsageStats=false"]