# Simplified Development Dockerfile for Business RAG System with FastAPI & provider-agnostic LLM
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF processing, ChromaDB, and builds
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    poppler-utils \
    libxml2-dev \
    libxslt-dev \
    rustc \
    cargo \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv and upgrade packaging tools
RUN pip install --upgrade pip setuptools wheel uv

# Copy dependency file
COPY pyproject.toml ./

# Install heavy dependencies first to leverage caching
RUN uv pip install --system --no-cache-dir \
    sentence-transformers \
    chromadb
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Install the rest including FastAPI & LLM deps
RUN uv pip install --system --no-cache-dir \
    python-dotenv \
    python-multipart \
    PyPDF2 \
    python-docx \
    tiktoken \
    numpy \
    scikit-learn \
    langchain \
    langchain-openai \
    langchain-community \
    fastapi \
    uvicorn[standard] \
    pytest \
    black \
    ruff

# Create necessary directories
RUN mkdir -p src/business_rag tests documents data

# Set Python path so we can import from src
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1


# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models


# Create non-root user for security
# In your Dockerfile, ensure proper user and permissions
RUN mkdir -p /app/data/uploads \
    && chmod -R 777 /app/data/uploads
    
RUN mkdir -p /app/data /root/.cache/huggingface && \
    chmod -R 777 /app/data && \
    chmod -R 777 /root/.cache/huggingface

# Or create a non-root user
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8001

# Health check endpoint (optional)
# HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=1 \
#     CMD curl -f http://localhost:8000/health || exit 1

# Default command to run FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
