# Simplified Development Dockerfile for Business RAG System
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF processing and ChromaDB
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    rustc \
    cargo \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv and upgrade packaging tools to ensure wheels are used
RUN pip install --upgrade pip setuptools wheel uv

# Copy dependency file
COPY pyproject.toml ./

# Install heavy deps first to cache separately
RUN uv pip install --system --no-cache-dir \
    sentence-transformers \
    chromadb

# Install the rest
RUN uv pip install --system --no-cache-dir \
    python-dotenv \
    PyPDF2 \
    python-docx \
    tiktoken \
    numpy \
    scikit-learn \
    pytest \
    black \
    ruff

# Create directories
RUN mkdir -p src/business_rag tests documents data

# Set Python path so we can import from src
ENV PYTHONPATH=/app/src

# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models

CMD ["bash"]
