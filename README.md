# Business RAG System

A Retrieval-Augmented Generation (RAG) system for business document analysis: ingest PDF/DOCX/TXT documents, chunk and embed them, store them in a vector database, and answer questions over them with an LLM — all exposed through a FastAPI service.

## 🧠 What's implemented

The full RAG pipeline is in place, end to end:

| Stage | Component | File |
|---|---|---|
| Read | `DocumentReader` (PDF/DOCX/TXT + metadata) | `src/rag/document_reader.py` |
| Chunk | `DocumentChunker` (recursive / fixed, tiktoken) | `src/rag/chunking.py` |
| Embed | `EmbeddingGenerator` (sentence-transformers) | `src/rag/embedding.py` |
| Store | `ChromaVectorStore` (LangChain `VectorStore`, cosine) | `src/rag/vector_store.py` |
| Retrieve | `RAGPipeline.search()` → `Document` objects | `src/rag/pipeline.py` |
| Generate | `LLMRAGPipeline` (provider-agnostic LLM — OpenAI-compatible, Azure, or local, grounded answers) | `src/rag/llm_rag.py` |
| API | FastAPI app with upload / search / Q&A | `src/api/main.py` |

## 🚀 Quick start (Docker)

```bash
# 1. Create a .env with your LLM provider + key (needed for Q&A only;
#    upload + semantic search work without it). See .env.example. For Kimi:
#    LLM_PROVIDER=kimi
#    KIMI_API_KEY=sk-...

# 2. Build and start the API (http://localhost:8080)
docker-compose up --build rag-dev
```

## 🚀 Quick start (local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]

# Run from the project root; PYTHONPATH=src makes the `rag`/`api` packages importable.
$env:PYTHONPATH = "src"     # PowerShell
# export PYTHONPATH=src     # bash
uvicorn api.main:app --reload --port 8001
```

> Verified on Python **3.14** with the current releases (chromadb 1.x, langchain 1.x, sentence-transformers 6.x). The code was modernized for those APIs.

## 🔌 API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health + availability of RAG/LLM pipelines |
| GET | `/system/info` | System configuration |
| POST | `/api/upload/single` | Upload a document (multipart), process immediately or in background |
| POST | `/api/search` | Semantic search (retrieval only, no LLM) |
| POST/GET | `/api/qa/ask` | Ask a question — retrieves chunks, LLM answers with sources |
| POST | `/api/qa/ask/stream` | Ask a question via Server-Sent Events — `sources`, then `token` deltas, then `done` |
| GET | `/api/qa/health` | Q&A pipeline health |
| GET | `/api/upload/status` | Upload config + vector store status |

Interactive docs at `http://localhost:8080/docs` (or the port you chose).

## ⚙️ Environment variables

The LLM layer is provider-agnostic — set `LLM_PROVIDER` to pick the provider, then give it a key:

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | Provider: `openai`, `kimi`, `deepseek`, `ollama`, `azure`, `custom` | `kimi` |
| `LLM_API_KEY` | Generic key, honored by any provider (optional if provider-specific key set) | — |
| `LLM_BASE_URL` | Generic OpenAI-compatible base URL override (not used by `azure`) | provider default |
| `LLM_MODEL` | Generic model override | provider default |
| `OPENAI_API_KEY` | Key when `LLM_PROVIDER=openai` | — |
| `KIMI_API_KEY` | Key when `LLM_PROVIDER=kimi` (Moonshot AI) | — |
| `KIMI_BASE_URL` | Kimi endpoint (`https://api.moonshot.cn/v1` China / `.ai/v1` international) | `https://api.moonshot.cn/v1` |
| `KIMI_MODEL` | Kimi model (e.g. `moonshot-v1-8k`, `moonshot-v1-32k`, `kimi-k2.6`) | `moonshot-v1-8k` |
| `DEEPSEEK_API_KEY` | Key when `LLM_PROVIDER=deepseek` | — |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | Endpoint / model when `LLM_PROVIDER=ollama` (no key) | `http://localhost:11434/v1` / `llama3.1` |
| `AZURE_OPENAI_API_KEY` | Key when `LLM_PROVIDER=azure` | — |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT` | Azure resource URL / deployment name (required for azure) | — |
| `AZURE_OPENAI_API_VERSION` | Azure API version | `2024-02-01` |
| `LLM_TEMPERATURE` | LLM sampling temperature | `0.1` |
| `LLM_MAX_TOKENS` | Max response tokens | `1024` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking params | `512` / `50` |
| `EMBEDDING_MODEL` | Sentence-transformer model | `all-MiniLM-L6-v2` |
| `VECTOR_STORE_PATH` | ChromaDB persist dir | `./data/chromadb` |
| `COLLECTION_NAME` | ChromaDB collection | `business_documents` |

## 📁 Project structure

```
├── src/
│   ├── api/main.py          # FastAPI application
│   └── rag/                 # RAG pipeline modules
├── documents/               # Add source documents here (optional)
├── uploads_data/            # Files uploaded via the API
├── data/chromadb/           # ChromaDB persistence
├── tests/                   # Manual test scripts (run with python tests/...)
├── docker-compose.yml       # rag-dev (API) + chromadb (server, unused by code)
└── Dockerfile
```

## 🧪 Testing

The `tests/` directory contains manual verification scripts (not pytest cases yet):

```bash
python tests/test_document_reader.py    # reader against documents/
python tests/test_vector_system.py      # interactive: ingest + search
```

## 🗺️ Roadmap

- [x] Document reading (PDF / DOCX / TXT)
- [x] Chunking (recursive / fixed)
- [x] Embeddings (sentence-transformers)
- [x] Vector storage + retrieval (ChromaDB)
- [x] LLM generation (provider-agnostic: OpenAI-compatible / Azure / local, grounded)
- [x] FastAPI service (upload, search, Q&A)
- [ ] Streamlit UI
- [ ] Proper pytest suite
- [ ] Production hardening (auth, monitoring, multi-tenant)

---

**Status**: 🏗️ Functional end-to-end pipeline + API, verified locally on Python 3.14 (ingest → search → Q&A). Changes are uncommitted.
