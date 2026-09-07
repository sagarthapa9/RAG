# Business RAG System

A Retrieval-Augmented Generation (RAG) system for business document analysis: ingest PDF/DOCX/TXT documents, chunk and embed them, store them in a vector database, and answer questions over them with an LLM — all exposed through a FastAPI service.

## 🧠 What's implemented

The full RAG pipeline is in place, end to end:

| Stage | Component | File |
|---|---|---|
| Read | `DocumentReader` (PDF/DOCX/TXT + metadata) | `src/rag/document_reader.py` |
| Chunk | `DocumentChunker` (recursive / fixed, tiktoken) | `src/rag/chunking.py` |
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
| POST | `/api/upload/bulk` | Upload many documents at once (multipart, multiple `files`); saves all, ingests one at a time in a background thread, returns a `job_id` |
| GET | `/api/upload/bulk/{job_id}` | Poll a bulk upload job — status + per-file results |
| POST | `/api/ingest/folder` | Ingest every supported document already present in a server-side folder (default `documents/`) — **no HTTP upload**; returns the same `job_id` |
| GET | `/api/ingest/jobs/{job_id}` | Poll a folder-ingest job — same registry + response shape as `GET /api/upload/bulk/{job_id}` |
| POST | `/api/search` | Semantic search (retrieval only, no LLM) |
| POST/GET | `/api/qa/ask` | Ask a question — retrieves chunks, LLM answers with sources |
| POST | `/api/qa/ask/stream` | Ask a question via Server-Sent Events — `sources`, then `token` deltas, then `done` |
| GET | `/api/qa/health` | Q&A pipeline health |
| GET | `/api/upload/status` | Upload config + vector store status |
| POST | `/api/vector-store/clear` | Empty the Chroma collection (chunks + vectors). Requires body `{"confirm": true}` — disk files and uploads untouched |
| GET | `/api/vector-store/rows` | Page over stored chunks — the `SELECT *` view: id, text, metadata, embedding. Params: `limit`, `offset`, `include_embeddings` |

Interactive docs at `http://localhost:8080/docs` (or the port you chose).

## 📤 Bulk upload (many files at once)

`POST /api/upload/bulk` ingests a whole folder of documents in one request without blocking the
server or holding every document in memory at once.

### How it works — accept fast, process in the background

The endpoint splits work into a quick **accept** phase and a slow **background** phase, so a
request for 100 files returns in the time it takes to *save* them, not to *embed* them.

```
CLIENT                      FastAPI (api/main.py)               Background thread
──────                      ─────────────────────               ─────────────────
POST /api/upload/bulk  ───► save each file to disk       ───►   one file at a time:
(100 files)                 record per-file pending/skipped      read → chunk (~200 tok)
                            create job {status:"queued"}          → embed → add to Chroma
                            return {job_id} immediately          failures stop only that file
                                    │                                   │
poll GET /api/upload/      ◄───────┴──────── status: queued → running → completed
bulk/{job_id}                       (per-file results: success / failed / skipped)
```

1. **`POST /api/upload/bulk`** validates and saves every uploaded file to disk, then registers
   an in-memory job and returns `{job_id, files_accepted, files_rejected}` **immediately** —
   before any embedding runs.
2. A **background thread** (`_run_bulk_job`) then processes the saved files **one at a time**,
   each through the normal read → chunk → embed → add pipeline. Every file lands in the same
   Chroma collection, so a later search spans all of them.
3. **Per-file isolation:** if one file fails to process, it is marked `failed` with its error and
   the remaining files keep going.
4. **`GET /api/upload/bulk/{job_id}`** returns the job snapshot — status plus per-file results —
   for the client to poll until `completed`.

Processing one file at a time keeps peak memory to a single file's chunks and lets a bad file
fail without losing the earlier ones. Because embedding runs on a background thread, the server
keeps answering other requests while a large batch is ingesting.

### Request body — multipart/form-data, two fields

| Field | Type | Meaning |
|---|---|---|
| `files` | **array** of files (each an `UploadFile`) | every file part named `files`; Swagger shows it as "array of string / binary" because OpenAPI models file bytes as strings |
| `chunking_strategy` | **string** | `recursive` (default) or `fixed` |

An "array" in multipart form data just means the `files` field repeats once per file (there is
no JSON array in the body). `-F "files=@path"` sends one file; repeat it for more:

```bash
curl -X POST http://localhost:8080/api/upload/bulk \
  -F "chunking_strategy=recursive" \
  -F "files=@note1.txt" \
  -F "files=@note2.pdf" \
  -F "files=@report.md"
```

Immediate response:

```json
{"job_id":"ab12cd34ef56","status":"queued","files_accepted":3,"files_rejected":0,
 "message":"Bulk upload queued: 3 file(s) accepted, 0 rejected. Poll GET /api/upload/bulk/ab12cd34ef56"}
```

Poll until `"status": "completed"`:

```bash
curl http://localhost:8080/api/upload/bulk/ab12cd34ef56
```

```json
{"job_id":"ab12cd34ef56","status":"completed","total_files":3,"succeeded":3,"failed":0,
 "skipped":0,"total_chunks":47,
 "files":[{"filename":"note1.txt","status":"success","chunks_created":12},
          {"filename":"note2.pdf","status":"success","chunks_created":30},
          {"filename":"report.md","status":"success","chunks_created":5}],
 "created_at":"2026-09-06T10:00:00","finished_at":"2026-09-06T10:00:42"}
```

### Notes & limits

- Jobs live **in memory**: a server restart loses them, and only the most recent 20 jobs are kept.
- Uploaded files stay on disk even if ingestion fails, and are never deleted by a later clear —
  `POST /api/vector-store/clear` only empties the Chroma collection.
- Files that fail extension validation or the 50 MB per-file cap are reported as `skipped`; the
  rest still process. If *no* file is accepted, the request returns `400`.
- Typical fresh-load workflow: clear the store, then bulk-upload everything.
- Avoid two concurrent ingests into the same store (a bulk job and a `process_immediately=true`
  single upload at once) — writes to the Chroma client are not serialized between them.

## 📁 Folder ingest (no upload — for large batches)

Have documents already on the machine the server runs on (like your 512 files)? Skip HTTP
entirely: drop them into the project's `documents/` folder — any subfolders are fine — and fire
one request. Under Docker, `documents/` is bind-mounted at `/app/documents`, so the container
sees the files as soon as you drop them there.

```
documents/  (drop 512 files here)        FastAPI                         Background thread
   report-01.pdf   ──►  POST /api/ingest/folder  ──►  scan folder (recursive)
   notes/*.txt          scans & counts supported files   register job {queued}
   ...                  returns {job_id} immediately           │
                                                               ▼
   poll GET /api/ingest/jobs/{job_id}  ◄─────────  one file at a time → chunk → embed → add
```

`POST /api/ingest/folder` — JSON body, all fields optional:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `folder` | string | `documents` | folder to scan, relative to the project root, or an absolute path inside it |
| `recursive` | boolean | `true` | also scan subfolders |
| `chunking_strategy` | string | `recursive` | `recursive` or `fixed` |

```bash
# Ingest the whole documents/ folder (subfolders included)
curl -X POST http://localhost:8080/api/ingest/folder \
  -H "Content-Type: application/json" -d '{}'

# A single subfolder, top level only
curl -X POST http://localhost:8080/api/ingest/folder \
  -H "Content-Type: application/json" \
  -d '{"folder": "documents/contracts", "recursive": false}'
```

Immediate response mirrors bulk upload (`{"job_id": "...", "status": "queued", ...}`); poll until
`completed`:

```bash
curl http://localhost:8080/api/ingest/jobs/ab12cd34ef56
```

Notes:

- Nothing crosses HTTP and nothing is copied to the uploads spool — the server reads each file
  **in place**, one at a time, in a background thread. The job response / polling is identical to
  bulk upload; both share one in-memory job registry (`/api/ingest/jobs/{job_id}` is an alias of
  `/api/upload/bulk/{job_id}`).
- Files with an unsupported extension or over the 50 MB cap are reported `skipped`; the rest
  still ingest. If no supported file is found, the request returns `400`.
- Only folders inside the project root can be scanned; internal dirs (`data/`, `.venv`, `src/`,
  `tests/`, `uploads_data/`, …) are refused so the Chroma store and the embedding model cache
  (which contain `.json` files) can't be ingested by accident.
- Same limits as bulk upload: jobs are in-memory (lost on restart, most recent 20 kept), and don't
  run two concurrent ingests into the same store.

## 🔍 Inspect the vector store (SELECT *)

`GET /api/vector-store/rows` pages over everything stored — the vector-store equivalent of
`SELECT * FROM collection LIMIT ? OFFSET ?`. Every row is one **chunk**: its id, the chunk text,
its metadata, and the embedding. Paginate so the response stays small:

```bash
# First 10 chunks, without the big float vectors
curl "http://localhost:8080/api/vector-store/rows?limit=10"

# Next page
curl "http://localhost:8080/api/vector-store/rows?limit=10&offset=10"

# Include the full embedding vector (384 floats per row for all-MiniLM-L6-v2)
curl "http://localhost:8080/api/vector-store/rows?limit=2&include_embeddings=true"
```

```json
{"total_chunks": 794, "offset": 0, "limit": 2, "rows": [
  {"id": "doc_0_1a2b3c4d5e6f", "text": "Principal risks ...", "metadata": {"source": "tesco-annual-report-2025.pdf", "page": 12}, "embedding_length": 384},
  {"id": "doc_1_9f8e7d6c5b4a", "text": "Financial review ...", "metadata": {"source": "tesco-annual-report-2025.pdf", "page": 12}, "embedding_length": 384}
]}
```

`embedding_length` is always reported; the vector itself is returned only when
`include_embeddings=true`. Rows come back in store order (newest ingested last).

> With `CHUNK_ENRICHMENT` on (the default), every chunk's text starts with its document
> identity — e.g. `Vanguard Japan Stock Index Fund EUR Acc [IE0007286036] | ...`. This prefix is
> part of what gets embedded, so a plain question like *"ongoing charges for the Vanguard Japan
> fund?"* finds that fund's chunks by similarity alone, even though 521 funds share near-identical
> KIID boilerplate. See the *Metadata* section in `docs/choosing-embedding-models.md`.

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
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking params | `200` / `50` |
| `CHUNK_ENRICHMENT` | Prefix each chunk's text with its document identity (filename stem) before embedding, so a generic question that names a document can retrieve its chunks without a metadata filter; requires re-ingest | `true` |
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
