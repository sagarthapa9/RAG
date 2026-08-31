# Choosing an embedding model

An embedding model converts a piece of text into a vector (a list of numbers) such that
semantically similar texts land close together in vector space. In this RAG system it is
the **retrieval half** — every chunk is embedded at ingest time (`vector_store.py`),
and every query is embedded at search time and compared against them.

There is no single "best" embedding model. The right choice depends on a handful of
constraints that interact with *your* data, your hardware, and how you deploy. This page
walks through the criteria, then gives a concrete decision path for this project.

---

## The criteria

### 1. Language of your data

The first filter. If your documents and queries are English, an English model is smaller,
faster, and often more accurate. If queries may arrive in multiple languages, you need a
multilingual model.

- English-only: `all-MiniLM-L6-v2`, `BAAI/bge-base-en-v1.5`
- Multilingual: `BAAI/bge-m3` (100+ langs), `intfloat/multilingual-e5-large`, `Qwen/Qwen3-Embedding-0.6B` (100+ langs)

### 2. Context window vs. your chunk size

The embedding model's **context window** (`max_seq_length`) is the maximum number of tokens
it can look at in a single pass. This is the most commonly overlooked criterion, and the one
this project currently trips over.

#### What the context window actually controls

An embedding model reads the entire input and collapses it into **one vector** that
represents the meaning of the whole text. The vector is always the same length (e.g. 384
floats for MiniLM) — the context window is *not* the vector size; it is how much of the text
gets *seen* before the meaning is squeezed into that fixed-size vector.

When the input is longer than the window, the model **silently truncates**: it keeps the
first `max_seq_length` tokens and drops the rest. No error, no warning:

```
Chunk (512 tokens):  [t1][t2] ... [t256][t257] ... [t512]
                            │               │
                 MiniLM (max 256)   ────────┘
                            │        dropped, never seen
              embedded into vector
```

The stored vector only represents the first 256 tokens. Tokens 257–512 contribute nothing to
retrieval.

#### Example: why this causes silent retrieval misses

Suppose a 512-token chunk contains background in its first half and the actual answer in its
second half:

```
[tokens 1–256]   →  "Tesco operates 4,500 stores across the UK. Founded in 1919,
                     the company has grown through acquisitions and a focus on
                     convenience retailing. ..."                            (background)

[tokens 257–512] →  "Revenue in 2024 reached £69.8 billion, up 6.3%, driven by
                     strong food sales. Net profit more than doubled to
                     £2.3 billion ..."                                        (the answer)
```

Ask *"What was Tesco's revenue in 2024?"* The query is embedded, but the chunk vector was
built only from tokens 1–256 — it has no idea "£69.8 billion" exists. Similarity comes out
low, the chunk isn't retrieved, and the RAG pipeline reports "not enough information" even
though the answer is in the document. Not a crash — a **silent miss** that sends you
debugging the LLM, the provider, or the API key instead of the real cause.

#### The rule

The model's `max_seq_length` must be **≥ your chunk's token count**.

> ⚠️ This is the criterion the project currently trips over:
> `DocumentChunker` makes **512-token** chunks (default `CHUNK_SIZE=512`), but the default
> model `all-MiniLM-L6-v2` only reads **256 tokens**. Half of every chunk is being thrown
> away at embedding time. `chunk_overlap` does not fix this — it only softens cuts at chunk
> boundaries, not the model's fixed truncation point.

| Model | Max context |
|---|---|
| `all-MiniLM-L6-v2` | 256 tokens |
| `BAAI/bge-base-en-v1.5` | 512 tokens |
| `intfloat/multilingual-e5-large` | 512 tokens |
| `BAAI/bge-m3` | 8192 tokens |
| `Qwen/Qwen3-Embedding-0.6B` | 32768 tokens |

#### Check a model's limit

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
print(m.max_seq_length)   # 256
```

#### Fixing the mismatch

| Option | Trade-off |
|---|---|
| **Shrink chunks** to ≤ the model window (e.g. `CHUNK_SIZE=200`) | Fits any model; but more, smaller chunks = more noise and more storage |
| **Pick a model with a bigger window** (≥ your chunk size) | The real fix for this setup; `bge-base-en-v1.5` at 512 is the natural match |
| Increase `chunk_overlap` | Only helps near boundaries — does not fix truncation |

### 3. Deployment: local model vs. API

- **Local (sentence-transformers):** free, private (data never leaves your machine), works
  offline after the first download. Cost is compute and model size. This codebase is built
  around it (`SentenceTransformer` in `vector_store.py:118`).
- **API (OpenAI `text-embedding-3`, Cohere, Voyage):** no local compute, larger models, but
  per-token cost, a network dependency, and your document contents are sent to a third party.

For a learning project running on CPU, local is the right default.

### 4. Compute budget / speed

In this setup embeddings run on **CPU** (the logs show `No device provided, using cpu`).
Inference time scales roughly with parameter count:

| Model | Params | Relative speed |
|---|---|---|
| `all-MiniLM-L6-v2` | 22M | 1× (fastest) |
| `BAAI/bge-base-en-v1.5` | 109M | ~2–3× |
| `BAAI/bge-m3` | 568M | ~5× |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | ~8–10× |

Speed matters twice: **indexing** does one pass per chunk (1000 chunks = 1000 model passes),
and **search** runs one pass per query before it can return anything.

### 5. Retrieval quality on *your* data

Leaderboards (MTEB, BEIR) give a general ranking, but the honest way to decide is a small
evaluation on your own documents:

1. Write 10–20 real questions you would actually ask.
2. For each, mark which chunk(s) contain the answer.
3. For each candidate model, retrieve `k=5` and measure **recall@5** — did the right chunk show up?
4. Take the smallest / fastest model whose recall is acceptable.

This can be done with `RAGPipeline.search()` you already have — roughly 20 lines of script
in `tests/`.

### 6. Instruction protocol

Some models expect **queries** to be encoded differently from **documents**:

| Model | Query handling |
|---|---|
| `all-MiniLM-L6-v2` | none — identical encoding for queries and documents |
| `BAAI/bge-*` | prepend a query prompt (e.g. `"Represent this sentence for searching relevant passages:"`) |
| `intfloat/multilingual-e5` | prefix `query: ` vs `passage: ` |
| `Qwen/Qwen3-Embedding` | `encode(query, prompt_name="query")` |

This codebase encodes everything identically (`vector_store.py:194`, `:402`, `:451`), so an
instruction-aware model needs its prompt wired in at the query call sites to reach full quality.

### 7. License and operational constraints

- MIT / Apache-2.0 (`all-MiniLM-L6-v2`, `bge-*`, `Qwen3`) → free to use commercially.
- Some models are gated behind a "request access" button on Hugging Face.
- Some require `trust_remote_code=True` (Qwen3) — a small code change in `vector_store.py:118`.
- **Dimension is fixed per Chroma collection.** Switching models usually changes the vector
  dimension, which means a new collection + re-ingestion (see below).

---

## Decision workflow

```
1. Filter by language            → English-only or multilingual?
2. Check context ≥ chunk size    → rules out models whose window is too small
3. Pick local vs API             → this project: local
4. Fit the compute budget        → smallest model whose speed is acceptable
5. Evaluate recall@5 on your own data
6. Wire up the model's query instruction (if any)
7. New collection + re-ingest (dimension changed)
```

## Comparison table

| Model | Dim | Max ctx | Params | Lang | License | Notes |
|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 256 | 22M | EN | MIT | Current default. Fast, but truncates 512-token chunks. |
| `BAAI/bge-base-en-v1.5` | 768 | 512 | 109M | EN | MIT | Natural upgrade: fits chunk size exactly, better quality, still CPU-friendly. |
| `BAAI/bge-m3` | 1024 | 8192 | 568M | multi | MIT | Multilingual + huge context, ~5× slower than MiniLM on CPU. |
| `intfloat/multilingual-e5-large` | 1024 | 512 | 560M | multi | MIT | Strong multilingual baseline; needs `query:` / `passage:` prefixes. |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | 32768 | 0.6B | multi | Apache-2.0 | Longest context; needs `trust_remote_code=True`. |
| `text-embedding-3-small` (API) | 1536 (MRL ↓512) | 8191 | — | multi | API | No local compute; costs per token; data leaves your machine. |

## Shortlist for this project

- **English, CPU-bound** → `BAAI/bge-base-en-v1.5` — matches the 512-token chunks, no
  `trust_remote_code`, modest speed cost.
- **Multilingual / robust** → `BAAI/bge-m3` — best quality-per-hassle trade-off.
- **Latest-gen or very long contexts** → `Qwen/Qwen3-Embedding-0.6B` — only if the speed hit
  and the `trust_remote_code` change are acceptable.

## Switching models in this codebase

1. Set the env var — the model is read from `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`)
   in `api/main.py:180`, threaded through `RAGPipeline` → `ChromaVectorStore`.
   ```ini
   # .env
   EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
   ```
2. **Dimension change → new collection.** The existing `business_documents` collection holds
   384-dim vectors. A different model almost always changes the dimension, and a Chroma
   collection has one fixed dimension. Use a new `COLLECTION_NAME` (or `clear_collection()`)
   and **re-ingest** — already-stored embeddings cannot be re-embedded in place.
3. **First run re-downloads** the weights into the model cache (`<persist_dir>/model_cache`,
   i.e. `data/chromadb/model_cache` by default) — the larger the model, the longer this takes.
4. If the model is instruction-aware (Qwen3, bge, e5), wire its query prompt into the
   `encode()` calls at `vector_store.py:402` / `:451`. Hardcoding a prompt breaks models
   that don't define one, so gate it on the model name.

---

**Bottom line:** an embedding model is not better "in the abstract" — it is better for your
language, your chunk size, your hardware, your privacy posture, and your measured retrieval
recall. For this exact codebase the two constraints that matter most are **context window ≥
chunk size** (currently violated by the default) and **CPU inference speed**.
