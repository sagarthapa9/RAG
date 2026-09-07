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
this project has already tripped over.

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

> ⚠️ This is the criterion that bit this project: the chunker used to make **512-token**
> chunks (default `CHUNK_SIZE=512`) while `all-MiniLM-L6-v2` only reads **256 tokens** — so
> half of every chunk was silently discarded at embed time. The default is now
> `CHUNK_SIZE=200`, sized to stay inside the model's window. `chunk_overlap` does not fix
> truncation — it only softens cuts at chunk boundaries.

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

> Before blaming the model, classify the miss: if the right chunk exists but fails to rank,
> that may be a **hard-constraint** problem (right fund, wrong scope) rather than a semantic
> one — which no embedding model fixes. That's the metadata side of retrieval; see
> [Metadata: the half of retrieval an embedding model can't do](#metadata-the-half-of-retrieval-an-embedding-model-cant-do).

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

## Metadata: the half of retrieval an embedding model can't do

An embedding model is only the **similarity** half of retrieval. Every stored chunk has two
parts, joined by its chunk `id`:

- **The embedding** (384 floats for MiniLM) = the chunk's *meaning*, used for ranking. It can
  only answer "which chunks *mean* the same as my query?"
- **The metadata** (a dict of scalars: `filename`, `page`, `start_char`, `fund_isin`, …) =
  the chunk's *facts and provenance*. It answers questions similarity physically cannot: "is
  this chunk *from* this document?", "is this chunk *allowed* for this user?"

Similarity is fuzzy; metadata is exact. A vector search has no idea that chunk #451 is the
Vanguard Japan fund while chunk #1,234 is a different fund that happens to share
byte-near-identical boilerplate. That exactness is why production RAG stores both, always.

### How the two halves combine at search time

A query goes through four steps, and each half is used at a different step:

```
Query: "What is the ongoing charges for the Japan fund?"
        │
        ▼
 1. Embed the query  ─────────────────────────────────►  query vector (384 floats)
        │
 2. If a metadata filter is supplied (e.g. filename = "...IE0007286036.pdf"):
        │
        ▼
    Chroma picks the candidate pool — only chunks whose metadata MATCHES ─┐
        │                                                                │
        ▼                                                                ▼
 3. Cosine similarity: query vector vs. the EMBEDDINGS of those chunks
        │
        ▼
 4. Return top-k of that pool, ranked by similarity
```

| Step | Uses | Decides |
|---|---|---|
| **Filter (`where`)** | **metadata** | who's *allowed* in the pool — inclusion/exclusion, no score |
| **Similarity (ranking)** | **embeddings** | order *within* the pool — the scores |

So the two halves meet at the chunk `id`: the filter produces a set of matching ids, and
similarity ranks within those ids. Concretely:

- **No filter** → the pool is the whole collection. The query vector is compared against
  every stored vector and you get the global top-k. This is what failed for the Japan fund:
  the right chunk existed, but 520 other funds' near-identical boilerplate out-ranked it, so
  it was not even in the top-200.
- **With a filter** → the pool shrinks to only the matching chunks (equality on `filename`
  gave 6 chunks for that PDF). The query vector is then compared *only within those 6*, so the
  charges chunk came back at **rank #1** — even though globally it was ~rank 800.

The metadata did not "help rank it higher" — it *removed the competition*. Three consequences
worth internalizing:

1. **A filter never adds a score; it only includes/excludes.** No "boost this chunk by +0.2
   because its metadata matches." A chunk that fails the `where` is simply not in the pool.
   Over-selective filters starve the LLM (see the caveats below).
2. **The query embedding is never compared to metadata.** Vectors compare only to vectors;
   metadata is consulted separately to decide the pool.
3. **Because the filter runs first, `k` counts *within the pool*.** `k=5` + a filter means 5
   chunks from that file, not 5 from the whole store — so scope by metadata with a `k` large
   enough to still answer. (What this is *not* — hybrid search — is unpacked below.)

### Filtered vector search vs. hybrid search

Filtering by metadata is **not** hybrid search — the terms get conflated, and the distinction
matters for what you build:

| Technique | What runs | How results combine |
|---|---|---|
| **Pure vector search** | one dense retriever (cosine) | rank by the one similarity score |
| **Filtered vector search** (this system) | one dense retriever, gated by a `where` constraint | metadata **excludes** non-matching chunks (no score); similarity ranks the rest |
| **Hybrid search** | **two score-producing retrievers** (e.g. BM25 keyword + dense vector) | both return ranked lists, then the two scores are **fused** |

The tell-tale question is: *does the second signal produce a score that gets blended, or is it
just a gate?*

- A metadata filter is a **gate**. It never scores anything — a chunk either passes `where`
  or it is out, and only one scoring signal (the vector) survives.
- Hybrid search runs **two gates and two scorers**, then merges the ranked lists with a fusion
  strategy — a weighted sum (`0.5·vector + 0.5·bm25`) or RRF (reciprocal rank fusion,
  `score = Σ 1/(k + rank)`), the usual no-tuning default.

The classic hybrid pair is **sparse + dense**, because each catches the other's misses:

- **Dense** (the embedding you have) captures *meaning* — synonyms, paraphrase. Weak at exact
  tokens, rare names, and codes.
- **Sparse / BM25** matches exact *strings* — `IE0007286036`, a precise fund name, product
  codes. It has no notion of meaning: a synonym that shares no token scores 0.

That is the flip side of the Japan-fund case: pure vector drowned the ISIN among 2,568
near-identical boilerplate chunks, but a BM25 retriever would have matched the exact string
instantly — and conversely a question phrased with synonyms would score 0 in BM25 yet rank #1
in vector.

For an exact code/figure that must be findable, there are three escalating options:

1. **Metadata filter** (scoping) — cheapest; the app knows the fund and injects `filename`.
2. **Enrichment** — bake the ISIN/name into the chunk *text* that gets embedded, so vector
   similarity itself can separate the funds (see the caveats below).
3. **Hybrid search** — add a BM25-style retriever so exact strings score independently; real
   added complexity (a second index + a fusion strategy), worth it only when queries genuinely
   mix "meaning questions" and "exact-code questions."

### What metadata is used for

1. **Hard constraints similarity can't express (filtering / scoping).** The killer use case —
   embed once into one collection, then slice it per query with a `where` filter:
   - *Access control / tenancy* — `where={"tenant_id": user.tenant}` so a query physically
     cannot return another tenant's or a confidential document's chunks. How most
     multi-tenant production RAG is built.
   - *Recency* — `where={"date": {"$gte": "2026-01-01"}}` to exclude stale documents.
   - *Domain narrowing* — `where={"filename": "Vanguard Japan ... [IE0007286036].pdf"}` to
     search one fund's KIID. The app/UI picks the scope; the user just types a question.
2. **Provenance & citations ("show your sources").** The Q&A `sources` list only exists
   because each chunk carries `filename` / `page` / char offsets — a path back to the real
   document, and a way to verify an answer.
3. **Lifecycle management.** Replace a corrected PDF? Delete only that document's old chunks
   (`collection.delete(where={"filename": old})`) before re-ingesting — with one shared
   collection, filename metadata is the only handle that isolates a document.
4. **Exact structured facts.** Numbers, dates, ISINs, enum values — things that must match
   *exactly*, not by meaning. `"ISIN IE0007286036"` is queryable with `==` as metadata; baked
   into prose it's just more words the embedding weighs loosely.
5. **Debugging & audit.** `chunk_id`, ingest timestamps, embedding-model version, char
   offsets — answerable "why did the LLM cite this?", and stale-chunk cleanup after a model
   swap.

### The two caveats (a production design must respect both)

**Metadata does not affect ranking.** It filters *before* or *after* the vector search; it
never moves a chunk up within the results. For a *ranking* problem — near-duplicate
boilerplate across many documents where the right chunk isn't even in the top-k — the real
fix is **enrichment**: bake the fund name / ISIN / key figure into the chunk *text* that gets
embedded, so similarity itself can separate them. Metadata and enrichment solve different
failures, and both are needed:

- metadata → *"I want only X's chunks"* (hard scoping, security, attribution)
- enrichment → *"the right chunk should rank highest"* (semantic disambiguation)

**Filters have sharp edges.** Over-selective pre-filters leave too few chunks to answer a
question; post-filters that drop most of the top-k silently starve the LLM. And filter
operators are *not* portable across vector DBs — chromadb's `$contains` on this build
silently returned 0 rows for a value that existed (equality worked). In production the system
injects filters from app context; users never type arbitrary `where` clauses.

### What this means for choosing an embedding model

Model quality and metadata fix *different* retrieval failures. A bigger, "better" embedding
model will **not** fix a miss caused by a missing filter scope (wrong document, wrong tenant,
wrong date) — spending compute on a 0.6B model to solve a metadata problem is the wrong
investment. Evaluate retrieval quality on your own data first, and classify each miss before
upgrading the model:

| Type of miss | Real fix |
|---|---|
| Semantic — right scope, but meaning not captured | Better embedding model, better chunking |
| Ranking — near-duplicate text pushes the target out of top-k | **Enrichment** — bake discriminating facts into the chunk text |
| Scope — right facts, wrong document / tenant / date | **Metadata filter** injected by the app |
| Both scope + ranking | Metadata filter + enrichment together |

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
| `all-MiniLM-L6-v2` | 384 | 256 | 22M | EN | MIT | Current default. Fast; truncates chunks longer than 256 tokens. |
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
chunk size** (the default chunk size is now 200, inside MiniLM's 256-token window) and
**CPU inference speed**.
