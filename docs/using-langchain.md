# Using LangChain in this project

LangChain is the library most RAG tutorials reach for first, so it's worth being precise about
what it actually is, what this project uses it for, and what it deliberately does *not* use it for.

The short version: **this project uses LangChain as a thin interop layer, not as a framework.**
It supplies the LLM client, the prompt abstraction, and a common `Document` type. Everything that
makes this system a *RAG* system — reading, chunking, embedding, storing, retrieving — is
hand-written in `src/rag/`.

---

## LangChain is a stack of packages, not one library

Installing "LangChain" pulls in a dozen Python distributions. The ones relevant here:

| Package | Installed | What it is | Used by our code? |
|---|---|---|---|
| `langchain-core` | 1.6.1 | Base interfaces everything depends on — `Document`, messages, prompts, runnables | ✅ **yes** (the important one) |
| `langchain-openai` | 1.6.0 | Provider integration wrapping the `openai` SDK (3.6.0) into LangChain's interface | ✅ yes |
| `langchain` | 1.3.18 | The umbrella package — chains, agents, LCEL glue | ❌ installed but never imported |
| `langchain-community` | 0.4.2 | Community third-party integrations | ❌ never imported |
| `langchain-classic`, `langchain-protocol` | 1.x | Absorbed legacy modules / protocol definitions | ❌ transitive only |
| `langchain-text-splitters` | 1.1.2 | Text splitting utilities | ❌ we use our own chunker |

> **Version drift worth knowing:** `pyproject.toml` declares `langchain>=0.2.0` and
> `langchain-openai>=0.1.0`, but the resolved install is **1.x**. The loose pins allow the whole
> 1.0 major line. If a reinstall ever behaves differently, this is the first place to look.

The takeaway: `langchain-core` is the piece that matters — it's really just "a set of standard
interfaces for LLM applications." The rest are integrations and conveniences.

---

## What this project actually imports

All of it — five import lines across three modules:

```python
# The data type that flows through every stage
from langchain_core.documents import Document          # pipeline.py:10, vector_store.py:14, llm_rag.py:19

# To build our own retriever against LangChain's interface
from langchain_core.retrievers import BaseRetriever    # vector_store.py:15
from langchain_core.callbacks import CallbackManagerForRetrieverRun   # vector_store.py:16

# The provider-agnostic chat model
from langchain_openai import AzureChatOpenAI, ChatOpenAI   # llm_rag.py:18

# The prompt template
from langchain_core.prompts import ChatPromptTemplate  # llm_rag.py:20
```

Four things, and that's the whole surface area:

1. **`Document`** — the currency of the pipeline: whatever the reader emits, the chunker splits,
   the store persists, and the retriever returns is a `Document` with `.page_content` and
   `.metadata`. It's why every stage can hand off to the next without custom glue types.
2. **`ChatOpenAI` / `AzureChatOpenAI`** — the object exposing `.invoke()`, `.astream()`, `.bind()`.
   Because both classes present the same interface, swapping DeepSeek for OpenAI or Azure is a
   *config* change (`LLMConfig`), not a rewrite.
3. **`ChatPromptTemplate`** — the reusable system/human prompt blueprint (`llm_rag.py:246`).
4. **`BaseRetriever`** — the abstract base `ChromaRetriever` subclasses so it looks LangChain-shaped
   from the outside, while the actual search is our own ChromaDB code.

> Note what's *absent*: there is no `langchain-chroma`, no `VectorStore` base, no LCEL chain
> (`prompt | llm | parser`), no agent, no memory, no output parser, no document loader, no text
> splitter. `ChromaVectorStore` talks to the raw `chromadb` client directly.

---

## LangChain is generic — it knows nothing about RAG

A useful reality check, because it's easy to assume the framework understands "context" and
"retrieval." `ChatPromptTemplate` does not. It knows exactly two things: the **roles**
(`system`/`human`) and the **literal `{placeholder}` names it finds by scanning its own strings**.

Probing the real template used in this project (`llm_rag.py:246-249`):

```
template input_variables: ['context', 'question', 'system_prompt']
rendered types: ['SystemMessage', 'HumanMessage']
extra key k=5  -> accepted (silently ignored)
missing context -> KeyError | 'context'
```

Two things worth reading off that output:

- **`k` is silently ignored if passed.** It's not in `input_variables`, because `k` is a
  *retrieval* parameter consumed by `retriever.search(query, k=k)` (`llm_rag.py:321`). Its only
  influence on the prompt is indirect: it decides how many chunks end up inside the `{context}`
  string. (`context` is the only template variable that participates in this negotiation.)
- **The template receives a `str`, not a list of chunks.** By the time `format_messages` is called,
  `_build_context_and_sources` (`llm_rag.py:283`) has already flattened `List[Document]` into one
  string via `"\n\n".join(context_lines)`. Nothing downstream knows those were ever documents.

Rename `{context}` to `{banana}` and LangChain wouldn't notice — only our own call site
(`llm_rag.py:326`) would break. Same at the next layer: `llm.invoke(messages)` is equally
agnostic; the model just sees a system message and a user message.

**So what makes this RAG?** Not the prompt library — the orchestration around it:

```
retrieve (RAGPipeline.search)  →  flatten to text (_build_context_and_sources)
       →  stuff into a generic template (ChatPromptTemplate)  →  call a generic LLM (invoke)
```

Swap `ChatPromptTemplate` for an f-string and this is still RAG. Remove the retrieval step and it's
just an LLM reading a long prompt. The framework is plumbing; the retrieval is the RAG.

---

## What the LLM actually receives — and returns

If the framework is agnostic about RAG, the *model* is even more so. Here is the wire-level view of
one request:

```
POST /api/qa/ask → answer(query) → build_messages → llm.invoke(messages) → RAGAnswer → QuestionResponse
                    llm_rag.py:330   llm_rag.py:308   llm_rag.py:350        llm_rag.py:356  main.py:479
```

### The input: two messages, one stateless turn

`build_messages()` produces a list of exactly **two** messages (`llm_rag.py:323`):

```
[system]  You are a helpful AI assistant that answers questions based on business documents
          and company information.  Use ONLY the information from the provided context ...
                                              (the full prompt lives at main.py:299-312)

[human]   Question: What are the ongoing charges for the Vanguard Japan fund?

          Context:
          [1] Vanguard Japan Stock Index Fund EUR Acc [IE0007286036] | Ongoing charges 0.16% ...
          [2] Vanguard Japan Stock Index Fund EUR Acc [IE0007286036] | ...
          [3] Vanguard US 500 Stock Index Fund EUR Acc [IE00B3XXRP09] | Ongoing charges 0.07% ...

          Format your answer first, then list sources.
```

Note what is **not** in there:

- **No conversation history.** Every request is a fresh, independent turn — the model has no memory of
  the previous question. The "conversation" is the context block alone.
- **No tools, no retrieval access.** The model cannot search. It can only read what was pasted in
  front of it.
- **No corpus.** It has never seen the 521 fund documents. It sees the handful of chunks retrieval
  chose, as plain text, numbered `[1]`…`[k]` — the numbering is what lets the `sources` list in the
  response line up with the context.

`k` is what controls how much lands in that block: `retriever.search(query, k=k)`
(`llm_rag.py:321`). It never appears in the prompt itself — it only changes how many lines the
`Context:` section has.

### The output: one AIMessage

`llm.invoke(messages)` (`llm_rag.py:350`) returns a single `AIMessage`:

| Field | Contents | Becomes |
|---|---|---|
| `.content` | the answer text | `RAGAnswer.answer` |
| `.usage_metadata` | `input_tokens` / `output_tokens` / `total_tokens` | token counts in the API response |
| `.response_metadata` / `.model` | model name | `RAGAnswer.model` |

`answer()` repackages that into the `RAGAnswer` dataclass (`llm_rag.py:356`) — attaching the `sources`
list it built during retrieval — and the endpoint wraps it in `QuestionResponse`
(`main.py:469-484`).

**The token counts are the tell.** `input_tokens` covers *system + question + context* — the entire
prompt — which makes it a cheap, direct read on how much context you actually sent. With `k=5` and
~200-token chunks, the context block dominates the input. `output_tokens` is just the answer. Those
two numbers are the fastest sanity check on your retrieval settings.

### Streaming: same input, different output shape

`/api/qa/ask/stream` builds the **identical** `messages` list, then calls `llm.astream(...)`
(`llm_rag.py:392`) instead of `invoke`. The model still generates one response, but it arrives as a
sequence of `AIMessageChunk` deltas that are forwarded as SSE `token` events, with the usage numbers
attached to the final chunk. Same prompt in; the only difference is how the output is consumed.

### Why this framing matters

The answer can only ever be as good as the context block. If the right chunk is missing from the
top-k, **no prompt engineering recovers it** — the model is instructed to answer only from the
context, and it genuinely cannot see anything else. That is exactly why this project's hardest bug
(the Vanguard "ongoing charges" chunk not surfacing) was a *retrieval* bug, fixed by enriching chunk
text at ingest — not a prompt bug.

Put another way: **neither LangChain nor the LLM knows it is part of a RAG system.** The RAG awareness
lives entirely in our orchestration — `answer()` is the place where a pile of retrieved text becomes
a grounded answer.

---

## What a fully "LangChain-ified" RAG would look like

Every stage *does* have an off-the-shelf component — this is LangChain's core pitch, and it's true:

| RAG stage | LangChain component | Package |
|---|---|---|
| **Load** | `PyPDFLoader`, `Docx2txtLoader`, `DirectoryLoader` | `langchain-community` |
| **Split** | `RecursiveCharacterTextSplitter`, `TokenTextSplitter` | `langchain-text-splitters` |
| **Embed** | `HuggingFaceEmbeddings`, `OpenAIEmbeddings` | `langchain-huggingface` / `langchain-openai` |
| **Store** | `Chroma`, `FAISS`, `PineconeVectorStore` | one package **per backend**: `langchain-chroma`, … |
| **Retrieve** | `.as_retriever()`, `EnsembleRetriever`, `MultiQueryRetriever`, `ParentDocumentRetriever`, `SelfQueryRetriever` | `langchain` / `langchain-classic` |
| **Prompt** | `ChatPromptTemplate` | `langchain-core` ← **we use this** |
| **Generate** | `ChatOpenAI`, `ChatAnthropic`, `ChatOllama` | `langchain-openai` etc. ← **we use this** |
| **Parse** | `StrOutputParser`, `PydanticOutputParser` | `langchain-core` |
| **Orchestrate** | LCEL (`prompt \| llm \| parser`), agents | `langchain` |

We could replace `document_reader.py`, `chunking.py`, and `vector_store.py` wholesale with loaders,
splitters, and `langchain-chroma`, and the API layer would barely change.

---

## The trade-offs

### Package-per-integration, and version churn

Notice the store row: **each backend is its own package.** Going full-LangChain means installing
`langchain-chroma`, `langchain-huggingface`, `langchain-text-splitters`, `langchain-community` — each
versioned independently, released on its own schedule, with **import paths that move between packages
across versions.** `HuggingFaceEmbeddings`, for instance, migrated from `langchain-community` to
`langchain-huggingface`; classes shuffle between `langchain`, `langchain-community`, and
`langchain-classic`. This is LangChain's most common real-world complaint, and it's why trusting a
tutorial's import path is risky — check the docs for the version you actually installed.

### Abstraction costs control and visibility

- `RecursiveCharacterTextSplitter`'s chunk boundaries are the library's opinion, not yours.
- `.as_retriever()` hides what happens inside — which distance metric, how a metadata filter is
  translated to the backend, whether it re-ranks.

For a **learning** project that's a real loss. The single hardest bug in this project — the Vanguard
Japan fund's "ongoing charges" chunk not surfacing — lived in exactly the layer an abstraction would
have hidden. Debugging it required seeing the chunk text, its metadata, and its cosine score.

### Abstractions leak

"Swap the vector store by changing one class" is rarely true in practice: metadata filter syntax,
scoring semantics, and pagination differ per backend. The abstraction covers the common case and
leaks on everything else.

---

## What this project chose, and why

| RAG stage | Who does it here | LangChain involved? |
|---|---|---|
| Read PDF/DOCX/TXT | `document_reader.py` | ❌ hand-written |
| Chunk | `chunking.py` (tiktoken) | ❌ hand-written |
| Embed | `sentence-transformers` directly | ❌ not LangChain |
| Store / retrieve | `vector_store.py` over the raw `chromadb` client | ❌ hand-written |
| Prompt | `ChatPromptTemplate` | ✅ |
| Generate | `ChatOpenAI` | ✅ |

This is a deliberate split: **hand-roll the stages that define what RAG does; take the library where
it's genuine interop value.** The retrieval layer is where answer quality is actually won or lost,
so it's owned in-house; the model-client layer is where provider churn would hurt, so it's abstracted
behind `LLMConfig` + the LangChain client interface.

That mirrors a fairly common production shape: abstract the LLM client, own the retrieval.

---

## Rule of thumb

- **Prototyping / throwaway:** use LangChain components everywhere — fastest path to a working demo.
- **Learning:** hand-roll the core. You understand this pipeline in a way that calling
  `.as_retriever()` would not have taught.
- **Production:** the split already in place here — abstract the LLM client, own retrieval — is a
  sound default.

## Alternatives worth knowing

- **LlamaIndex** — more RAG-focused; stronger retrieval/indexing abstractions out of the box.
- **Haystack** — explicit, node-based pipelines; less magic, more wiring.
- **Raw SDKs** — the `openai` client plus your own vector-store code. That is essentially what this
  project is, minus the `Document` type and `ChatOpenAI`.

---

**Status:** `data/` is untracked; this document is project documentation only — no code changes.
