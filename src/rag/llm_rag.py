"""
LLM-integrated RAG pipeline using LangChain + a configurable chat model.

The LLM provider is chosen via the ``LLM_PROVIDER`` env var (default: ``kimi``);
see ``LLMConfig`` / ``PROVIDERS`` for the supported set. Works on top of the
legacy Simple RAGPipeline (Chroma-backed).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import asyncio
import os
import logging

from rag.pipeline import RAGPipeline
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


def _chunk_text_delta(content: Any) -> str:
    """Extract the text delta from an AIMessageChunk.content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            else:  # ContentBlock-like object
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    return ""


@dataclass
class RAGAnswer:
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: Optional[str] = None

    def dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderSpec:
    """Static defaults + env-var conventions for one LLM provider."""

    key_envs: Tuple[str, ...] = ()
    base_url: Optional[str] = None
    model: Optional[str] = None
    key_required: bool = True
    required_env: Tuple[str, ...] = ()


PROVIDERS: Dict[str, ProviderSpec] = {
    # Generic OpenAI — base URL / model are overridable via LLM_BASE_URL / LLM_MODEL.
    "openai": ProviderSpec(
        key_envs=("OPENAI_API_KEY",),
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    ),
    # Kimi / Moonshot AI — OpenAI-compatible. China endpoint by default;
    # use https://api.moonshot.ai/v1 for the international platform.
    "kimi": ProviderSpec(
        key_envs=("KIMI_API_KEY",),
        base_url="https://api.moonshot.cn/v1",
        model="moonshot-v1-8k",
    ),
    "deepseek": ProviderSpec(
        key_envs=("DEEPSEEK_API_KEY",),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    ),
    # Local models served by Ollama's OpenAI-compatible endpoint (no key needed).
    "ollama": ProviderSpec(
        key_envs=("OLLAMA_API_KEY",),
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        key_required=False,
    ),
    # Azure OpenAI — built with AzureChatOpenAI (endpoint + deployment required).
    "azure": ProviderSpec(
        key_envs=("AZURE_OPENAI_API_KEY",),
        model="gpt-4o-mini",
        required_env=("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"),
    ),
    # Any other OpenAI-compatible service: set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL.
    "custom": ProviderSpec(
        key_envs=("LLM_API_KEY",),
    ),
}


def _first_set(*env_names: str) -> Optional[str]:
    """Return the value of the first environment variable that is set (or None)."""
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value
    return None


class LLMConfig:
    """
    Provider-agnostic LLM configuration read from the environment.

    The provider is selected by ``LLM_PROVIDER`` (default ``kimi``, to stay
    compatible with existing setups). Each provider has its own env vars (see
    ``PROVIDERS``) but also honors the generic ``LLM_API_KEY`` /
    ``LLM_BASE_URL`` / ``LLM_MODEL`` overrides, so switching providers usually
    means just changing ``LLM_PROVIDER`` (and adding that provider's key).

    Supported providers: openai, kimi, deepseek, ollama, azure, custom.
    """

    DEFAULT_PROVIDER = "kimi"

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.provider = (
            provider or os.getenv("LLM_PROVIDER") or self.DEFAULT_PROVIDER
        ).lower().strip()
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER={self.provider!r}. Valid options: "
                f"{', '.join(sorted(PROVIDERS))}."
            )
        spec = PROVIDERS[self.provider]
        self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(max_tokens)))

        # API key: explicit arg > provider-specific env > generic LLM_API_KEY.
        self.api_key = api_key or _first_set(*spec.key_envs, "LLM_API_KEY")
        if spec.key_required and not self.api_key:
            raise ValueError(
                f"LLM_PROVIDER={self.provider!r} needs an API key. Set "
                f"{' or '.join(spec.key_envs)} (or the generic LLM_API_KEY) "
                f"in your .env (see .env.example)."
            )

        if self.provider == "azure":
            # Azure stores the endpoint/deployment separately from the model.
            self.base_url = os.getenv("AZURE_OPENAI_ENDPOINT") or base_url
            self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            self.model = model or os.getenv("LLM_MODEL") or self.deployment or spec.model
            missing = [name for name in spec.required_env if not os.getenv(name)]
            if missing:
                raise ValueError(
                    f"LLM_PROVIDER=azure needs {', '.join(missing)} in your .env "
                    f"(see .env.example)."
                )
        else:
            # OpenAI-compatible providers: provider-specific env > generic > default.
            self.base_url = (
                base_url
                or _first_set(f"{self.provider.upper()}_BASE_URL", "LLM_BASE_URL")
                or spec.base_url
            )
            self.model = (
                model
                or _first_set(f"{self.provider.upper()}_MODEL", "LLM_MODEL")
                or spec.model
            )
            self.deployment = None
            self.api_version = None
            if self.base_url is None:
                raise ValueError(
                    f"LLM_PROVIDER={self.provider!r} needs LLM_BASE_URL in your .env."
                )
            if self.model is None:
                raise ValueError(
                    f"LLM_PROVIDER={self.provider!r} needs LLM_MODEL in your .env."
                )

    def create_llm(self):
        """Build the LangChain chat model for the configured provider."""
        if self.provider == "azure":
            return AzureChatOpenAI(
                azure_endpoint=self.base_url,
                api_key=self.api_key,
                deployment_name=self.deployment,
                api_version=self.api_version,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return ChatOpenAI(
            # Providers without keys (e.g. Ollama) use the standard "not-needed"
            # placeholder accepted by OpenAI-compatible endpoints.
            api_key=self.api_key or "not-needed",
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class LLMRAGPipeline:
    """
    Composition: uses your Simple RAGPipeline for retrieval + a provider-agnostic
    chat model (provider chosen via LLM_PROVIDER, default: Kimi / Moonshot AI) for
    generation.
    """

    def __init__(
        self,
        retriever: Optional[RAGPipeline] = None,
        llm_config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        self.retriever = retriever or RAGPipeline()
        self.llm_config = llm_config or LLMConfig()
        self.llm = self.llm_config.create_llm()

        # Simple, grounded system prompt
        self.system_prompt = system_prompt or (
            "You are a precise assistant. Answer the user's question using ONLY the provided context. "
            "If the answer cannot be found in the context, say you don't know. Keep answers concise."
        )

        # LangChain chat prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nFormat your answer first, then list sources.")
        ])

        logger.info(
            f"LLMRAGPipeline initialized (retriever = Simple RAG, LLM = "
            f"provider={self.llm_config.provider}, model={self.llm_config.model}, "
            f"base_url={self.llm_config.base_url})"
        )

    # def _build_context_and_sources(
    #     self, hits: Sequence[Dict[str, Any]]
    # ) -> Tuple[str, List[Dict[str, Any]]]:
    #     """
    #     Combine retrieved chunks into a single context string and normalized sources.
    #     Expects hits from RAGPipeline.vector_store.similarity_search(...)
    #     """
    #     context_lines: List[str] = []
    #     sources: List[Dict[str, Any]] = []

    #     for idx, h in enumerate(hits, start=1):
    #         text = h.get("text") or h.get("page_content") or ""
    #         meta = h.get("metadata") or {}
    #         source = {
    #             "rank": idx,
    #             "score": h.get("score"),
    #             "filename": meta.get("filename") or meta.get("source") or meta.get("file_path"),
    #             "page": meta.get("page"),
    #             "id": meta.get("id") or meta.get("doc_id"),
    #         }
    #         context_lines.append(f"[{idx}] {text}")
    #         sources.append(source)

    #     return "\n\n".join(context_lines), sources
    

    def _build_context_and_sources(
        self, hits: Sequence[Document]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Combine retrieved chunks into a single context string and normalized sources.
        Expects hits from RAGPipeline.vector_store.similarity_search(...).
        """
        context_lines: List[str] = []
        sources: List[Dict[str, Any]] = []

        for idx, h in enumerate(hits, start=1):
            # Access attributes directly from the Document object
            text = h.page_content  # Use `page_content` for the document text
            meta = h.metadata  # Use `metadata` for the document metadata
            source = {
                "rank": idx,
                "score": meta.get("score"),  # Ensure metadata contains "score"
                "filename": meta.get("filename") or meta.get("source") or meta.get("file_path"),
                "page": meta.get("page"),
                "id": meta.get("id") or meta.get("doc_id"),
            }
            context_lines.append(f"[{idx}] {text}")
            sources.append(source)

        return "\n\n".join(context_lines), sources
    def build_messages(
        self,
        query: str,
        *,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List, List[Dict[str, Any]]]:
        """
        Retrieve top-k chunks and format the prompt. Returns (messages, sources).

        Split out of answer() so the streaming endpoint can run retrieval before
        the response starts (retrieval failures then surface as normal HTTP errors).
        """
        hits = self.retriever.search(query, k=k, filter_metadata=filter_metadata) or []
        context, sources = self._build_context_and_sources(hits)
        messages = self.prompt.format_messages(
            system_prompt=self.system_prompt,
            question=query,
            context=context if context.strip() else "(no relevant context retrieved)",
        )
        return messages, sources

    def answer(
        self,
        query: str,
        *,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        llm_overrides: Optional[Dict[str, Any]] = None,
    ) -> RAGAnswer:
        """
        Retrieve top-k chunks and ask the chat model to answer with grounding.
        """
        # 1) Retrieve + prepare prompt (shared with the streaming path)
        messages, sources = self.build_messages(query, k=k, filter_metadata=filter_metadata)

        # 2) Optional LLM param overrides (temperature, max_tokens, etc.)
        llm = self.llm
        if llm_overrides:
            llm = llm.bind(**llm_overrides)

        # 3) Invoke
        resp = llm.invoke(messages)

        # 5) Usage & metadata (best-effort; depends on provider returning these fields)
        usage = getattr(resp, "usage_metadata", None) or {}
        answer_text = resp.content if hasattr(resp, "content") else str(resp)

        return RAGAnswer(
            query=query,
            answer=answer_text.strip(),
            sources=sources,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            model=getattr(resp, "model", None) or self.llm_config.model,
        )

    async def answer_stream(
        self,
        messages: List,
        sources: List[Dict[str, Any]],
        *,
        llm_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Stream answer token deltas + usage from the LLM.

        Yields (event, payload) pairs:
          ("token", {"text": <delta>})  -- one per chunk
          ("error", {"detail": ...})    -- mid-stream failure, then generator ends
          ("done", {model, prompt_tokens, completion_tokens, total_tokens, sources_count})
        The caller emits the leading "sources" event (sources come from build_messages()).
        """
        llm = self.llm
        if llm_overrides:
            llm = llm.bind(**llm_overrides)

        usage: Optional[Dict[str, Any]] = None
        model: Optional[str] = None

        try:
            # stream_usage=True is required for usage metadata on OpenAI-compatible
            # providers with custom base URLs (self.stream_usage stays None there).
            async for chunk in llm.astream(messages, stream_usage=True):
                if model is None:
                    model = (chunk.response_metadata or {}).get("model_name") or getattr(
                        chunk, "model", None
                    )
                if getattr(chunk, "usage_metadata", None):
                    usage = chunk.usage_metadata  # last non-None wins (final total)
                text = _chunk_text_delta(getattr(chunk, "content", None))
                if text:
                    yield ("token", {"text": text})
        except asyncio.CancelledError:
            logger.info("LLM stream cancelled (client disconnected)")
            raise
        except Exception as e:
            logger.error("LLM streaming error: %s", e)
            yield ("error", {"detail": f"Answer streaming failed: {str(e)}"})
            return
        finally:
            logger.info("Stream finished for %d sources", len(sources))

        yield ("done", {
            "model": model or self.llm_config.model,
            "prompt_tokens": (usage or {}).get("input_tokens"),
            "completion_tokens": (usage or {}).get("output_tokens"),
            "total_tokens": (usage or {}).get("total_tokens"),
            "sources_count": len(sources),
        })
