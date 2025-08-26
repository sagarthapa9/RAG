"""
LLM-integrated RAG pipeline using LangChain + AzureChatOpenAI
Works on top of the legacy Simple RAGPipeline (Chroma-backed).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import logging

from rag.pipeline import RAGPipeline  # your legacy class from the snippet
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


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


class AzureLLMConfig:
    """Reads Azure OpenAI config from environment with sensible defaults."""
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_AZURE_ENDPOINT")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_AZURE_DEPLOYMENT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-02-01")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(max_tokens)))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not (self.api_key and self.azure_endpoint and self.deployment):
            missing = [k for k, v in {
                "AZURE_OPENAI_API_KEY": self.api_key,
                "AZURE_OPENAI_ENDPOINT": self.azure_endpoint,
                "AZURE_OPENAI_DEPLOYMENT": self.deployment,
            }.items() if not v]
            raise ValueError(
                f"Azure OpenAI configuration incomplete. Missing: {', '.join(missing)}"
            )


class LLMRAGPipeline:
    """
    Composition: uses your Simple RAGPipeline for retrieval + AzureChatOpenAI for generation.
    """

    def __init__(
        self,
        retriever: Optional[RAGPipeline] = None,
        llm_config: Optional[AzureLLMConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        self.retriever = retriever or RAGPipeline()
        self.llm_config = llm_config or AzureLLMConfig()
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.llm_config.azure_endpoint,
            api_key=self.llm_config.api_key,
            deployment_name=self.llm_config.deployment,
            model=self.llm_config.model,  # harmless for Azure; primary is deployment_name
            api_version=self.llm_config.api_version,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )

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

        logger.info("LLMRAGPipeline initialized (retriever = Simple RAG, LLM = AzureChatOpenAI)")

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
    def answer(
        self,
        query: str,
        *,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        llm_overrides: Optional[Dict[str, Any]] = None,
    ) -> RAGAnswer:
        """
        Retrieve top-k chunks and ask AzureChatOpenAI to answer with grounding.
        """
        # 1) Retrieve
        hits = self.retriever.search(query, k=k, filter_metadata=filter_metadata) or []
        context, sources = self._build_context_and_sources(hits)

        # 2) Prepare prompt
        messages = self.prompt.format_messages(
            system_prompt=self.system_prompt,
            question=query,
            context=context if context.strip() else "(no relevant context retrieved)"
        )

        # 3) Optional LLM param overrides (temperature, max_tokens, etc.)
        llm = self.llm
        if llm_overrides:
            llm = llm.bind(**llm_overrides)

        # 4) Invoke
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
            model=getattr(resp, "model", None) or self.llm_config.deployment,
        )
