"""Community evidence search connector.

Pipeline: DuckDuckGo search → Playwright page reading → LLM content analysis.

Each stage is traced as a separate MLflow span under a shared parent,
using the same session_id as the downstream recommendation call.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from domain.evidence import EvidenceItem

logger = logging.getLogger(__name__)

VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() not in ("false", "0", "no")

PREFERRED_DOMAINS = [
    "huggingface.co",
    "docs.vllm.ai",
    "github.com/vllm-project",
    "arxiv.org",
    "access.redhat.com",
    "blog.vllm.ai",
]

_QUERY_TEMPLATES = {
    "compatibility": "{repo_id} vLLM compatibility issues deployment",
    "performance": "{repo_id} vLLM benchmark throughput latency GPU",
    "deployment": "{repo_id} vLLM deployment guide tensor parallel configuration",
}

_EVIDENCE_TYPE_TO_CLAIM: dict[str, str] = {
    "compatibility": "compatibility",
    "performance": "serving_performance",
    "deployment": "tested_hardware",
}

_OFFICIAL_DOMAINS = {
    "access.redhat.com",
    "docs.vllm.ai",
    "blog.vllm.ai",
    "docs.redhat.com",
}

_OFFICIAL_SECONDARY_DOMAINS = {
    "huggingface.co",
    "arxiv.org",
    "github.com",
    "pytorch.org",
    "nvidia.com",
    "developer.nvidia.com",
}

# Only read pages from these domains with Playwright.
# Other domains are kept as snippet-only evidence (no page reading).
_READABLE_DOMAINS = {
    "huggingface.co",
    "docs.vllm.ai",
    "blog.vllm.ai",
    "recipes.vllm.ai",
    "github.com",
    "arxiv.org",
    "access.redhat.com",
    "docs.redhat.com",
    "pytorch.org",
    "nvidia.com",
    "developer.nvidia.com",
    "rocm.docs.amd.com",
    "cloud.google.com",
    "aws.amazon.com",
    "learn.microsoft.com",
    "simplismart.ai",
    "anyscale.com",
    "modal.com",
    "bentoml.com",
    "medium.com",
    "towardsdatascience.com",
    "wandb.ai",
}

# Domains that are never relevant to ML/AI deployment evidence.
_BLOCKED_DOMAINS = {
    "answers.com", "quora.com", "reddit.com",
    "rocketreach.co", "leadiq.com", "zoominfo.com",
    "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "youtube.com",
    "amazon.com", "ebay.com", "walmart.com",
    "wikipedia.org",  # too generic
    "pinterest.com", "yelp.com",
}

_MAX_TOTAL_RESULTS = 9
_DDG_LIB_TIMEOUT_SECONDS = 20
_DDG_TIMEOUT_SECONDS = 10
_INTER_QUERY_DELAY_SECONDS = 0.8
_MAX_PAGES_TOTAL = 3

_LLM_ANALYSIS_SYSTEM = """\
You are a technical analyst. Given a model repo ID, evidence type, and web page content, \
extract a concise factual summary (2-4 sentences) focused on the evidence type.

Evidence types:
- compatibility: vLLM support status, known issues, required flags
- performance: throughput, latency, tokens/s benchmarks on specific GPUs
- deployment: tensor parallelism config, recommended GPU count, vLLM flags

Rules:
- Only state facts present in the content. Do not speculate.
- Include specific numbers (throughput, latency, GPU type) when available.
- If the content is irrelevant to the evidence type, respond with "NOT_RELEVANT".
"""

_LLM_ANALYSIS_USER = """\
Model: {repo_id}
Evidence type: {evidence_type}
Source URL: {url}

Page content:
{content}

Extract a factual summary for this evidence type."""

# MLflow helpers
_mlflow_available = False
try:
    import mlflow
    _mlflow_available = True
except ImportError:
    pass


class CommunitySearchConnector:
    """Discovers community evidence: DuckDuckGo → Playwright → LLM analysis."""

    def __init__(
        self,
        on_screenshot: Callable[[str, str, str], Any] | None = None,
    ):
        self._on_screenshot = on_screenshot

    async def search_model_evidence(
        self,
        repo_id: str,
        evidence_types: list[str] | None = None,
        max_results_per_type: int = 3,
        session_id: str = "",
    ) -> list[EvidenceItem]:
        """Search for community evidence about a model.

        Pipeline per query:
          1. DuckDuckGo search → URLs + snippets
          2. Playwright → read top page content
          3. LLM → analyze content and produce structured summary

        All three stages are traced as MLflow spans under one parent span.
        """
        types = evidence_types or list(_QUERY_TEMPLATES.keys())
        types = [t for t in types if t in _QUERY_TEMPLATES]
        if not types:
            logger.warning("No valid evidence types requested for %s", repo_id)
            return []

        # Check tracing availability
        should_trace = _mlflow_available
        if should_trace:
            try:
                from backend.tracing import is_tracing_enabled
                should_trace = is_tracing_enabled()
            except ImportError:
                should_trace = False

        import contextlib
        parent_cm: Any = contextlib.nullcontext()
        if should_trace:
            try:
                parent_cm = mlflow.start_span(
                    name="community_evidence_search", span_type="CHAIN",
                )
            except Exception:
                parent_cm = contextlib.nullcontext()

        t0 = time.monotonic()
        seen_urls: set[str] = set()
        evidence: list[EvidenceItem] = []
        pages_read_total = 0

        with parent_cm as parent_span:
            if parent_span is not None:
                try:
                    parent_span.set_inputs({
                        "repo_id": repo_id,
                        "evidence_types": types,
                        "session_id": session_id,
                    })
                    if session_id:
                        mlflow.update_current_trace(session_id=session_id)
                except Exception:
                    pass

            for i, etype in enumerate(types):
                if i > 0:
                    await asyncio.sleep(_INTER_QUERY_DELAY_SECONDS)

                pages_remaining = max(0, _MAX_PAGES_TOTAL - pages_read_total)
                query = _QUERY_TEMPLATES[etype].format(repo_id=repo_id)
                items, pages_read = await self._search_read_analyze(
                    query=query,
                    repo_id=repo_id,
                    evidence_type=etype,
                    max_results=max_results_per_type,
                    seen_urls=seen_urls,
                    should_trace=should_trace,
                    max_pages=pages_remaining,
                )
                evidence.extend(items)
                pages_read_total += pages_read

                if len(evidence) >= _MAX_TOTAL_RESULTS:
                    break

            total_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "Community search for %s: %d evidence items, %d types, %.0fms",
                repo_id, len(evidence), len(types), total_ms,
            )

            if parent_span is not None:
                try:
                    parent_span.set_outputs({
                        "evidence_count": len(evidence),
                        "total_ms": round(total_ms),
                    })
                except Exception:
                    pass

        return evidence[:_MAX_TOTAL_RESULTS]

    async def _search_read_analyze(
        self,
        query: str,
        repo_id: str,
        evidence_type: str,
        max_results: int,
        seen_urls: set[str],
        should_trace: bool,
        max_pages: int = 1,
    ) -> tuple[list[EvidenceItem], int]:
        """Single query pipeline: DuckDuckGo → Playwright → LLM.

        Returns (evidence_items, pages_read_count).
        """
        import contextlib

        # ── Step 1: DuckDuckGo Search ──
        search_span_cm: Any = contextlib.nullcontext()
        if should_trace:
            try:
                search_span_cm = mlflow.start_span(
                    name=f"ddg_search_{evidence_type}", span_type="RETRIEVER",
                )
            except Exception:
                search_span_cm = contextlib.nullcontext()

        t1 = time.monotonic()
        with search_span_cm as search_span:
            if search_span is not None:
                try:
                    search_span.set_inputs({"query": query})
                except Exception:
                    pass

            results = await self._ddg_search(query, num_results=max_results + 2)

            if search_span is not None:
                try:
                    search_span.set_outputs({
                        "result_count": len(results),
                        "urls": [r.get("url", "")[:100] for r in results[:5]],
                        "latency_ms": round((time.monotonic() - t1) * 1000),
                    })
                except Exception:
                    pass

        if not results:
            return [], 0

        # ── Step 2: Playwright Page Reading ──
        read_span_cm: Any = contextlib.nullcontext()
        if should_trace:
            try:
                read_span_cm = mlflow.start_span(
                    name=f"playwright_read_{evidence_type}", span_type="TOOL",
                )
            except Exception:
                read_span_cm = contextlib.nullcontext()

        t2 = time.monotonic()
        enriched_results: list[dict[str, Any]] = []

        with read_span_cm as read_span:
            from connectors.web_reader import read_page

            pages_read = 0
            for r in results:
                url = r.get("url", "")
                if not url or url in seen_urls:
                    continue

                domain = urlparse(url).netloc.lower()
                is_readable = any(
                    domain == d or domain.endswith(f".{d}") for d in _READABLE_DOMAINS
                )

                if pages_read < max_pages and is_readable:
                    try:
                        page_text = await read_page(
                            url, max_chars=2000,
                            on_screenshot=self._on_screenshot,
                        )
                        if page_text and not page_text.startswith("[Failed") and not page_text.startswith("[Skipped"):
                            r["page_content"] = page_text
                            pages_read += 1
                    except Exception as exc:
                        logger.debug("Page read failed for %s: %s", url, exc)

                enriched_results.append(r)
                seen_urls.add(url)

                if len(enriched_results) >= max_results:
                    break

            if read_span is not None:
                try:
                    read_span.set_outputs({
                        "pages_read": pages_read,
                        "total_results": len(enriched_results),
                        "latency_ms": round((time.monotonic() - t2) * 1000),
                    })
                except Exception:
                    pass

        # ── Step 3: LLM Content Analysis ──
        llm_span_cm: Any = contextlib.nullcontext()
        if should_trace:
            try:
                llm_span_cm = mlflow.start_span(
                    name=f"llm_analyze_{evidence_type}", span_type="LLM",
                )
            except Exception:
                llm_span_cm = contextlib.nullcontext()

        t3 = time.monotonic()
        evidence_items: list[EvidenceItem] = []
        llm_calls = 0
        total_input = 0
        total_output = 0

        with llm_span_cm as llm_span:
            for r in enriched_results:
                content = r.get("page_content") or r.get("content", "")
                if not content or len(content) < 50:
                    # Too short for LLM analysis — use raw snippet
                    item = self._to_evidence(r, repo_id, evidence_type, summary=content)
                    evidence_items.append(item)
                    continue

                summary, usage = await self._llm_analyze(
                    repo_id=repo_id,
                    evidence_type=evidence_type,
                    url=r.get("url", ""),
                    content=content,
                )
                llm_calls += 1
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)

                if summary and summary != "NOT_RELEVANT":
                    item = self._to_evidence(r, repo_id, evidence_type, summary=summary)
                    evidence_items.append(item)

            if llm_span is not None:
                try:
                    llm_span.set_outputs({
                        "llm_calls": llm_calls,
                        "evidence_produced": len(evidence_items),
                        "latency_ms": round((time.monotonic() - t3) * 1000),
                        "token_usage": {
                            "input_tokens": total_input,
                            "output_tokens": total_output,
                            "total_tokens": total_input + total_output,
                        },
                    })
                    llm_span.set_attribute("mlflow.chat.tokenUsage", {
                        "input_tokens": total_input,
                        "output_tokens": total_output,
                        "total_tokens": total_input + total_output,
                    })
                except Exception:
                    pass

        return evidence_items, pages_read

    # ── DuckDuckGo Search (library → HTML scraping fallback) ──

    async def _ddg_search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """Search via duckduckgo-search library, fallback to HTML scraping."""
        results = await self._ddg_library_search(query, num_results)
        if results:
            return results

        logger.info("DDG library returned no results, falling back to HTML scraping")
        return await self._ddg_html_search(query, num_results)

    async def _ddg_library_search(
        self, query: str, num_results: int,
    ) -> list[dict[str, Any]]:
        """Search via duckduckgo-search Python library in a thread pool."""
        try:
            from duckduckgo_search import DDGS

            def _sync_search() -> list[dict[str, Any]]:
                with DDGS(timeout=_DDG_LIB_TIMEOUT_SECONDS) as ddgs:
                    raw = list(ddgs.text(query, max_results=num_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "content": r.get("body", ""),
                    }
                    for r in raw
                ]

            results = await asyncio.to_thread(_sync_search)
            if results:
                logger.debug("DuckDuckGo library returned %d results", len(results))
            return results
        except ImportError:
            logger.debug("duckduckgo-search package not installed")
            return []
        except Exception as exc:
            logger.warning("duckduckgo-search library failed: %s", exc)
            return []

    async def _ddg_html_search(
        self, query: str, num_results: int,
    ) -> list[dict[str, Any]]:
        """Direct DuckDuckGo HTML search (last resort)."""
        import httpx

        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        try:
            async with httpx.AsyncClient(
                timeout=_DDG_TIMEOUT_SECONDS, follow_redirects=True,
            ) as client:
                resp = await client.post(url, data={"q": query}, headers=headers)
                if resp.status_code == 202:
                    logger.debug("DuckDuckGo 202 rate-limit, retrying")
                    await asyncio.sleep(1.5)
                    resp = await client.post(url, data={"q": query}, headers=headers)
                resp.raise_for_status()
                return self._parse_ddg_html(resp.text, num_results)
        except httpx.HTTPError as exc:
            logger.warning("DuckDuckGo HTML search failed: %s", exc)
            return []

    @staticmethod
    def _parse_ddg_html(html: str, max_results: int) -> list[dict[str, Any]]:
        """Extract results from DuckDuckGo HTML response."""
        results: list[dict[str, Any]] = []
        marker = 'class="result__a"'
        parts = html.split(marker)

        for part in parts[1:]:
            if len(results) >= max_results:
                break

            href = ""
            href_start = part.find('href="')
            if href_start != -1:
                href_end = part.find('"', href_start + 6)
                if href_end != -1:
                    href = part[href_start + 6 : href_end]

            if not href or href.startswith("javascript:"):
                continue

            if "uddg=" in href:
                uddg_start = href.find("uddg=") + 5
                uddg_end = href.find("&", uddg_start)
                raw = href[uddg_start:] if uddg_end == -1 else href[uddg_start:uddg_end]
                from urllib.parse import unquote
                href = unquote(raw)

            title = ""
            tag_end = part.find(">")
            close_a = part.find("</a>", tag_end) if tag_end != -1 else -1
            if tag_end != -1 and close_a != -1:
                raw_title = part[tag_end + 1 : close_a]
                title = raw_title.replace("<b>", "").replace("</b>", "").strip()

            snippet = ""
            snippet_marker = 'class="result__snippet"'
            snippet_pos = part.find(snippet_marker)
            if snippet_pos != -1:
                stag_end = part.find(">", snippet_pos)
                sclose = part.find("</", stag_end) if stag_end != -1 else -1
                if stag_end != -1 and sclose != -1:
                    raw_snippet = part[stag_end + 1 : sclose]
                    snippet = (
                        raw_snippet.replace("<b>", "").replace("</b>", "").strip()
                    )

            if href:
                results.append({"title": title or href, "url": href, "content": snippet})

        return results

    # ── LLM Content Analysis ──

    async def _llm_analyze(
        self,
        repo_id: str,
        evidence_type: str,
        url: str,
        content: str,
    ) -> tuple[str, dict[str, int]]:
        """Use LLM to analyze page content and extract a structured summary.

        Returns ``(summary, token_usage)`` where *token_usage* has keys
        ``input_tokens``, ``output_tokens``, ``total_tokens``.
        *summary* is ``"NOT_RELEVANT"`` when the content doesn't match the
        evidence type, or the raw snippet on LLM failure.
        """
        _empty_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        try:
            from backend.config import get_settings
            settings = get_settings()
        except Exception:
            return content[:500], _empty_usage

        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = settings.openai_base_url or os.environ.get("OPENAI_BASE_URL", "")
        model_name = settings.llm_model_name or os.environ.get("LLM_MODEL_NAME", "")

        if not api_key or not base_url or not model_name:
            logger.debug("LLM not configured, skipping content analysis")
            return content[:500], _empty_usage

        try:
            import httpx
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            kwargs: dict[str, Any] = {
                "model": model_name,
                "base_url": base_url,
                "api_key": api_key,
                "temperature": 0.1,
                "max_tokens": 300,
                "timeout": 30,
                "streaming": True,
            }
            if not settings.verify_ssl:
                kwargs["http_async_client"] = httpx.AsyncClient(verify=False)

            llm = ChatOpenAI(**kwargs)
            user_msg = _LLM_ANALYSIS_USER.format(
                repo_id=repo_id,
                evidence_type=evidence_type,
                url=url,
                content=content[:3000],
            )
            response = await llm.ainvoke([
                SystemMessage(content=_LLM_ANALYSIS_SYSTEM),
                HumanMessage(content=user_msg),
            ])

            usage = getattr(response, "usage_metadata", None)
            token_usage = {
                "input_tokens": usage.get("input_tokens", 0) if usage else 0,
                "output_tokens": usage.get("output_tokens", 0) if usage else 0,
                "total_tokens": usage.get("total_tokens", 0) if usage else 0,
            }
            return response.content.strip(), token_usage
        except Exception as exc:
            logger.warning("LLM analysis failed for %s: %s", url, exc)
            return content[:500], _empty_usage

    # ── Evidence Construction ──

    def _to_evidence(
        self,
        result: dict[str, Any],
        repo_id: str,
        evidence_type: str,
        summary: str,
    ) -> EvidenceItem:
        """Convert a search result + summary into an EvidenceItem."""
        url = result.get("url", "")
        title = result.get("title", "")
        domain = urlparse(url).netloc if url else None

        raw_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest() if summary else None
        claim_type = self._determine_claim_type(evidence_type, summary)
        source_tier = self._determine_source_tier(url)

        if not summary:
            summary = f"Community result for {repo_id} ({evidence_type})"

        return EvidenceItem(
            evidence_id=uuid4(),
            category="community",
            claim_type=claim_type,
            title=title or f"{repo_id} — {evidence_type}",
            summary=summary[:500],
            source_url=url,
            source_domain=domain,
            retrieved_at=datetime.now(UTC),
            raw_excerpt_hash=raw_hash,
            source_tier=source_tier,
            verification_level="reported",
        )

    def _determine_claim_type(self, evidence_type: str, content: str) -> str:
        """Map evidence_type + content heuristics to a claim_type literal."""
        base = _EVIDENCE_TYPE_TO_CLAIM.get(evidence_type, "compatibility")
        if not content:
            return base  # type: ignore[return-value]

        lower = content.lower()
        if any(kw in lower for kw in ("limitation", "not support", "unsupported", "issue", "bug")):
            return "limitation"
        if any(kw in lower for kw in ("benchmark", "throughput", "latency", "tokens/s", "tok/s")):
            return "serving_performance"
        if any(kw in lower for kw in ("accuracy", "mmlu", "eval", "score", "perplexity")):
            return "accuracy"
        return base  # type: ignore[return-value]

    def _determine_source_tier(self, url: str) -> str:
        """Determine source tier from URL domain."""
        if not url:
            return "community"
        domain = urlparse(url).netloc.lower()

        for official in _OFFICIAL_DOMAINS:
            if domain == official or domain.endswith(f".{official}"):
                return "primary"

        for secondary in _OFFICIAL_SECONDARY_DOMAINS:
            if domain == secondary or domain.endswith(f".{secondary}"):
                return "official_secondary"

        return "community"
