"""Community evidence search connector.

Uses web search (SearXNG + DuckDuckGo fallback) to discover community evidence
about model deployments, compatibility issues, and operational characteristics.

Can call the web-search MCP server via HTTP, or fall back to direct DuckDuckGo.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from domain.evidence import EvidenceItem

logger = logging.getLogger(__name__)

MCP_WEB_SEARCH_URL = os.getenv("MCP_WEB_SEARCH_URL", "http://127.0.0.1:9003")

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
    "strengths": "{repo_id} model strengths use cases evaluation",
    "deployment": "{repo_id} vLLM deployment guide tensor parallel configuration",
    "tool_calling": "{repo_id} tool calling function calling vLLM",
}

_EVIDENCE_TYPE_TO_CLAIM: dict[str, str] = {
    "compatibility": "compatibility",
    "performance": "serving_performance",
    "strengths": "model_strength",
    "deployment": "tested_hardware",
    "tool_calling": "compatibility",
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

_MAX_TOTAL_RESULTS = 25
_MCP_TIMEOUT_SECONDS = 10
_DDG_TIMEOUT_SECONDS = 10


class CommunitySearchConnector:
    """Discovers community evidence about models via web search."""

    def __init__(self, mcp_url: str | None = None):
        self._mcp_url = mcp_url or MCP_WEB_SEARCH_URL

    async def search_model_evidence(
        self,
        repo_id: str,
        evidence_types: list[str] | None = None,
        max_results_per_type: int = 3,
        published_after: str | None = None,
    ) -> list[EvidenceItem]:
        """Search for community evidence about a model.

        Args:
            repo_id: HuggingFace model repo ID (e.g. "meta-llama/Llama-3-8B").
            evidence_types: Types to search for (default: all template keys).
            max_results_per_type: Max results kept per query type.
            published_after: ISO date string filter (best-effort, appended to query).

        Returns:
            Deduplicated list of EvidenceItem objects with category="community".
        """
        types = evidence_types or list(_QUERY_TEMPLATES.keys())
        types = [t for t in types if t in _QUERY_TEMPLATES]
        if not types:
            logger.warning("No valid evidence types requested for %s", repo_id)
            return []

        seen_urls: set[str] = set()
        evidence: list[EvidenceItem] = []

        for etype in types:
            query = _QUERY_TEMPLATES[etype].format(repo_id=repo_id)
            if published_after:
                query += f" after:{published_after}"

            results = await self._web_search(query, num_results=max_results_per_type + 2)
            kept = 0
            for result in results:
                if kept >= max_results_per_type:
                    break
                url = result.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                item = self._search_result_to_evidence(result, repo_id, etype)
                evidence.append(item)
                kept += 1

            if len(evidence) >= _MAX_TOTAL_RESULTS:
                break

        logger.info(
            "Community search for %s returned %d evidence items across %d types",
            repo_id,
            len(evidence),
            len(types),
        )
        return evidence[:_MAX_TOTAL_RESULTS]

    async def _web_search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """Execute web search via MCP server or direct DuckDuckGo fallback."""
        results = await self._call_mcp_web_search(query, num_results)
        if results:
            return results

        logger.info("MCP web-search unavailable, falling back to direct DuckDuckGo")
        return await self._direct_duckduckgo(query, num_results)

    async def _call_mcp_web_search(
        self, query: str, num_results: int
    ) -> list[dict[str, Any]]:
        """Call the web-search MCP server via streamable HTTP.

        Sends a JSON-RPC 2.0 ``tools/call`` request to the MCP endpoint.
        Returns an empty list on any transport or protocol error so the
        caller can fall back gracefully.
        """
        endpoint = f"{self._mcp_url.rstrip('/')}/mcp"
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "tools/call",
            "params": {
                "name": "web_search",
                "arguments": {"query": query, "num_results": num_results},
            },
        }

        try:
            async with httpx.AsyncClient(timeout=_MCP_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                )
                resp.raise_for_status()

                data = resp.json()
                result = data.get("result", {})

                content_list = result.get("content", [])
                if not content_list:
                    return []

                import json

                for content_item in content_list:
                    if content_item.get("type") == "text":
                        parsed = json.loads(content_item["text"])
                        if isinstance(parsed, list):
                            return parsed
                        return []

                return []
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.debug("MCP web-search call failed: %s", exc)
            return []

    async def _direct_duckduckgo(
        self, query: str, num_results: int
    ) -> list[dict[str, Any]]:
        """Direct DuckDuckGo HTML search fallback (no MCP needed).

        Uses the DuckDuckGo HTML-only endpoint and parses result links.
        Returns results in the same ``[{"title", "url", "content"}]`` shape
        as the MCP server.
        """
        url = "https://html.duckduckgo.com/html/"
        try:
            async with httpx.AsyncClient(
                timeout=_DDG_TIMEOUT_SECONDS,
                follow_redirects=True,
            ) as client:
                resp = await client.post(
                    url,
                    data={"q": query},
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                )
                resp.raise_for_status()
                return self._parse_ddg_html(resp.text, num_results)
        except httpx.HTTPError as exc:
            logger.warning("DuckDuckGo fallback failed: %s", exc)
            return []

    @staticmethod
    def _parse_ddg_html(html: str, max_results: int) -> list[dict[str, Any]]:
        """Extract search results from DuckDuckGo HTML response.

        The HTML-only endpoint returns ``<a class="result__a">`` links with
        ``<a class="result__snippet">`` snippets.  We use simple string
        splitting to avoid requiring an HTML parser dependency.
        """
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

            # DuckDuckGo wraps URLs through a redirect; extract the actual URL.
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
                        raw_snippet.replace("<b>", "")
                        .replace("</b>", "")
                        .strip()
                    )

            if href:
                results.append({"title": title or href, "url": href, "content": snippet})

        return results

    def _search_result_to_evidence(
        self, result: dict[str, Any], repo_id: str, evidence_type: str
    ) -> EvidenceItem:
        """Convert a search result dict to an EvidenceItem."""
        url = result.get("url", "")
        title = result.get("title", "")
        content = result.get("content", "")
        domain = urlparse(url).netloc if url else None

        raw_hash = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None

        claim_type = self._determine_claim_type(evidence_type, content)
        source_tier = self._determine_source_tier(url)

        summary = content[:500] if content else title
        if not summary:
            summary = f"Community result for {repo_id} ({evidence_type})"

        return EvidenceItem(
            evidence_id=uuid4(),
            category="community",
            claim_type=claim_type,
            title=title or f"{repo_id} — {evidence_type}",
            summary=summary,
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
        """Determine source tier from URL domain.

        Returns ``"primary"`` for official Red Hat / vLLM docs,
        ``"official_secondary"`` for HuggingFace / arXiv / GitHub,
        and ``"community"`` for everything else.
        """
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
