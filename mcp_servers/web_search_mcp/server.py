"""Web Search MCP Server — web search tool with SearXNG + DuckDuckGo fallback.

Reused from the baseline research lab architecture.
Provides a streamable-HTTP MCP endpoint for the community_search connector.
"""

import logging
import os

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(override=True)

logger = logging.getLogger(__name__)

SEARXNG_URL = os.getenv("SEARXNG_URL", "")
_VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() not in ("false", "0", "no")
_DEFAULT_ENGINES = os.getenv("SEARXNG_ENGINES", "google,duckduckgo,brave")

mcp = FastMCP("web-search-mcp", host="0.0.0.0", port=9003, stateless_http=True)


def _search_searxng(query: str, num_results: int) -> list[dict]:
    """Search via SearXNG instance."""
    client = httpx.Client(verify=_VERIFY_SSL, timeout=httpx.Timeout(15.0))
    resp = client.get(
        f"{SEARXNG_URL}/search",
        params={"q": query, "format": "json", "engines": _DEFAULT_ENGINES},
    )
    client.close()
    resp.raise_for_status()
    data = resp.json()
    results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in data.get("results", [])[:num_results]
    ]
    if not results:
        unresponsive = data.get("unresponsive_engines", [])
        if unresponsive:
            logger.warning("SearXNG returned 0 results — unresponsive engines: %s", unresponsive)
            raise RuntimeError(f"SearXNG engines unavailable: {unresponsive}")
    return results


_DDG_TIMEOUT = int(os.getenv("DDG_TIMEOUT", "20"))


def _search_duckduckgo(query: str, num_results: int) -> list[dict]:
    """Search via DuckDuckGo (local, no server required)."""
    from duckduckgo_search import DDGS

    with DDGS(timeout=_DDG_TIMEOUT) as ddgs:
        raw = list(ddgs.text(query, max_results=num_results))
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
        for r in raw
    ]


@mcp.tool()
def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web for model evidence. Uses SearXNG if configured, falls back to DuckDuckGo."""
    if SEARXNG_URL:
        try:
            results = _search_searxng(query, num_results)
            if results:
                return results
        except Exception as e:
            logger.warning("SearXNG search failed, falling back to DuckDuckGo: %s", e)

    try:
        return _search_duckduckgo(query, num_results)
    except Exception as e:
        logger.error("Web search failed (all backends): %s", e)
        return []


@mcp.tool()
def search_model_evidence(
    repo_id: str,
    evidence_types: list[str] | None = None,
    preferred_domains: list[str] | None = None,
    published_after: str | None = None,
    max_results: int = 5,
) -> list[dict]:
    """Search for model-specific evidence with domain-aware queries.

    Args:
        repo_id: HuggingFace model repo ID (e.g. "meta-llama/Llama-3.1-70B-Instruct")
        evidence_types: Types to search (compatibility, performance, strengths, deployment, tool_calling)
        preferred_domains: Preferred result domains
        published_after: ISO date filter (best-effort via query)
        max_results: Maximum results to return

    Returns:
        List of search results with title, url, content, and metadata.
    """
    types = evidence_types or ["compatibility", "strengths"]
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    query_templates = {
        "compatibility": f"{repo_id} vLLM compatibility issues deployment",
        "performance": f"{repo_id} vLLM benchmark throughput latency GPU",
        "strengths": f"{repo_id} model strengths use cases evaluation",
        "deployment": f"{repo_id} vLLM deployment guide tensor parallel",
        "tool_calling": f"{repo_id} tool calling function calling vLLM",
    }

    for etype in types[:3]:
        query = query_templates.get(etype, f"{repo_id} {etype}")
        if published_after:
            query += f" after:{published_after[:10]}"

        results = web_search(query, num_results=max_results)
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                r["evidence_type"] = etype
                all_results.append(r)

    return all_results[:max_results]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
