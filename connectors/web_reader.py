"""Web page content reader.

Fetches a URL and extracts the main text content.

Priority:
  1. Playwright (headless Chromium — handles JS-rendered pages)
  2. httpx + simple HTML tag stripping (lightweight fallback)

The extracted text is truncated to a configurable max length to keep
LLM context windows manageable.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() not in ("false", "0", "no")
_PAGE_TIMEOUT_MS = 15_000
_MAX_CONTENT_CHARS = 4000
_HTTPX_TIMEOUT_SECONDS = 12
_THUMBNAIL_WIDTH = 480
_THUMBNAIL_HEIGHT = 300

_BLOCKED_EXTENSIONS = {".pdf", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mp3"}

# Callback type for streaming page preview screenshots
OnScreenshotCallback = Callable[[str, str, str], Any]  # (url, title, base64_png) -> None

_NOISE_TAGS = re.compile(
    r"<\s*(script|style|nav|footer|header|aside|iframe|noscript|svg)"
    r"[^>]*>[\s\S]*?</\s*\1\s*>",
    re.IGNORECASE,
)
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace to extract readable text."""
    text = _NOISE_TAGS.sub(" ", html)
    text = _HTML_TAG.sub(" ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = _WHITESPACE.sub("\n\n", text)
    return text.strip()


def _should_skip(url: str) -> bool:
    """Check if the URL points to a binary/non-text resource."""
    from urllib.parse import urlparse

    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _BLOCKED_EXTENSIONS)


async def read_page(
    url: str,
    max_chars: int = _MAX_CONTENT_CHARS,
    on_screenshot: OnScreenshotCallback | None = None,
) -> str:
    """Fetch a URL and return extracted text content.

    Tries Playwright first (handles JS-rendered pages like HuggingFace model
    cards). Falls back to httpx + HTML stripping if Playwright is unavailable.

    Args:
        url: The page URL to read.
        max_chars: Maximum characters to return (default 4000).
        on_screenshot: Optional callback invoked with (url, title, base64_png)
            when a page screenshot is captured.

    Returns:
        Extracted text content, truncated to max_chars.
    """
    if _should_skip(url):
        return f"[Skipped: binary/non-text resource at {url}]"

    text = await _read_with_playwright(url, on_screenshot=on_screenshot)
    if not text:
        text = await _read_with_httpx(url)

    if not text:
        return f"[Failed to read content from {url}]"

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... content truncated]"
    return text


async def _read_with_playwright(
    url: str,
    on_screenshot: OnScreenshotCallback | None = None,
) -> str:
    """Render page with headless Chromium, capture screenshot, and extract text."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.debug("Playwright not installed, skipping browser-based reading")
        return ""

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                    ),
                    java_script_enabled=True,
                    viewport={"width": _THUMBNAIL_WIDTH, "height": _THUMBNAIL_HEIGHT},
                    device_scale_factor=1,
                )
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=_PAGE_TIMEOUT_MS)
                await page.wait_for_timeout(1500)

                # Capture lightweight thumbnail (~5-15KB JPEG)
                if on_screenshot:
                    try:
                        title = await page.title() or url
                        screenshot_bytes = await page.screenshot(
                            type="jpeg", quality=30,
                            clip={"x": 0, "y": 0,
                                  "width": _THUMBNAIL_WIDTH,
                                  "height": _THUMBNAIL_HEIGHT},
                        )
                        b64 = base64.b64encode(screenshot_bytes).decode("ascii")
                        logger.debug(
                            "Thumbnail captured for %s: %d bytes",
                            url, len(screenshot_bytes),
                        )
                        on_screenshot(url, title, b64)
                    except Exception as exc:
                        logger.debug("Screenshot capture failed for %s: %s", url, exc)

                text = await page.evaluate("""() => {
                    for (const sel of ['nav', 'footer', 'header', '.sidebar',
                                       '#sidebar', 'script', 'style', 'noscript',
                                       '[role="navigation"]', '[role="banner"]',
                                       '.cookie-banner', '.ad', '.advertisement']) {
                        document.querySelectorAll(sel).forEach(el => el.remove());
                    }

                    const main = document.querySelector('main')
                        || document.querySelector('article')
                        || document.querySelector('[role="main"]')
                        || document.querySelector('.markdown-body')
                        || document.querySelector('.prose')
                        || document.querySelector('#content')
                        || document.body;

                    return main ? main.innerText : document.body.innerText;
                }""")

                await context.close()
                return text.strip() if text else ""
            finally:
                await browser.close()
    except Exception as exc:
        logger.warning("Playwright page read failed for %s: %s", url, exc)
        return ""


async def _read_with_httpx(url: str) -> str:
    """Fetch page with httpx and strip HTML tags (fallback)."""
    try:
        async with httpx.AsyncClient(
            timeout=_HTTPX_TIMEOUT_SECONDS,
            follow_redirects=True,
            verify=VERIFY_SSL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return ""

            return _strip_html(resp.text)
    except httpx.HTTPError as exc:
        logger.warning("httpx page read failed for %s: %s", url, exc)
        return ""


async def search_and_read(
    query: str,
    num_results: int = 3,
    max_chars_per_page: int = 3000,
) -> str:
    """DuckDuckGo search → read top result pages → return combined content.

    Convenience function that combines search + page reading.
    Returns formatted text suitable for LLM consumption.
    """
    from connectors.community_search import CommunitySearchConnector

    connector = CommunitySearchConnector()
    results = await connector._ddg_search(query, num_results=num_results + 2)

    if not results:
        return "No search results found."

    sections: list[str] = []
    read_count = 0

    for r in results:
        if read_count >= num_results:
            break

        url = r.get("url", "")
        title = r.get("title", "")

        if not url or _should_skip(url):
            continue

        content = await read_page(url, max_chars=max_chars_per_page)
        if content.startswith("[Failed") or content.startswith("[Skipped"):
            snippet = r.get("content", "")
            if snippet:
                sections.append(f"### {title}\nSource: {url}\n\n{snippet}")
            continue

        sections.append(f"### {title}\nSource: {url}\n\n{content}")
        read_count += 1

    if not sections:
        snippet_lines = []
        for r in results[:num_results]:
            snippet_lines.append(
                f"- [{r.get('title', '')}]({r.get('url', '')}): {r.get('content', '')[:200]}"
            )
        return "Could not read page contents. Search snippets:\n" + "\n".join(snippet_lines)

    return "\n\n---\n\n".join(sections)
