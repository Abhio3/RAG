"""Local web search (SearXNG) + fetch→markdown (trafilatura). No cloud, no browser.

SearXNG runs locally (docker-compose) and exposes a JSON search API. trafilatura
downloads a page and extracts the main content as markdown — no Playwright/Chromium,
which is why we skip Crawl4AI here (see docs/DATA_MODEL.md §1; fetcher='http').
"""
import os

import httpx
import trafilatura

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")


def search(query: str, k: int = 5) -> list[dict]:
    """Top-k results from the local SearXNG JSON API → [{title, url}]."""
    r = httpx.get(f"{SEARXNG_URL}/search", params={"q": query, "format": "json"}, timeout=20)
    r.raise_for_status()
    out = []
    for x in r.json().get("results", []):
        url = x.get("url")
        if url:
            out.append({"title": x.get("title") or url, "url": url})
        if len(out) >= k:
            break
    return out


def fetch_markdown(url: str) -> str:
    """Fetch a page and extract its main content as markdown ('' on failure)."""
    html = trafilatura.fetch_url(url)
    if not html:
        return ""
    return trafilatura.extract(html, output_format="markdown", include_links=False) or ""
