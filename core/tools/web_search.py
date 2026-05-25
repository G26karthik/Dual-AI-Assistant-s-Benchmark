from __future__ import annotations

import os

import httpx


async def web_search(query: str, max_results: int = 3) -> str:
    """Return top web results from Tavily as formatted text."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search unavailable: missing TAVILY_API_KEY."

    payload = {"api_key": api_key, "query": query, "max_results": max_results}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        return f"Web search error: {exc}"

    results = data.get("results", [])
    if not results:
        return "No search results found."

    lines: list[str] = []
    for idx, item in enumerate(results[:max_results], start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        snippet = item.get("content", "")
        lines.append(f"{idx}. {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n".join(lines)
