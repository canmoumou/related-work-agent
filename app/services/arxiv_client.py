"""arXiv API 客户端，负责检索、限速与结果解析。"""

from __future__ import annotations

import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Iterable

import httpx

from app.core.config import Settings, get_settings
from app.models.schemas import CandidatePaper
from app.utils.text_utils import normalize_whitespace


LOGGER = logging.getLogger(__name__)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivClient:
    """封装 arXiv 查询构造、请求控制和 Atom 结果解析。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._request_lock = asyncio.Lock()
        self._last_request_started_at = 0.0

    @staticmethod
    def build_query(terms: Iterable[str], field: str = "all") -> str:
        """构造 arXiv 查询串，并在 term 层面先做去重。"""
        clean_terms = list(dict.fromkeys(term.strip() for term in terms if term.strip()))
        if not clean_terms:
            return "all:machine learning"
        return " OR ".join(f'{field}:"{term}"' for term in clean_terms)

    async def search(self, query: str, start: int = 0, max_results: int = 10) -> list[CandidatePaper]:
        """发起带限速和重试的 arXiv 检索请求。"""
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        headers = {
            "User-Agent": self.settings.arxiv_user_agent,
        }

        last_error: Exception | None = None
        for attempt in range(self.settings.arxiv_retry_count + 1):
            try:
                response = await self._throttled_get(params=params, headers=headers)
                # arXiv 限流时显式转成 HTTPStatusError，便于统一走退避重试逻辑。
                if response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        message=f"Client error '429 Too Many Requests' for url '{response.request.url}'",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return self.parse_feed(response.text)
            except (httpx.HTTPStatusError, httpx.ReadTimeout) as exc:
                last_error = exc
                should_retry = self._should_retry(exc) and attempt < self.settings.arxiv_retry_count
                if not should_retry:
                    raise
                wait_seconds = self.settings.arxiv_backoff_seconds * (2**attempt)
                LOGGER.warning(
                    "arXiv request failed on attempt %s/%s, retrying in %.1fs: %s",
                    attempt + 1,
                    self.settings.arxiv_retry_count + 1,
                    wait_seconds,
                    exc,
                )
                await asyncio.sleep(wait_seconds)

        if last_error is not None:
            raise last_error
        return []

    async def _throttled_get(self, params: dict[str, object], headers: dict[str, str]) -> httpx.Response:
        """通过锁和最小间隔控制请求频率，降低触发限流的概率。"""
        async with self._request_lock:
            elapsed = time.monotonic() - self._last_request_started_at
            wait_seconds = max(0.0, self.settings.arxiv_min_interval_seconds - elapsed)
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._last_request_started_at = time.monotonic()

            async with httpx.AsyncClient(
                timeout=self.settings.arxiv_timeout_seconds,
                follow_redirects=True,
                headers=headers,
            ) as client:
                return await client.get(self.settings.arxiv_base_url, params=params)

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        """只有短暂性错误才值得重试。"""
        if isinstance(exc, httpx.ReadTimeout):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {429, 500, 502, 503, 504}
        return False

    def parse_feed(self, feed_text: str) -> list[CandidatePaper]:
        """把 arXiv Atom Feed 解析成统一的 CandidatePaper 结构。"""
        root = ET.fromstring(feed_text)
        papers: list[CandidatePaper] = []

        for entry in root.findall("atom:entry", ATOM_NS):
            entry_id = normalize_whitespace(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
            arxiv_id = entry_id.rstrip("/").split("/")[-1]
            title = normalize_whitespace(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
            abstract = normalize_whitespace(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
            authors = [
                normalize_whitespace(author.findtext("atom:name", default="", namespaces=ATOM_NS))
                for author in entry.findall("atom:author", ATOM_NS)
            ]
            categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", ATOM_NS)]
            published = self._parse_datetime(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
            updated = self._parse_datetime(entry.findtext("atom:updated", default="", namespaces=ATOM_NS))

            pdf_url = None
            abs_url = entry_id or None
            for link in entry.findall("atom:link", ATOM_NS):
                href = link.attrib.get("href")
                title_attr = link.attrib.get("title", "")
                if title_attr == "pdf" and href:
                    pdf_url = href

            papers.append(
                CandidatePaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=[author for author in authors if author],
                    abstract=abstract,
                    categories=[category for category in categories if category],
                    published=published,
                    updated=updated,
                    pdf_url=pdf_url,
                    abs_url=abs_url,
                )
            )
        return papers

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        """解析 arXiv 返回的时间字段。"""
        if not value:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
