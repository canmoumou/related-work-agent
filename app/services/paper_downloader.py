"""下载论文 PDF，并转换为本地 Markdown 缓存。"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
import pymupdf4llm

from app.core.config import Settings, get_settings
from app.models.schemas import CandidatePaper


LOGGER = logging.getLogger(__name__)


class PaperDownloader:
    """负责下载候选论文 PDF，并复用 Markdown 缓存。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def download_many(self, papers: list[CandidatePaper]) -> dict[str, str]:
        """批量下载并转换论文，返回 arXiv ID 到 Markdown 路径的映射。"""
        downloads: dict[str, str] = {}
        for paper in papers:
            path = await self.download_one(paper)
            if path is not None:
                downloads[paper.arxiv_id] = str(path)
        return downloads

    async def download_one(self, paper: CandidatePaper) -> Path | None:
        """下载单篇论文并转换为 Markdown；若无 PDF 链接则返回 None。"""
        if not paper.pdf_url:
            LOGGER.info("Skip downloading %s because pdf_url is missing", paper.arxiv_id)
            return None

        pdf_path = self._get_paper_path(paper.arxiv_id, ".pdf")
        markdown_path = self._get_paper_path(paper.arxiv_id, ".md")
        if markdown_path.exists() and markdown_path.stat().st_size > 0:
            return markdown_path

        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            success = await self._download_pdf(paper, pdf_path)
            if not success:
                return None

        try:
            await self._convert_pdf_to_markdown(pdf_path, markdown_path)
            return markdown_path
        except Exception as exc:
            LOGGER.warning("Failed to convert PDF for %s: %s", paper.arxiv_id, exc)
            return None

    def _get_paper_path(self, paper_id: str, suffix: str) -> Path:
        """返回统一的本地缓存路径。"""
        self.settings.download_dir.mkdir(parents=True, exist_ok=True)
        return self.settings.download_dir / f"{paper_id}{suffix}"

    async def _download_pdf(self, paper: CandidatePaper, target_path: Path) -> bool:
        """下载 PDF 到本地缓存。"""
        last_error: Exception | None = None
        for attempt in range(self.settings.pdf_download_retry_count + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self.settings.pdf_download_timeout_seconds,
                    follow_redirects=True,
                    headers={"User-Agent": self.settings.arxiv_user_agent},
                ) as client:
                    response = await client.get(str(paper.pdf_url))
                    response.raise_for_status()
                    if "pdf" not in response.headers.get("content-type", "").lower():
                        LOGGER.warning(
                            "Unexpected content type for %s: %s",
                            paper.arxiv_id,
                            response.headers.get("content-type", ""),
                        )
                    target_path.write_bytes(response.content)
                    return True
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                if attempt >= self.settings.pdf_download_retry_count:
                    break
                wait_seconds = 2**attempt
                LOGGER.warning(
                    "PDF download failed for %s on attempt %s/%s, retrying in %ss: %s",
                    paper.arxiv_id,
                    attempt + 1,
                    self.settings.pdf_download_retry_count + 1,
                    wait_seconds,
                    exc,
                )
                await asyncio.sleep(wait_seconds)

        LOGGER.warning("Failed to download PDF for %s: %s", paper.arxiv_id, last_error)
        return False

    async def _convert_pdf_to_markdown(self, pdf_path: Path, markdown_path: Path) -> None:
        """把 PDF 转成 Markdown，复用 arxiv-mcp-server 风格的本地阅读格式。"""
        markdown = await asyncio.to_thread(self._render_markdown, pdf_path)
        if not markdown.strip():
            raise ValueError(f"Markdown conversion returned empty content for {pdf_path.name}")
        markdown_path.write_text(markdown, encoding="utf-8")

    @staticmethod
    def _render_markdown(pdf_path: Path) -> str:
        """同步执行 PDF -> Markdown 转换。"""
        return pymupdf4llm.to_markdown(pdf_path, show_progress=False)
