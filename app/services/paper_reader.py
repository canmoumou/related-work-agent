"""论文内容读取与启发式 section 抽取。"""

from __future__ import annotations

import logging
import re

from app.models.schemas import CandidatePaper, PaperSection, PaperSections
from app.utils.text_utils import normalize_whitespace


LOGGER = logging.getLogger(__name__)


class PaperReader:
    """当前版本主要基于 arXiv 摘要生成 section 级候选内容。"""

    async def read(self, papers: list[CandidatePaper]) -> list[PaperSections]:
        """批量读取论文，并为后续 method extraction 准备结构化片段。"""
        return [self._extract_sections(paper) for paper in papers]

    def _extract_sections(self, paper: CandidatePaper) -> PaperSections:
        # 当前优先走摘要级启发式抽取，避免把不稳定的全文解析直接塞进主流程。
        notes = [
            "Current version prefers metadata-based extraction from arXiv abstract.",
            "PDF full-text extraction is comparatively hard to make stable across papers; using heuristic fallback.",
        ]
        abstract = normalize_whitespace(paper.abstract)

        sections = [
            PaperSection(label="abstract", content=abstract, source="arxiv_abstract"),
            PaperSection(
                label="introduction_like",
                content=self._build_introduction_like(abstract),
                source="heuristic_from_abstract",
            ),
            PaperSection(
                label="method_like",
                content=self._build_method_like(abstract),
                source="heuristic_from_abstract",
            ),
            PaperSection(
                label="contribution_like",
                content=self._build_contribution_like(abstract),
                source="heuristic_from_abstract",
            ),
        ]

        if not sections[2].content:
            sections.append(
                PaperSection(
                    label="fallback_summary",
                    content=abstract,
                    source="fallback_to_abstract",
                )
            )
            notes.append("Method-like extraction was weak, so the reader fell back to abstract-only content.")

        return PaperSections(
            paper_id=paper.arxiv_id,
            title=paper.title,
            sections=[section for section in sections if section.content],
            extraction_notes=notes,
        )

    @staticmethod
    def _sentences(text: str) -> list[str]:
        """按句子粗切分文本，服务于后续启发式抽取。"""
        candidates = re.split(r"(?<=[.!?])\s+", text)
        return [normalize_whitespace(item) for item in candidates if normalize_whitespace(item)]

    def _build_introduction_like(self, abstract: str) -> str:
        """使用摘要开头句子近似引言段。"""
        sentences = self._sentences(abstract)
        return " ".join(sentences[:2]) if sentences else abstract

    def _build_method_like(self, abstract: str) -> str:
        """优先抓取带方法关键词的句子，否则退回到中间句。"""
        keywords = ("we propose", "we present", "our method", "framework", "model", "approach", "architecture")
        for sentence in self._sentences(abstract):
            if any(keyword in sentence.lower() for keyword in keywords):
                return sentence
        return " ".join(self._sentences(abstract)[1:3])

    def _build_contribution_like(self, abstract: str) -> str:
        """抽取更像结果或贡献描述的句子。"""
        keywords = ("results", "achieve", "improve", "outperform", "contribution", "demonstrate")
        matched = [sentence for sentence in self._sentences(abstract) if any(keyword in sentence.lower() for keyword in keywords)]
        return " ".join(matched[:2])
