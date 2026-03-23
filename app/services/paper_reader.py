"""论文内容读取与方法论 section 抽取。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.models.schemas import CandidatePaper, PaperSection, PaperSections
from app.utils.text_utils import normalize_whitespace


LOGGER = logging.getLogger(__name__)
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
METHOD_HEADING_HINTS = {
    "method",
    "methods",
    "approach",
    "methodology",
    "algorithm",
    "framework",
    "implementation",
    "model",
    "models",
    "system",
}


@dataclass
class MarkdownSection:
    """Markdown 标题块。"""

    raw_heading: str
    normalized_heading: str
    structural_depth: int
    content: str


class PaperReader:
    """优先基于 Markdown 全文抽取方法论 section，失败时回退到摘要。"""

    def __init__(self) -> None:
        self.max_section_chars = 9000

    async def read(
        self,
        papers: list[CandidatePaper],
        markdown_paths: Optional[dict[str, str]] = None,
    ) -> list[PaperSections]:
        """批量读取论文，并为后续 method extraction 准备结构化片段。"""
        markdown_paths = markdown_paths or {}
        return [self._extract_sections(paper, markdown_paths.get(paper.arxiv_id)) for paper in papers]

    def _extract_sections(self, paper: CandidatePaper, markdown_path: Optional[str] = None) -> PaperSections:
        """从 Markdown 或摘要中提取结构化片段。"""
        notes: list[str] = []
        abstract = normalize_whitespace(paper.abstract)

        if markdown_path:
            try:
                sections, markdown_notes = self._extract_from_markdown(markdown_path)
                notes.extend(markdown_notes)
                if sections:
                    return PaperSections(
                        paper_id=paper.arxiv_id,
                        title=paper.title,
                        sections=sections,
                        extraction_notes=notes,
                    )
                notes.append("Markdown parsing produced no usable method-related sections, so the workflow fell back to abstract evidence.")
            except Exception as exc:
                LOGGER.warning("Failed to extract Markdown sections from %s: %s", markdown_path, exc)
                notes.append(f"Markdown extraction failed: {str(exc)[:120]}")
        else:
            notes.append("No local Markdown cache was available, so the workflow used abstract-derived evidence.")

        return self._build_abstract_fallback(paper, abstract, notes)

    def _build_abstract_fallback(self, paper: CandidatePaper, abstract: str, notes: list[str]) -> PaperSections:
        """回退到基于摘要的 method_like 构造。"""
        method_like = self._build_method_like(abstract) or abstract
        sections = [PaperSection(label="method_like", content=method_like, source="heuristic_from_abstract")]
        if not self._build_method_like(abstract):
            notes.append("Method-like extraction from abstract was weak, so the workflow used the abstract itself as method_like evidence.")

        return PaperSections(
            paper_id=paper.arxiv_id,
            title=paper.title,
            sections=sections,
            extraction_notes=notes,
        )

    def _extract_from_markdown(self, markdown_path: str) -> tuple[list[PaperSection], list[str]]:
        """从 Markdown 中提取结构化方法论 section。"""
        markdown_text = Path(markdown_path).read_text(encoding="utf-8")
        normalized_markdown = self._normalize_markdown(markdown_text)
        if not normalized_markdown:
            return [], [f"Markdown cache was empty for {markdown_path}."]

        sections = self._collect_markdown_sections(normalized_markdown)
        notes = [
            "Paper content was read from local Markdown cache.",
            f"Detected {len(sections)} Markdown sections from converted paper content.",
        ]
        method_content = self._extract_method_section_hierarchy(sections)
        if method_content:
            notes.append("Method-like content was extracted by aggregating the Markdown method section hierarchy.")
            return [
                PaperSection(
                    label="method_like",
                    content=self._truncate_text(method_content),
                    source="markdown_section_hierarchy",
                )
            ], notes

        heuristic_method = self._extract_method_content(normalized_markdown)
        if heuristic_method:
            notes.append("No explicit method heading was found in Markdown; method-like content was assembled heuristically from body text.")
            return [
                PaperSection(
                    label="method_like",
                    content=self._truncate_text(heuristic_method),
                    source="markdown_extraction_heuristic",
                )
            ], notes

        notes.append("Markdown was readable but no reliable method-oriented content was found.")
        return [], notes

    def _normalize_markdown(self, text: str) -> str:
        """对 Markdown 做轻量清洗，保留标题边界。"""
        lines = [line.rstrip() for line in text.splitlines()]
        cleaned_lines: list[str] = []
        blank_run = 0
        for line in lines:
            normalized = re.sub(r"[ \t]+", " ", line).strip()
            if not normalized:
                blank_run += 1
                if blank_run <= 1:
                    cleaned_lines.append("")
                continue
            blank_run = 0
            cleaned_lines.append(normalized)
        return "\n".join(cleaned_lines).strip()

    def _collect_markdown_sections(self, text: str) -> list[MarkdownSection]:
        """按 Markdown 标题切分结构化 section。"""
        sections: list[MarkdownSection] = []
        current_heading = "document_start"
        current_depth = 0
        current_lines: list[str] = []

        for line in text.splitlines():
            match = MARKDOWN_HEADING_RE.match(line)
            if match:
                if current_lines:
                    sections.append(
                        MarkdownSection(
                            raw_heading=current_heading,
                            normalized_heading=self._normalize_heading_label(current_heading),
                            structural_depth=current_depth,
                            content=self._strip_heading_markers("\n".join(current_lines)),
                        )
                    )
                current_heading = self._clean_heading_text(match.group(2))
                current_depth = self._infer_structural_depth(current_heading, len(match.group(1)))
                current_lines = [f"[heading_depth={current_depth}]"]
                continue
            current_lines.append(line)

        if current_lines:
            sections.append(
                MarkdownSection(
                    raw_heading=current_heading,
                    normalized_heading=self._normalize_heading_label(current_heading),
                    structural_depth=current_depth,
                    content=self._strip_heading_markers("\n".join(current_lines)),
                )
            )
        return [section for section in sections if section.content]

    def _clean_heading_text(self, heading: str) -> str:
        """去掉 Markdown 标题中的强调符号等噪声。"""
        cleaned = heading.replace("*", "").replace("`", "").strip()
        return normalize_whitespace(cleaned)

    def _normalize_heading_label(self, heading: str) -> str:
        """把 Markdown 标题转成更稳定的标签文本。"""
        label = self._clean_heading_text(heading).lower()
        label = re.sub(r"^\d+(?:\.\d+)*[\.)]?\s*", "", label)
        label = re.sub(r"^[ivxlcdm]+(?:[\.)]|\s)\s*", "", label)
        label = re.sub(r"^[a-z][\.)]?\s+", "", label)
        label = re.sub(r"[^a-z0-9 ]+", " ", label)
        return normalize_whitespace(label)

    def _infer_structural_depth(self, heading: str, markdown_level: int) -> int:
        """从标题编号推断章节层级，弥补 PDF->Markdown 后标题级别不稳定的问题。"""
        cleaned = self._clean_heading_text(heading)
        digit_match = re.match(r"^(\d+(?:\.\d+)*)[\.)]?\s+", cleaned)
        if digit_match:
            return digit_match.group(1).count(".") + 1
        roman_match = re.match(r"^([IVXLCDM]+)[\.)]?\s+", cleaned, re.IGNORECASE)
        if roman_match:
            return 1
        alpha_match = re.match(r"^([A-Z])[\.)]?\s+", cleaned)
        if alpha_match:
            return 2
        return markdown_level

    def _extract_method_section_hierarchy(self, sections: list[MarkdownSection]) -> str:
        """提取方法主章节，并聚合其下属子章节。"""
        root_index = self._find_method_root_index(sections)
        if root_index is None:
            return ""

        root = sections[root_index]
        selected_chunks: list[str] = []
        for index in range(root_index, len(sections)):
            section = sections[index]
            if index > root_index and section.structural_depth <= root.structural_depth:
                break
            if section.normalized_heading == "references":
                break
            selected_chunks.append(self._format_section_chunk(section))

        return normalize_whitespace("\n\n".join(chunk for chunk in selected_chunks if chunk))

    def _find_method_root_index(self, sections: list[MarkdownSection]) -> int | None:
        """找到最适合作为方法论根节点的章节。"""
        best_index: int | None = None
        best_score = -1
        for index, section in enumerate(sections):
            heading = section.normalized_heading
            if not any(candidate in heading for candidate in METHOD_HEADING_HINTS):
                continue
            score = 0
            if any(token in heading for token in ("methodology", "method", "methods")):
                score += 4
            if any(token in heading for token in ("approach", "framework", "implementation", "model", "system")):
                score += 2
            if section.structural_depth == 1:
                score += 3
            elif section.structural_depth == 2:
                score += 1
            if index < 20:
                score += 1
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _format_section_chunk(self, section: MarkdownSection) -> str:
        """保留小节标题，便于 LLM 理解方法结构。"""
        title = self._clean_heading_text(section.raw_heading)
        if title == "document_start":
            return section.content
        return f"{title}\n{section.content}"

    def _pick_first_matching_block(self, blocks: list[tuple[str, str]], candidate_headings: set[str]) -> str:
        """兼容旧逻辑的占位实现。"""
        for heading, content in blocks:
            if any(candidate in heading for candidate in candidate_headings):
                normalized = self._strip_heading_markers(content)
                if normalized:
                    return normalized
        return ""

    def _strip_heading_markers(self, text: str) -> str:
        """移除内部用的 heading level 标记。"""
        lines = [
            line
            for line in text.splitlines()
            if not line.startswith("[heading_level=") and not line.startswith("[heading_depth=")
        ]
        return "\n".join(lines).strip()

    def _extract_method_content(self, text: str) -> str:
        """从 Markdown 正文中启发式提取方法论相关段落。"""
        paragraphs = [normalize_whitespace(item) for item in text.split("\n\n") if normalize_whitespace(item)]
        method_keywords = (
            "we propose",
            "we present",
            "our method",
            "our approach",
            "framework",
            "architecture",
            "algorithm",
            "training",
            "inference",
            "module",
        )
        matched = [paragraph for paragraph in paragraphs if any(keyword in paragraph.lower() for keyword in method_keywords)]
        if matched:
            return " ".join(matched[:4])

        sentences = self._sentences(text)
        if not sentences:
            return ""
        midpoint = max(0, len(sentences) // 3)
        return " ".join(sentences[midpoint : midpoint + 12])

    @staticmethod
    def _sentences(text: str) -> list[str]:
        """按句子粗切分文本，服务于启发式抽取。"""
        candidates = re.split(r"(?<=[.!?])\s+", text)
        return [normalize_whitespace(item) for item in candidates if normalize_whitespace(item)]

    def _build_method_like(self, text: str) -> str:
        """优先抓取带方法关键词的句子，否则退回到中段。"""
        keywords = ("we propose", "we present", "our method", "our approach", "framework", "model", "architecture")
        for sentence in self._sentences(text):
            if any(keyword in sentence.lower() for keyword in keywords):
                return sentence
        sentences = self._sentences(text)
        return " ".join(sentences[1:4]) if len(sentences) > 1 else text

    def _truncate_text(self, text: str, limit: Optional[int] = None) -> str:
        """截断过长章节，避免给 LLM 传入过多噪声。"""
        normalized = normalize_whitespace(text)
        char_limit = limit or self.max_section_chars
        if len(normalized) <= char_limit:
            return normalized
        return normalized[:char_limit].rsplit(" ", 1)[0] + " ..."
