"""论文全文方法抽取测试。"""

from app.models.schemas import CandidatePaper
from app.services.paper_reader import PaperReader


def test_extract_from_markdown_prefers_method_section(tmp_path) -> None:
    """验证 Markdown 存在方法章节时，会聚合其子章节内容。"""
    reader = PaperReader()
    markdown_path = tmp_path / "paper.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Introduction",
                "Retrieval augmented generation improves factuality but still suffers from noisy evidence.",
                "",
                "## 3 Method",
                "We propose a two-stage framework with a dense retriever, a reranker, and a constrained generator.",
                "",
                "## 3.1 Retriever Design",
                "The retriever is aligned with downstream supervision and trained with hard negatives.",
                "",
                "## 3.2 Generator Design",
                "The generator performs constrained decoding with retrieved evidence.",
                "",
                "## 4 Experiments",
                "Results show consistent gains on open-domain QA benchmarks.",
            ]
        ),
        encoding="utf-8",
    )

    sections, notes = reader._extract_from_markdown(str(markdown_path))

    labels = [section.label for section in sections]
    assert labels == ["method_like"]
    assert any("dense retriever" in section.content.lower() for section in sections if section.label == "method_like")
    assert any("hard negatives" in section.content.lower() for section in sections if section.label == "method_like")
    assert any("constrained decoding" in section.content.lower() for section in sections if section.label == "method_like")
    assert all("open-domain qa benchmarks" not in section.content.lower() for section in sections if section.label == "method_like")
    assert any("Markdown cache" in note for note in notes)


def test_read_falls_back_to_abstract_without_pdf() -> None:
    """验证缺少 PDF 时仍只返回 method_like 回退路径。"""
    reader = PaperReader()
    paper = CandidatePaper(
        arxiv_id="2401.12345v1",
        title="A Test Paper",
        authors=["Test Author"],
        abstract="We propose a modular retrieval framework and demonstrate improved grounding.",
        categories=["cs.CL"],
        pdf_url=None,
        abs_url=None,
    )

    result = reader._extract_sections(paper, markdown_path=None)

    assert result.paper_id == "2401.12345v1"
    assert [section.label for section in result.sections] == ["method_like"]
    assert any("abstract" in note.lower() for note in result.extraction_notes)
