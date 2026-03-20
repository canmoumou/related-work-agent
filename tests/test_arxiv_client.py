"""arXiv 客户端测试。"""

from pathlib import Path

from app.services.arxiv_client import ArxivClient


def test_build_query_deduplicates_terms() -> None:
    """验证 query 构造会自动去重重复 term。"""
    query = ArxivClient.build_query(
        ["3D open-world semantic segmentation", "3D open-world semantic segmentation", "open world"]
    )

    assert query == 'all:"3D open-world semantic segmentation" OR all:"open world"'


def test_arxiv_result_schema() -> None:
    """验证 Atom feed 能被解析成标准论文结构。"""
    xml_text = Path("tests/fixtures/arxiv_sample.xml").read_text(encoding="utf-8")
    client = ArxivClient()

    papers = client.parse_feed(xml_text)

    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2401.12345v1"
    assert paper.title
    assert paper.abstract
    assert paper.pdf_url is not None
    assert paper.abs_url is not None
    assert paper.categories == ["cs.CL"]
