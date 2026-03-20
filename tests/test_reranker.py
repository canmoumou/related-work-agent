"""论文重排测试。"""

from app.models.schemas import CandidatePaper
from app.services.reranker import PaperReranker


def build_paper(arxiv_id: str, title: str, abstract: str) -> CandidatePaper:
    """构造测试用候选论文。"""
    return CandidatePaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Author"],
        abstract=abstract,
        categories=["cs.CL"],
        pdf_url=f"http://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"http://arxiv.org/abs/{arxiv_id}",
    )


def test_reranker_count_and_deduplication() -> None:
    """验证重排后数量受限且能够按 ID 去重。"""
    papers = [
        build_paper("1", "Retrieval Augmented Generation for QA", "We propose a retrieval augmented generation model for question answering."),
        build_paper("1", "Retrieval Augmented Generation for QA", "Duplicate entry with the same paper id."),
        build_paper("2", "Dense Retrieval for Language Models", "This paper studies dense retrieval and generation pipelines."),
        build_paper("3", "Knowledge Grounded Text Generation", "A grounded generation framework using external documents."),
    ]
    reranker = PaperReranker()

    selected = reranker.rerank(
        papers=papers,
        topic="retrieval augmented generation",
        subtopics=["dense retrieval", "grounded generation"],
        max_papers=3,
    )

    assert len(selected) <= 3
    assert len({paper.arxiv_id for paper in selected}) == len(selected)
