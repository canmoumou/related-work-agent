"""方法卡片与 related work 引用结构测试。"""

from app.models.schemas import PaperSection, ThemeCluster
from app.workflows.related_work_workflow import RelatedWorkWorkflow


def test_method_card_schema() -> None:
    """验证仅使用 method_like 时 fallback method card 的字段结构完整。"""
    workflow = RelatedWorkWorkflow()
    card = workflow._fallback_method_card(
        paper_id="2401.12345v1",
        title="A Test Paper",
        sections=[
            PaperSection(label="method_like", content="Our method uses a retriever, reranker, and generator.", source="test"),
        ],
    )

    assert card.paper_id == "2401.12345v1"
    assert card.title == "A Test Paper"
    assert isinstance(card.key_modules, list)
    assert len(card.evidence_spans) >= 1


def test_fallback_related_work_contains_valid_citations() -> None:
    """验证 fallback related work 会输出带精确引用的段落结构。"""
    workflow = RelatedWorkWorkflow()
    card = workflow._fallback_method_card(
        paper_id="2401.12345v1",
        title="A Test Paper",
        sections=[
            PaperSection(
                label="method_like",
                content="Our method uses a dense retriever, a reranker, and a constrained generator.",
                source="test",
            ),
        ],
    )

    payload = workflow._fallback_related_work(
        topic="retrieval augmented generation",
        clusters=[
            ThemeCluster(
                cluster_id="cluster_1",
                theme="retrieval modeling",
                summary="test cluster",
                paper_ids=[card.paper_id],
                representative_keywords=["retriever", "generator"],
            )
        ],
        method_cards=[card],
    )

    assert payload.paragraphs
    assert payload.paragraphs[0].citations
    assert payload.paragraphs[0].citations[0].paper_id == "2401.12345v1"
    assert payload.paragraphs[0].citations[0].quote == card.evidence_spans[0].quote
    assert "[2401.12345v1]" in payload.related_work
