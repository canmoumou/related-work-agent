"""方法卡片结构测试。"""

from app.models.schemas import PaperSection
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
