"""LangGraph 工作流状态定义。"""

from __future__ import annotations

from typing import Any, TypedDict

from app.models.schemas import (
    CandidatePaper,
    IntentDecomposition,
    MethodCard,
    PaperSections,
    ParagraphEvidence,
    QueryPlan,
    RankedPaper,
    ThemeCluster,
    VerificationReport,
)


class WorkflowState(TypedDict, total=False):
    """定义各工作流节点之间共享的状态字段。"""
    topic: str
    max_papers: int
    expanded_intents: IntentDecomposition
    queries: list[QueryPlan]
    candidate_papers: list[CandidatePaper]
    merged_papers: list[CandidatePaper]
    selected_papers: list[RankedPaper]
    paper_sections: list[PaperSections]
    method_cards: list[MethodCard]
    clusters: list[ThemeCluster]
    related_work: str
    evidence_map: list[ParagraphEvidence]
    verification_report: VerificationReport
    debug: dict[str, Any]
