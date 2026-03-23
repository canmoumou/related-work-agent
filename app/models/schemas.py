"""工作流输入输出与中间结构的 Pydantic Schema。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class WorkflowRunRequest(BaseModel):
    """工作流运行请求。"""
    topic: str = Field(min_length=3, description="User input topic.")
    max_papers: int = Field(default=10, ge=1, le=20)


class IntentDecomposition(BaseModel):
    """方向拆解结果。"""
    normalized_topic: str
    subtopics: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    related_phrases: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """单条 arXiv 查询计划。"""
    name: str
    intent: str
    arxiv_query: str
    expected_focus: str


class CandidatePaper(BaseModel):
    """arXiv 候选论文的统一结构。"""
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str
    categories: list[str] = Field(default_factory=list)
    published: datetime | None = None
    updated: datetime | None = None
    pdf_url: HttpUrl | None = None
    abs_url: HttpUrl | None = None
    source_queries: list[str] = Field(default_factory=list)


class RankedPaper(CandidatePaper):
    """带有多维排序分数的候选论文。"""
    semantic_relevance_score: float = 0.0
    coverage_score: float = 0.0
    centrality_proxy_score: float = 0.0
    diversity_adjustment: float = 0.0
    metadata_quality_score: float = 0.0
    final_rank_score: float = 0.0
    selected_reason: str = ""


class PaperSection(BaseModel):
    """论文片段的结构化表示。"""
    label: str
    content: str
    source: str


class PaperSections(BaseModel):
    """单篇论文抽取出的多个 section。"""
    paper_id: str
    title: str
    sections: list[PaperSection] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)


class EvidenceSpan(BaseModel):
    """方法卡片中引用的证据片段。"""
    section_label: str
    quote: str
    rationale: str


class RelatedWorkCitation(BaseModel):
    """related work 段落中展示的引用证据。"""
    paper_id: str
    title: str
    section_label: str
    quote: str
    rationale: str


class MethodCard(BaseModel):
    """从论文方法论中抽取出的结构化卡片。"""
    paper_id: str
    title: str
    problem: str
    core_idea: str
    method_summary: str
    key_modules: list[str] = Field(default_factory=list)
    training_or_inference: str
    claimed_contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)


class ThemeCluster(BaseModel):
    """按技术路线聚合后的主题簇。"""
    cluster_id: str
    theme: str
    summary: str
    paper_ids: list[str] = Field(default_factory=list)
    representative_keywords: list[str] = Field(default_factory=list)


class RelatedWorkParagraph(BaseModel):
    """related work 的单段正文与其引用。"""
    paragraph_index: int
    paragraph_text: str
    citations: list[RelatedWorkCitation] = Field(default_factory=list)


class ParagraphEvidence(BaseModel):
    """related work 段落与论文证据之间的映射。"""
    paragraph_index: int
    paragraph_text: str
    paper_ids: list[str] = Field(default_factory=list)
    supporting_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """证据校验汇总报告。"""
    supported_paragraphs: int = 0
    flagged_paragraphs: int = 0
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class WorkflowRunResponse(BaseModel):
    """标准工作流响应。"""
    topic: str
    expanded_intents: IntentDecomposition
    queries: list[QueryPlan]
    selected_papers: list[RankedPaper]
    method_cards: list[MethodCard]
    clusters: list[ThemeCluster]
    related_work: str
    related_work_paragraphs: list[RelatedWorkParagraph]
    evidence_map: list[ParagraphEvidence]
    verification_report: VerificationReport


class WorkflowDebugResponse(WorkflowRunResponse):
    """调试模式工作流响应，包含中间状态。"""
    candidate_papers: list[CandidatePaper]
    paper_sections: list[PaperSections]
    debug: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    app_name: str


class LLMMessage(BaseModel):
    """统一的聊天消息结构。"""
    role: Literal["system", "user", "assistant"]
    content: str


class IntentDecompositionPayload(BaseModel):
    """LLM 返回的方向拆解结构。"""
    normalized_topic: str
    subtopics: list[str]
    aliases: list[str]
    related_phrases: list[str]

    @field_validator("subtopics", "aliases", "related_phrases")
    @classmethod
    def trim_values(cls, value: list[str]) -> list[str]:
        """去掉空字符串和多余空白。"""
        return [item.strip() for item in value if item.strip()]


class MethodCardPayload(BaseModel):
    """LLM 返回的方法卡片结构。"""
    problem: str
    core_idea: str
    method_summary: str
    key_modules: list[str]
    training_or_inference: str
    claimed_contributions: list[str]
    limitations: list[str]
    evidence_spans: list[EvidenceSpan]


class RelatedWorkPayload(BaseModel):
    """LLM 返回的 related work 结构。"""
    related_work: str
    paragraph_summaries: list[str]
    paragraphs: list[RelatedWorkParagraph] = Field(default_factory=list)


class VerificationPayload(BaseModel):
    """LLM 返回的证据校验结构。"""
    evidence_map: list[ParagraphEvidence]
    verification_report: VerificationReport
