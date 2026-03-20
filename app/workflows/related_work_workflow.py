"""Related Work 主工作流定义。"""

from __future__ import annotations

import logging
from collections import defaultdict

from langgraph.graph import END, START, StateGraph

from app.models.schemas import (
    CandidatePaper,
    IntentDecomposition,
    IntentDecompositionPayload,
    LLMMessage,
    MethodCard,
    MethodCardPayload,
    PaperSection,
    ParagraphEvidence,
    QueryPlan,
    RelatedWorkPayload,
    ThemeCluster,
    VerificationPayload,
    VerificationReport,
    WorkflowDebugResponse,
    WorkflowRunResponse,
)
from app.models.workflow_state import WorkflowState
from app.services.arxiv_client import ArxivClient
from app.services.llm_client import LLMClient
from app.services.paper_reader import PaperReader
from app.services.prompt_service import PromptService
from app.services.reranker import PaperReranker
from app.utils.text_utils import normalize_whitespace, safe_slug, tokenize


LOGGER = logging.getLogger(__name__)


class RelatedWorkWorkflow:
    """把论文检索、阅读、抽取与写作串成一条可观测工作流。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        arxiv_client: ArxivClient | None = None,
        paper_reader: PaperReader | None = None,
        reranker: PaperReranker | None = None,
        prompt_service: PromptService | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.arxiv_client = arxiv_client or ArxivClient()
        self.paper_reader = paper_reader or PaperReader()
        self.reranker = reranker or PaperReranker()
        self.prompt_service = prompt_service or PromptService()
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 有向工作流。"""
        graph = StateGraph(WorkflowState)
        graph.add_node("intent_decomposer", self.intent_decomposer)
        graph.add_node("query_planner", self.query_planner)
        graph.add_node("arxiv_retriever", self.arxiv_retriever)
        graph.add_node("candidate_merger", self.candidate_merger)
        graph.add_node("paper_reranker", self.paper_reranker)
        graph.add_node("paper_reader", self.paper_reader_node)
        graph.add_node("method_extractor", self.method_extractor)
        graph.add_node("theme_clusterer", self.theme_clusterer)
        graph.add_node("related_work_writer", self.related_work_writer)
        graph.add_node("evidence_mapper", self.evidence_mapper)

        graph.add_edge(START, "intent_decomposer")
        graph.add_edge("intent_decomposer", "query_planner")
        graph.add_edge("query_planner", "arxiv_retriever")
        graph.add_edge("arxiv_retriever", "candidate_merger")
        graph.add_edge("candidate_merger", "paper_reranker")
        graph.add_edge("paper_reranker", "paper_reader")
        graph.add_edge("paper_reader", "method_extractor")
        graph.add_edge("method_extractor", "theme_clusterer")
        graph.add_edge("theme_clusterer", "related_work_writer")
        graph.add_edge("related_work_writer", "evidence_mapper")
        graph.add_edge("evidence_mapper", END)
        return graph.compile()

    async def run(self, topic: str, max_papers: int = 10, debug: bool = False) -> WorkflowRunResponse | WorkflowDebugResponse:
        """执行完整工作流，并按模式返回标准结果或调试结果。"""
        result = await self.graph.ainvoke(
            {
                "topic": topic,
                "max_papers": max_papers,
                "debug": {},
            }
        )
        if debug:
            return WorkflowDebugResponse(
                topic=result["topic"],
                expanded_intents=result["expanded_intents"],
                queries=result["queries"],
                candidate_papers=result.get("merged_papers", []),
                selected_papers=result["selected_papers"],
                paper_sections=result["paper_sections"],
                method_cards=result["method_cards"],
                clusters=result["clusters"],
                related_work=result["related_work"],
                evidence_map=result["evidence_map"],
                verification_report=result["verification_report"],
                debug=result.get("debug", {}),
            )

        return WorkflowRunResponse(
            topic=result["topic"],
            expanded_intents=result["expanded_intents"],
            queries=result["queries"],
            selected_papers=result["selected_papers"],
            method_cards=result["method_cards"],
            clusters=result["clusters"],
            related_work=result["related_work"],
            evidence_map=result["evidence_map"],
            verification_report=result["verification_report"],
        )

    async def intent_decomposer(self, state: WorkflowState) -> WorkflowState:
        """把用户输入主题拆成规范主题、子方向和别名。"""
        LOGGER.info("IntentDecomposer started")
        topic = state["topic"]
        fallback = self._fallback_intent_decomposition(topic)

        if self.llm_client.is_configured():
            prompt = self.prompt_service.render("intent_decompose.txt", topic=topic)
            payload = await self.llm_client.chat_json(
                messages=[
                    LLMMessage(role="system", content="Return strict JSON only."),
                    LLMMessage(role="user", content=prompt),
                ],
                model_cls=IntentDecompositionPayload,
                fallback_factory=lambda: IntentDecompositionPayload.model_validate(fallback.model_dump()),
            )
            result = IntentDecomposition.model_validate(payload.model_dump())
        else:
            result = fallback

        LOGGER.info("IntentDecomposer finished with %s subtopics", len(result.subtopics))
        return {
            "expanded_intents": result,
            "debug": {"intent_decomposer": result.model_dump()},
        }

    async def query_planner(self, state: WorkflowState) -> WorkflowState:
        """基于主题拆解结果生成多路 arXiv 查询计划，并去重。"""
        LOGGER.info("QueryPlanner started")
        expanded = state["expanded_intents"]
        candidate_terms = [expanded.normalized_topic, *expanded.subtopics[:4], *expanded.aliases[:2]]
        query_plans: list[QueryPlan] = []
        seen_queries: set[str] = set()

        for index, term in enumerate(candidate_terms[:6]):
            arxiv_query = self.arxiv_client.build_query([expanded.normalized_topic, term])
            if arxiv_query in seen_queries:
                continue
            seen_queries.add(arxiv_query)
            query_plans.append(
                QueryPlan(
                    name=f"query_{len(query_plans) + 1}_{safe_slug(term)}",
                    intent=f"Cover topic aspect: {term}",
                    arxiv_query=arxiv_query,
                    expected_focus=term,
                )
            )

        LOGGER.info("QueryPlanner finished with %s query plans", len(query_plans))
        return {
            "queries": query_plans,
            "debug": {**state.get("debug", {}), "query_planner": [plan.model_dump() for plan in query_plans]},
        }

    async def arxiv_retriever(self, state: WorkflowState) -> WorkflowState:
        """按查询计划从 arXiv 拉取候选论文。"""
        LOGGER.info("ArxivRetriever started")
        query_plans = state["queries"]
        max_results_per_query = max(5, min(8, state["max_papers"]))
        candidates: list[CandidatePaper] = []

        for plan in query_plans:
            try:
                papers = await self.arxiv_client.search(plan.arxiv_query, max_results=max_results_per_query)
                for paper in papers:
                    paper.source_queries.append(plan.name)
                candidates.extend(papers)
            except Exception as exc:
                LOGGER.exception("Arxiv query failed for %s: %s", plan.name, exc)

        LOGGER.info("ArxivRetriever finished with %s raw candidates", len(candidates))
        return {
            "candidate_papers": candidates,
            "debug": {**state.get("debug", {}), "arxiv_retriever_count": len(candidates)},
        }

    async def candidate_merger(self, state: WorkflowState) -> WorkflowState:
        """按 arXiv ID 合并重复论文。"""
        LOGGER.info("CandidateMerger started")
        merged: dict[str, CandidatePaper] = {}
        for paper in state.get("candidate_papers", []):
            existing = merged.get(paper.arxiv_id)
            if existing is None:
                merged[paper.arxiv_id] = paper
                continue
            existing.source_queries = sorted(set(existing.source_queries + paper.source_queries))
            if len(paper.abstract) > len(existing.abstract):
                existing.abstract = paper.abstract

        merged_papers = list(merged.values())
        LOGGER.info("CandidateMerger finished with %s unique papers", len(merged_papers))
        return {
            "merged_papers": merged_papers,
            "debug": {**state.get("debug", {}), "candidate_merger_count": len(merged_papers)},
        }

    async def paper_reranker(self, state: WorkflowState) -> WorkflowState:
        """基于多维代理分数重排候选论文。"""
        LOGGER.info("PaperReranker started")
        selected = self.reranker.rerank(
            papers=state.get("merged_papers", []),
            topic=state["expanded_intents"].normalized_topic,
            subtopics=state["expanded_intents"].subtopics,
            max_papers=state["max_papers"],
        )
        LOGGER.info("PaperReranker finished with %s selected papers", len(selected))
        return {
            "selected_papers": selected,
            "debug": {**state.get("debug", {}), "paper_reranker_count": len(selected)},
        }

    async def paper_reader_node(self, state: WorkflowState) -> WorkflowState:
        """抽取每篇论文可供后续分析的 section 内容。"""
        LOGGER.info("PaperReader started")
        paper_sections = await self.paper_reader.read(state.get("selected_papers", []))
        LOGGER.info("PaperReader finished with %s section bundles", len(paper_sections))
        return {
            "paper_sections": paper_sections,
            "debug": {**state.get("debug", {}), "paper_reader_count": len(paper_sections)},
        }

    async def method_extractor(self, state: WorkflowState) -> WorkflowState:
        """把论文片段整理成结构化 method card。"""
        LOGGER.info("MethodExtractor started")
        cards: list[MethodCard] = []
        for bundle in state.get("paper_sections", []):
            fallback = self._fallback_method_card(bundle.paper_id, bundle.title, bundle.sections)
            if self.llm_client.is_configured():
                prompt = self.prompt_service.render(
                    "method_extract.txt",
                    title=bundle.title,
                    sections="\n\n".join(f"[{section.label}] {section.content}" for section in bundle.sections),
                )
                payload = await self.llm_client.chat_json(
                    messages=[
                        LLMMessage(role="system", content="Return strict JSON only."),
                        LLMMessage(role="user", content=prompt),
                    ],
                    model_cls=MethodCardPayload,
                    fallback_factory=lambda value=fallback: MethodCardPayload.model_validate(value.model_dump(exclude={"paper_id", "title"})),
                )
                card = MethodCard(
                    paper_id=bundle.paper_id,
                    title=bundle.title,
                    **payload.model_dump(),
                )
            else:
                card = fallback
            cards.append(card)

        LOGGER.info("MethodExtractor finished with %s method cards", len(cards))
        return {
            "method_cards": cards,
            "debug": {**state.get("debug", {}), "method_extractor_count": len(cards)},
        }

    async def theme_clusterer(self, state: WorkflowState) -> WorkflowState:
        """按技术路线对 method cards 做轻量聚类。"""
        LOGGER.info("ThemeClusterer started")
        subtopics = state["expanded_intents"].subtopics or [state["expanded_intents"].normalized_topic]
        buckets: dict[str, list[MethodCard]] = defaultdict(list)

        for card in state.get("method_cards", []):
            # 当前版本使用关键词重叠做一个轻量聚类，优先保证流程稳定可运行。
            best_theme = max(subtopics, key=lambda subtopic: self._keyword_overlap_score(subtopic, f"{card.title} {card.method_summary} {card.core_idea}"))
            buckets[best_theme].append(card)

        clusters: list[ThemeCluster] = []
        for index, (theme, cards) in enumerate(buckets.items(), start=1):
            keywords = sorted({token for card in cards for token in tokenize(" ".join(card.key_modules)) if len(token) > 3})[:5]
            clusters.append(
                ThemeCluster(
                    cluster_id=f"cluster_{index}",
                    theme=theme,
                    summary=f"Papers in this cluster primarily focus on {theme}.",
                    paper_ids=[card.paper_id for card in cards],
                    representative_keywords=keywords,
                )
            )

        if len(clusters) > 5:
            clusters = clusters[:5]
        if len(clusters) < 3 and state.get("method_cards"):
            clusters = self._pad_clusters(clusters, state["method_cards"])

        LOGGER.info("ThemeClusterer finished with %s clusters", len(clusters))
        return {
            "clusters": clusters,
            "debug": {**state.get("debug", {}), "theme_clusterer_count": len(clusters)},
        }

    async def related_work_writer(self, state: WorkflowState) -> WorkflowState:
        """基于 clusters 和 method cards 生成 related work。"""
        LOGGER.info("RelatedWorkWriter started")
        fallback_payload = self._fallback_related_work(
            topic=state["topic"],
            clusters=state.get("clusters", []),
            method_cards=state.get("method_cards", []),
        )

        if self.llm_client.is_configured():
            prompt = self.prompt_service.render(
                "related_work_write.txt",
                topic=state["topic"],
                clusters=[cluster.model_dump() for cluster in state.get("clusters", [])],
                method_cards=[card.model_dump() for card in state.get("method_cards", [])],
            )
            payload = await self.llm_client.chat_json(
                messages=[
                    LLMMessage(role="system", content="Return strict JSON only."),
                    LLMMessage(role="user", content=prompt),
                ],
                model_cls=RelatedWorkPayload,
                fallback_factory=lambda value=fallback_payload: value,
            )
        else:
            payload = fallback_payload

        LOGGER.info("RelatedWorkWriter finished")
        return {
            "related_work": payload.related_work,
            "debug": {**state.get("debug", {}), "paragraph_summaries": payload.paragraph_summaries},
        }

    async def evidence_mapper(self, state: WorkflowState) -> WorkflowState:
        """把 related work 段落映射回论文证据，并给出校验报告。"""
        LOGGER.info("EvidenceMapper started")
        fallback_payload = self._fallback_verification(
            state["related_work"],
            state.get("method_cards", []),
        )

        if self.llm_client.is_configured():
            prompt = self.prompt_service.render(
                "verify.txt",
                related_work=state["related_work"],
                method_cards=[card.model_dump() for card in state.get("method_cards", [])],
            )
            payload = await self.llm_client.chat_json(
                messages=[
                    LLMMessage(role="system", content="Return strict JSON only."),
                    LLMMessage(role="user", content=prompt),
                ],
                model_cls=VerificationPayload,
                fallback_factory=lambda value=fallback_payload: value,
            )
        else:
            payload = fallback_payload

        LOGGER.info("EvidenceMapper finished with %s paragraph mappings", len(payload.evidence_map))
        return {
            "evidence_map": payload.evidence_map,
            "verification_report": payload.verification_report,
            "debug": {**state.get("debug", {}), "verification": payload.verification_report.model_dump()},
        }

    def _fallback_intent_decomposition(self, topic: str) -> IntentDecomposition:
        """当 LLM 不可用时，使用启发式方式做基础方向拆解。"""
        tokens = [token for token in tokenize(topic) if len(token) > 2]
        unique_tokens = list(dict.fromkeys(tokens))
        subtopics = [normalize_whitespace(" ".join(unique_tokens[i : i + 2])) for i in range(0, min(len(unique_tokens), 8), 2)]
        aliases = [topic, topic.replace("-", " "), topic.upper()[:40]]
        related_phrases = [f"{topic} methods", f"{topic} applications", f"{topic} benchmark", f"{topic} framework"]
        return IntentDecomposition(
            normalized_topic=normalize_whitespace(topic),
            subtopics=[item for item in subtopics if item][:4] or [normalize_whitespace(topic)],
            aliases=[item for item in aliases if item][:3],
            related_phrases=related_phrases[:4],
        )

    def _fallback_method_card(self, paper_id: str, title: str, sections: list[PaperSection]) -> MethodCard:
        """在无法稳定调用 LLM 时，保守生成 method card。"""
        abstract = next((section.content for section in sections if section.label == "abstract"), "")
        method_like = next((section.content for section in sections if section.label == "method_like"), abstract)
        contribution_like = next((section.content for section in sections if section.label == "contribution_like"), abstract)
        modules = [token for token in tokenize(method_like) if len(token) > 5][:5]
        return MethodCard(
            paper_id=paper_id,
            title=title,
            problem=abstract[:240] or "Problem statement is only partially observable from arXiv metadata.",
            core_idea=method_like[:240] or "Core idea could not be extracted reliably; falling back to abstract.",
            method_summary=method_like[:360] or abstract[:360],
            key_modules=modules,
            training_or_inference="Not explicitly available from metadata; inferred conservatively from abstract.",
            claimed_contributions=[contribution_like[:180] or abstract[:180]],
            limitations=["The current extraction relies on abstract-level evidence and heuristic sectioning."],
            evidence_spans=[
                {
                    "section_label": sections[0].label if sections else "abstract",
                    "quote": (method_like or abstract)[:240],
                    "rationale": "Fallback extraction is tied to the provided abstract-derived section.",
                }
            ],
        )

    def _fallback_related_work(self, topic: str, clusters: list[ThemeCluster], method_cards: list[MethodCard]) -> RelatedWorkPayload:
        """在没有 LLM 时，用模板化方式生成保守的 related work。"""
        paragraphs: list[str] = []
        summaries: list[str] = []
        card_lookup = {card.paper_id: card for card in method_cards}

        for cluster in clusters:
            cards = [card_lookup[paper_id] for paper_id in cluster.paper_ids if paper_id in card_lookup]
            if not cards:
                continue
            paragraph = (
                f"In research on {topic}, one visible line of work centers on {cluster.theme}. "
                f"Representative papers in this group commonly emphasize {', '.join(cluster.representative_keywords[:3]) or 'closely related technical modules'}. "
                f"Across these studies, the methods tend to share the following pattern: "
                f"{' '.join(card.method_summary for card in cards[:2])} "
                f"At the same time, the available evidence suggests trade-offs around "
                f"{'; '.join(cards[0].limitations[:2]) if cards[0].limitations else 'evaluation scope and abstraction level'}."
            )
            paragraphs.append(paragraph)
            summaries.append(f"Cluster {cluster.cluster_id} discusses {cluster.theme}.")

        if not paragraphs:
            paragraphs.append(
                f"The current workflow retrieved arXiv papers related to {topic}, but the evidence remains sparse. "
                f"The generated related work therefore stays conservative and focuses on method-level commonalities visible from abstracts."
            )
            summaries.append("Conservative fallback paragraph based on abstract-level evidence.")

        return RelatedWorkPayload(
            related_work="\n\n".join(paragraphs),
            paragraph_summaries=summaries,
        )

    def _fallback_verification(self, related_work: str, method_cards: list[MethodCard]) -> VerificationPayload:
        """基于词项重叠做一个保守的证据校验。"""
        paragraphs = [normalize_whitespace(item) for item in related_work.split("\n\n") if normalize_whitespace(item)]
        evidence_map: list[ParagraphEvidence] = []
        warnings: list[str] = []

        for index, paragraph in enumerate(paragraphs):
            matched_ids = [
                card.paper_id
                for card in method_cards
                if self._keyword_overlap_score(paragraph, f"{card.title} {card.method_summary} {' '.join(card.claimed_contributions)}") > 0
            ]
            unsupported_claims = []
            if not matched_ids:
                unsupported_claims.append("No strong paper-level lexical evidence was found for this paragraph.")
                warnings.append(f"Paragraph {index} may be weakly supported.")

            evidence_map.append(
                ParagraphEvidence(
                    paragraph_index=index,
                    paragraph_text=paragraph,
                    paper_ids=matched_ids,
                    supporting_claims=[f"Supported by method cards from papers: {', '.join(matched_ids[:5])}"] if matched_ids else [],
                    unsupported_claims=unsupported_claims,
                )
            )

        verification_report = VerificationReport(
            supported_paragraphs=sum(1 for item in evidence_map if item.paper_ids and not item.unsupported_claims),
            flagged_paragraphs=sum(1 for item in evidence_map if item.unsupported_claims),
            warnings=warnings,
            notes=[
                "Verification is conservative and only uses the provided method cards.",
                "Without stable full-text parsing, some nuanced claims may remain weakly grounded.",
            ],
        )
        return VerificationPayload(
            evidence_map=evidence_map,
            verification_report=verification_report,
        )

    @staticmethod
    def _keyword_overlap_score(left: str, right: str) -> int:
        """计算两个文本的关键词重叠数量。"""
        return len(set(tokenize(left)) & set(tokenize(right)))

    def _pad_clusters(self, clusters: list[ThemeCluster], method_cards: list[MethodCard]) -> list[ThemeCluster]:
        """当聚类数量不足时，补足到更适合 related work 写作的规模。"""
        if len(clusters) == 1 and len(method_cards) >= 3:
            # 只有一个大簇时，按顺序拆成多个小簇，避免 related work 只有单段输出。
            target_count = min(3, len(method_cards))
            split_groups: list[list[MethodCard]] = [[] for _ in range(target_count)]
            for index, card in enumerate(method_cards):
                split_groups[index % target_count].append(card)

            rebuilt_clusters: list[ThemeCluster] = []
            for index, cards in enumerate(split_groups, start=1):
                if not cards:
                    continue
                rebuilt_clusters.append(
                    ThemeCluster(
                        cluster_id=f"cluster_{index}",
                        theme=cards[0].title if len(cards) == 1 else f"technical_line_{index}",
                        summary="Heuristic split cluster created to preserve topic diversity in the output.",
                        paper_ids=[card.paper_id for card in cards],
                        representative_keywords=sorted(
                            {
                                token
                                for card in cards
                                for token in tokenize(" ".join(card.key_modules))
                                if len(token) > 3
                            }
                        )[:5],
                    )
                )
            return rebuilt_clusters

        existing_ids = {paper_id for cluster in clusters for paper_id in cluster.paper_ids}
        next_index = len(clusters) + 1
        for card in method_cards:
            if card.paper_id in existing_ids:
                continue
            clusters.append(
                ThemeCluster(
                    cluster_id=f"cluster_{next_index}",
                    theme=card.title,
                    summary="Single-paper cluster created to preserve diversity in the output.",
                    paper_ids=[card.paper_id],
                    representative_keywords=card.key_modules[:5],
                )
            )
            next_index += 1
            if len(clusters) >= 3:
                break
        return clusters
