"""论文重排与代表性评分模块。"""

from __future__ import annotations

from collections import Counter

from app.models.schemas import CandidatePaper, RankedPaper
from app.utils.text_utils import cosine_similarity, jaccard_similarity, tokenize


class PaperReranker:
    """基于多维启发式分数筛选更相关、更有代表性的论文。"""

    def rerank(
        self,
        papers: list[CandidatePaper],
        topic: str,
        subtopics: list[str],
        max_papers: int,
    ) -> list[RankedPaper]:
        """对候选论文打分、去重并选出最终结果。"""
        if not papers:
            return []

        corpus = [f"{paper.title} {paper.abstract}" for paper in papers]
        token_document_frequency = Counter()
        for doc in corpus:
            for token in set(tokenize(doc)):
                token_document_frequency[token] += 1

        ranked: list[RankedPaper] = []
        for paper in papers:
            full_text = f"{paper.title} {paper.abstract}"
            # 先计算各个可替换的代理分数，再组合成最终排序分数。
            semantic = self.semantic_relevance_score(topic, full_text, subtopics)
            coverage = self.coverage_score(full_text, subtopics)
            centrality = self.centrality_proxy_score(full_text, corpus)
            diversity = self.diversity_adjustment(full_text, [candidate.title for candidate in ranked])
            metadata_quality = self.metadata_quality_score(paper, token_document_frequency)
            final_score = self.final_rank_score(semantic, coverage, centrality, diversity, metadata_quality)

            ranked.append(
                RankedPaper(
                    **paper.model_dump(),
                    semantic_relevance_score=semantic,
                    coverage_score=coverage,
                    centrality_proxy_score=centrality,
                    diversity_adjustment=diversity,
                    metadata_quality_score=metadata_quality,
                    final_rank_score=final_score,
                    selected_reason="Balanced high relevance, coverage, centrality proxy, and diversity.",
                )
            )

        ranked.sort(key=lambda paper: paper.final_rank_score, reverse=True)
        selected: list[RankedPaper] = []
        seen_titles: list[str] = []
        seen_ids: set[str] = set()
        for paper in ranked:
            if paper.arxiv_id in seen_ids:
                continue
            if any(jaccard_similarity(paper.title, title) > 0.9 for title in seen_titles):
                continue
            paper.diversity_adjustment = self.diversity_adjustment(paper.title, seen_titles)
            paper.final_rank_score = self.final_rank_score(
                paper.semantic_relevance_score,
                paper.coverage_score,
                paper.centrality_proxy_score,
                paper.diversity_adjustment,
                paper.metadata_quality_score,
            )
            selected.append(paper)
            seen_ids.add(paper.arxiv_id)
            seen_titles.append(paper.title)
            if len(selected) >= max_papers:
                break

        selected.sort(key=lambda paper: paper.final_rank_score, reverse=True)
        return selected

    @staticmethod
    def semantic_relevance_score(topic: str, paper_text: str, subtopics: list[str]) -> float:
        """衡量论文与主主题、子主题的语义接近程度。"""
        base = cosine_similarity(topic, paper_text)
        subtopic_bonus = sum(cosine_similarity(subtopic, paper_text) for subtopic in subtopics) / max(len(subtopics), 1)
        return round((0.7 * base) + (0.3 * subtopic_bonus), 4)

    @staticmethod
    def coverage_score(paper_text: str, subtopics: list[str]) -> float:
        """衡量论文能覆盖多少重要子方向。"""
        if not subtopics:
            return 0.3
        hit_count = sum(1 for subtopic in subtopics if jaccard_similarity(subtopic, paper_text) > 0.05)
        return round(hit_count / len(subtopics), 4)

    @staticmethod
    def centrality_proxy_score(paper_text: str, corpus: list[str]) -> float:
        """用与候选集合的相似程度近似方法中心性。"""
        if not corpus:
            return 0.0
        similarity_sum = sum(cosine_similarity(paper_text, other_text) for other_text in corpus)
        return round(similarity_sum / len(corpus), 4)

    @staticmethod
    def diversity_adjustment(paper_text: str, selected_titles: list[str]) -> float:
        """对和已选论文过度相似的候选项施加惩罚。"""
        if not selected_titles:
            return 1.0
        penalty = max(jaccard_similarity(paper_text, title) for title in selected_titles)
        return round(max(0.1, 1.0 - penalty), 4)

    @staticmethod
    def metadata_quality_score(paper: CandidatePaper, token_document_frequency: Counter[str]) -> float:
        """根据标题、摘要、类别等元数据完整度给出质量分。"""
        title_quality = min(len(tokenize(paper.title)) / 12, 1.0)
        abstract_quality = min(len(tokenize(paper.abstract)) / 120, 1.0)
        category_quality = 1.0 if paper.categories else 0.3
        rarity_bonus = 0.0
        paper_tokens = set(tokenize(f"{paper.title} {paper.abstract}"))
        if paper_tokens:
            rarity_bonus = sum(1 / token_document_frequency[token] for token in paper_tokens if token_document_frequency[token]) / len(paper_tokens)
        return round(min(1.0, (0.3 * title_quality) + (0.4 * abstract_quality) + (0.2 * category_quality) + (0.1 * rarity_bonus)), 4)

    @staticmethod
    def final_rank_score(
        semantic_relevance_score: float,
        coverage_score: float,
        centrality_proxy_score: float,
        diversity_adjustment: float,
        metadata_quality_score: float,
    ) -> float:
        """把各个代理指标按权重合成为最终排序分数。"""
        score = (
            0.35 * semantic_relevance_score
            + 0.2 * coverage_score
            + 0.2 * centrality_proxy_score
            + 0.15 * diversity_adjustment
            + 0.1 * metadata_quality_score
        )
        return round(score, 4)
