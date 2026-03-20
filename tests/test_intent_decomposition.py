"""方向拆解与查询规划相关测试。"""

import asyncio

from app.models.schemas import IntentDecomposition
from app.workflows.related_work_workflow import RelatedWorkWorkflow


def test_intent_decomposition_output_format() -> None:
    """验证方向拆解结果的基本结构。"""
    workflow = RelatedWorkWorkflow()
    result = asyncio.run(workflow.intent_decomposer({"topic": "retrieval augmented generation"}))
    expanded = result["expanded_intents"]

    assert expanded.normalized_topic == "retrieval augmented generation"
    assert isinstance(expanded.subtopics, list)
    assert len(expanded.subtopics) >= 1
    assert isinstance(expanded.aliases, list)
    assert isinstance(expanded.related_phrases, list)


def test_query_planner_deduplicates_equivalent_queries() -> None:
    """验证查询规划阶段不会产出重复 query。"""
    workflow = RelatedWorkWorkflow()
    result = asyncio.run(
        workflow.query_planner(
            {
                "expanded_intents": IntentDecomposition(
                    normalized_topic="3D open-world semantic segmentation",
                    subtopics=["3D open-world semantic segmentation", "open world", "semantic segmentation"],
                    aliases=["3D open-world semantic segmentation", "3D open world semantic segmentation"],
                    related_phrases=[],
                )
            }
        )
    )

    queries = result["queries"]
    assert len(queries) == len({query.arxiv_query for query in queries})
