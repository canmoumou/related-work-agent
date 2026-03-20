"""Streamlit 前端页面。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st

from app.core.logging import setup_logging
from app.workflows.related_work_workflow import RelatedWorkWorkflow


setup_logging()


@st.cache_resource
def get_workflow() -> RelatedWorkWorkflow:
    """缓存工作流实例，避免页面每次重绘都重复初始化。"""
    return RelatedWorkWorkflow()


def run_async(coro):
    """在 Streamlit 环境中安全执行异步工作流。"""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def save_result(topic: str, payload: dict) -> Path:
    """把前端运行结果保存到 output 目录。"""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{topic.lower().replace(' ', '-')[:50] or 'result'}-streamlit.json"
    output_path = output_dir / file_name
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def render_sidebar() -> tuple[str, int, bool]:
    """渲染左侧参数面板。"""
    st.sidebar.header("运行配置")
    topic = st.sidebar.text_area(
        "研究方向",
        value="retrieval augmented generation",
        help="输入你想生成 related work 的研究方向，可以是较模糊的主题。",
    )
    max_papers = st.sidebar.slider("论文数量上限", min_value=3, max_value=10, value=5)
    debug_mode = st.sidebar.checkbox("返回调试信息", value=False)
    return topic.strip(), max_papers, debug_mode


def render_overview(result: dict) -> None:
    """展示方向拆解与查询规划。"""
    expanded = result["expanded_intents"]
    st.subheader("方向拆解")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Normalized Topic**: `{expanded['normalized_topic']}`")
        st.markdown("**Subtopics**")
        st.write(expanded["subtopics"])
    with col2:
        st.markdown("**Aliases**")
        st.write(expanded["aliases"])
        st.markdown("**Related Phrases**")
        st.write(expanded["related_phrases"])

    st.subheader("查询规划")
    st.dataframe(result["queries"], width="stretch")


def render_papers(result: dict) -> None:
    """展示筛选后的论文列表。"""
    st.subheader("Top 论文")
    for index, paper in enumerate(result["selected_papers"], start=1):
        with st.expander(f"{index}. {paper['title']}", expanded=index == 1):
            st.markdown(f"**arXiv ID**: `{paper['arxiv_id']}`")
            st.markdown(f"**Final Rank Score**: `{paper['final_rank_score']}`")
            st.markdown(f"**Authors**: {', '.join(paper['authors'])}")
            st.markdown(f"**Categories**: {', '.join(paper['categories'])}")
            st.write(paper["abstract"])


def render_method_cards(result: dict) -> None:
    """展示 method cards。"""
    st.subheader("Method Cards")
    for card in result["method_cards"]:
        with st.expander(card["title"]):
            st.markdown(f"**Problem**: {card['problem']}")
            st.markdown(f"**Core Idea**: {card['core_idea']}")
            st.markdown(f"**Method Summary**: {card['method_summary']}")
            st.markdown(f"**Key Modules**: {', '.join(card['key_modules'])}")
            st.markdown(f"**Training / Inference**: {card['training_or_inference']}")
            st.markdown("**Claimed Contributions**")
            st.write(card["claimed_contributions"])
            st.markdown("**Limitations**")
            st.write(card["limitations"])


def render_related_work(result: dict) -> None:
    """展示聚类结果、related work 与证据映射。"""
    st.subheader("Related Work")
    st.write(result["related_work"])

    st.subheader("主题聚类")
    st.dataframe(result["clusters"], width="stretch")

    st.subheader("证据映射")
    st.dataframe(result["evidence_map"], width="stretch")

    st.subheader("校验报告")
    st.json(result["verification_report"], expanded=True)


def main() -> None:
    """Streamlit 页面主入口。"""
    st.set_page_config(page_title="Related Work Workflow Agent", layout="wide")
    st.title("Related Work Workflow Agent")
    st.caption("基于 arXiv + LangGraph + Qwen 的 Related Work 生成前端")

    topic, max_papers, debug_mode = render_sidebar()
    workflow = get_workflow()

    if st.button("运行 Workflow", type="primary", width="stretch"):
        if not topic:
            st.warning("请先输入研究方向。")
            return

        with st.spinner("正在运行 workflow，请稍候..."):
            # 页面当前直接调用本地 workflow，而不是通过 HTTP API 转发。
            result = run_async(workflow.run(topic=topic, max_papers=max_papers, debug=debug_mode))
            payload = result.model_dump(mode="json")
            output_path = save_result(topic, payload)

        st.success(f"运行完成，结果已保存到 `{output_path}`")
        tab_overview, tab_papers, tab_cards, tab_output, tab_raw = st.tabs(
            ["概览", "论文", "方法卡片", "Related Work", "原始 JSON"]
        )
        with tab_overview:
            render_overview(payload)
        with tab_papers:
            render_papers(payload)
        with tab_cards:
            render_method_cards(payload)
        with tab_output:
            render_related_work(payload)
        with tab_raw:
            st.json(payload, expanded=False)


if __name__ == "__main__":
    main()
