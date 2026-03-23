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
        "研究方向（目前只支持英文查询）",
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
            if paper.get("pdf_url"):
                st.markdown(f"**PDF**: {paper['pdf_url']}")
            st.markdown("**Abstract**")
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


def render_method_like_sections(result: dict) -> None:
    """展示每篇论文抽取出的 method_like 内容。"""
    st.subheader("提取出的 Method Like")
    paper_sections = result.get("paper_sections", [])
    if not paper_sections:
        st.info("当前结果中没有可展示的 method_like 内容。")
        return

    for bundle in paper_sections:
        method_like = next((section for section in bundle.get("sections", []) if section.get("label") == "method_like"), None)
        if method_like is None:
            continue
        title = bundle.get("title", bundle.get("paper_id", "Untitled Paper"))
        with st.expander(title, expanded=False):
            st.markdown(f"**Paper ID**: `{bundle.get('paper_id', '')}`")
            st.markdown(f"**Source**: `{method_like.get('source', '')}`")
            st.write(method_like.get("content", ""))


def render_related_work(result: dict) -> None:
    """展示聚类结果、related work 与证据映射。"""
    st.subheader("Related Work")
    st.write(result["related_work"])

    paragraphs = result.get("related_work_paragraphs", [])
    if paragraphs:
        st.subheader("段落引用")
        for paragraph in paragraphs:
            paragraph_index = paragraph.get("paragraph_index", 0) + 1
            with st.expander(f"段落 {paragraph_index}", expanded=paragraph_index == 1):
                st.write(paragraph.get("paragraph_text", ""))
                citations = paragraph.get("citations", [])
                if not citations:
                    st.info("该段当前没有可展示的有效引用。")
                    continue
                for citation_index, citation in enumerate(citations, start=1):
                    st.markdown(
                        f"**[{citation_index}] {citation.get('title', 'Unknown Paper')}** "
                        f"(`{citation.get('paper_id', '')}`)"
                    )
                    st.markdown(f"**Section**: `{citation.get('section_label', '')}`")
                    st.markdown(f"**Rationale**: {citation.get('rationale', '')}")
                    st.code(citation.get("quote", ""), language=None)

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
    st.caption("基于 arXiv 检索、PDF 下载、方法论抽取与 Qwen 生成的 Related Work 前端")

    topic, max_papers, debug_mode = render_sidebar()
    workflow = get_workflow()

    if st.button("运行 Workflow", type="primary", width="stretch"):
        if not topic:
            st.warning("请先输入研究方向。")
            return

        with st.spinner("正在运行 workflow，请稍候..."):
            try:
                # 页面当前直接调用本地 workflow，而不是通过 HTTP API 转发。
                result = run_async(workflow.run(topic=topic, max_papers=max_papers, debug=True))
                payload = result.model_dump(mode="json")
                output_path = save_result(topic, payload)
            except Exception as e:
                st.error("无法连接到llm，请检查网络和环境配置")
                return

        st.success(f"运行完成，结果已保存到 `{output_path}`")
        tab_overview, tab_papers, tab_methods, tab_cards, tab_output = st.tabs(
            ["概览", "论文", "提取方法论", "方法卡片", "Related Work"]
        )
        with tab_overview:
            render_overview(payload)
        with tab_papers:
            render_papers(payload)
        with tab_methods:
            render_method_like_sections(payload)
        with tab_cards:
            render_method_cards(payload)
        with tab_output:
            render_related_work(payload)


if __name__ == "__main__":
    main()
