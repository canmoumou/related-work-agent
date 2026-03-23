# Related Work Workflow Agent

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#)

一个基于 `Python + FastAPI + LangGraph + Qwen(OpenAI-compatible) + Streamlit` 的 Related Work 生成工作流。  
输入研究方向，系统会自动拆解主题、检索 arXiv、下载筛选后的论文 PDF、抽取方法论内容、生成 method cards、写出 related work，并给出证据映射。

## Table of Contents

- [Features](#features)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Run](#run)
- [API](#api)
- [Ranking Strategy](#ranking-strategy)
- [Tests](#tests)
- [Limitations](#limitations)
- [Why Not Citation Count](#why-not-citation-count)
- [Roadmap](#roadmap)

## Features ✨

- 明确的 workflow，而不是自由发挥式 agent
- arXiv-only 文献来源
- 代表性排序使用“高相关 + 高代表性”代理策略，不伪造 citation count
- LLM 输出统一走结构化解析，失败时有 fallback
- 优先下载 PDF 并从全文方法论章节提取证据，而不是只依赖摘要
- 提供 `FastAPI`、`CLI`、`Streamlit` 三种使用方式

## Workflow 🔄

```text
User Topic
  -> IntentDecomposer
  -> QueryPlanner
  -> ArxivRetriever
  -> CandidateMerger
  -> PaperReranker
  -> PaperReader
  -> MethodExtractor
  -> ThemeClusterer
  -> RelatedWorkWriter
  -> EvidenceMapper / Verifier
```

## Project Structure 🗂️

```text
app/
  api/
  core/
  models/
  prompts/
  services/
  workflows/
  cli.py
  main.py
  streamlit_app.py
tests/
output/
.env.example
pyproject.toml
README.md
```

## Quick Start 🚀

推荐使用 `uv`：

```bash
uv venv paper
uv pip install -e .[dev]
```

Windows PowerShell:

```powershell
.\paper\Scripts\Activate.ps1
```

配置 API Key（二选一）：

**方式 1：使用 .env 文件（推荐开发环境）**

```bash
copy .env.example .env
```

然后编辑 `.env` 文件，填入你的 API key：

```env
QWEN_API_KEY=your-api-key-here
```

**方式 2：使用环境变量（推荐生产环境）**

Linux/macOS:
```bash
export QWEN_API_KEY=your-api-key-here
```

Windows PowerShell:
```powershell
$env:QWEN_API_KEY="your-api-key-here"
```

Windows CMD:
```cmd
set QWEN_API_KEY=your-api-key-here
```

> **注意**：环境变量优先级高于 `.env` 文件。如果两者都设置了，以环境变量为准。

可选配置项：
- `QWEN_API_KEY` - 阿里云百炼 API key（必填）
- `QWEN_BASE_URL` - API 基础地址（默认已配置）
- `QWEN_MODEL` - 模型名称，默认 qwen-plus

不配置 Qwen 也可以运行，但会走启发式 fallback，生成质量会下降。

## Run ▶️

启动 API：

```bash
uvicorn app.main:app --reload
```

启动 CLI：

```bash
python -m app.cli --topic "retrieval augmented generation"
```

启动 Streamlit：

```bash
streamlit run app/streamlit_app.py
```

默认情况下：

- API 通过 `FastAPI` 暴露工作流接口
- CLI 适合本地快速演示
- Streamlit 直接调用本地 workflow，适合交互式查看结果

## API 🌐

### `GET /health`

返回服务健康状态。

### `POST /workflow/run`

请求示例：

```json
{
  "topic": "retrieval augmented generation",
  "max_papers": 10
}
```

返回内容包括：

- `expanded_intents`
- `queries`
- `selected_papers`
- `method_cards`
- `clusters`
- `related_work`
- `evidence_map`
- `verification_report`

### `POST /workflow/debug`

在标准结果基础上额外返回：

- `candidate_papers`
- `paper_sections`
- `debug`

## Ranking Strategy 🧠

当前论文重排使用可替换的代理评分模块：

- `semantic_relevance_score`
- `coverage_score`
- `centrality_proxy_score`
- `diversity_adjustment`
- `metadata_quality_score`
- `final_rank_score`

注意：这里不把“代表性”写成“高被引”。

## Tests ✅

```bash
pytest
```

当前覆盖：

- intent decomposition 输出格式
- arXiv 结果解析
- reranker 去重与数量控制
- method card schema
- health API

## Limitations ⚠️

- 不做离线索引、长期缓存、数据库持久化
- 不实现 citation graph
- 不做高性能并发调度
- PDF 全文解析仍然是较难稳定完成的部分，当前采用“章节标题 + 关键词”的启发式抽取
- `Streamlit` 页面是原型前端，不是生产级 UI

## Why Not Citation Count 📉

arXiv API 不直接提供 citation count，也没有可靠官方 citation graph 接口。  
因此在 arXiv-only 约束下，本项目只能采用“高相关 + 高代表性”的代理策略，而不能伪造“高被引”。

## Roadmap 🛣️

- 接入 Crossref / Semantic Scholar / OpenAlex
- 增加 PDF 文本抽取与 section classifier
- 接入 embedding 相似度或学习排序
- 增加缓存和异步任务队列
- 增加人工审阅与编辑能力
