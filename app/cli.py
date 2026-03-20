"""命令行演示入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.utils.text_utils import safe_slug
from app.workflows.related_work_workflow import RelatedWorkWorkflow


async def run_cli(topic: str, max_papers: int) -> None:
    """运行工作流并把核心结果打印到终端。"""
    settings = get_settings()
    setup_logging(settings.log_level)
    workflow = RelatedWorkWorkflow()
    result = await workflow.run(topic=topic, max_papers=max_papers, debug=True)

    print("=== Expanded Intents ===")
    for item in result.expanded_intents.subtopics:
        print(f"- {item}")

    print("\n=== Top Papers ===")
    for index, paper in enumerate(result.selected_papers, start=1):
        print(f"{index}. {paper.title}")

    print("\n=== Related Work ===")
    print(result.related_work)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{safe_slug(topic)}_{timestamp}.json"
    output_path = settings.output_dir / output_name
    output_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON output saved to {output_path}")


def main() -> None:
    """解析命令行参数并启动 CLI。"""
    parser = argparse.ArgumentParser(description="Run the related work workflow agent.")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--max-papers", type=int, default=10, help="Maximum number of selected papers")
    args = parser.parse_args()
    asyncio.run(run_cli(topic=args.topic, max_papers=args.max_papers))


if __name__ == "__main__":
    main()
