"""Prompt 模板加载与渲染服务。"""

from pathlib import Path

from jinja2 import Template


class PromptService:
    """从磁盘读取 prompt 模板，并用 Jinja2 渲染变量。"""

    def __init__(self, prompt_dir: str | Path = "app/prompts") -> None:
        self.prompt_dir = Path(prompt_dir)

    def render(self, template_name: str, **kwargs: object) -> str:
        """渲染指定模板，供各个 LLM 节点使用。"""
        template_path = self.prompt_dir / template_name
        content = template_path.read_text(encoding="utf-8")
        return Template(content).render(**kwargs)
