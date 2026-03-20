"""JSON 解析辅助函数。"""

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """优先直接解析 JSON，失败时再从文本中提取最外层对象。"""
    text = text.strip()
    if not text:
        raise ValueError("Empty text cannot be parsed as JSON.")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in text.")

    snippet = text[start : end + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")
    return parsed
