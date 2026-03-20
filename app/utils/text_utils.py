"""文本清洗、分词与相似度计算工具。"""

import math
import re
from collections import Counter


WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def normalize_whitespace(text: str) -> str:
    """压缩多余空白，便于统一比较和展示。"""
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    """做一个轻量分词，主要服务于启发式排序与匹配。"""
    return [match.group(0).lower() for match in WORD_RE.finditer(text or "")]


def jaccard_similarity(left: str, right: str) -> float:
    """使用集合重叠度估计两个短文本的相似性。"""
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def cosine_similarity(left: str, right: str) -> float:
    """使用词频向量余弦相似度估计文本接近程度。"""
    left_counter = Counter(tokenize(left))
    right_counter = Counter(tokenize(right))
    if not left_counter or not right_counter:
        return 0.0

    common = set(left_counter) & set(right_counter)
    dot = sum(left_counter[token] * right_counter[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left_counter.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counter.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def safe_slug(text: str) -> str:
    """把任意文本转成适合文件名或标识符的 slug。"""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return normalized or "result"
