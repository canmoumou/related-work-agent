"""日志初始化工具。"""

import logging


def setup_logging(level: str = "INFO") -> None:
    """按统一格式初始化项目日志。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
