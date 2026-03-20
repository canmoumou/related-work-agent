"""项目配置定义与加载。"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中管理运行时配置。

    配置优先级（从高到低）：
    1. 环境变量（如 export QWEN_API_KEY=xxx）
    2. .env 文件中的变量
    3. 默认值
    """

    app_name: str = "Related Work Workflow Agent"
    app_env: str = "development"
    app_debug: bool = False
    log_level: str = "INFO"

    # Qwen API 配置 - 从环境变量 QWEN_API_KEY 读取
    qwen_api_key: str = Field(default="", alias="QWEN_API_KEY")
    qwen_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", alias="QWEN_BASE_URL")
    qwen_model: str = Field(default="qwen-plus", alias="QWEN_MODEL")

    arxiv_base_url: str = "https://export.arxiv.org/api/query"
    arxiv_timeout_seconds: float = 30.0
    arxiv_min_interval_seconds: float = 3.0
    arxiv_retry_count: int = 3
    arxiv_backoff_seconds: float = 2.0
    arxiv_user_agent: str = "related-work-agent/0.1 (contact: local-dev)"
    request_timeout_seconds: float = 60.0
    max_llm_retries: int = 2
    output_dir: Path = Path("output")

    model_config = SettingsConfigDict(
        # 优先从环境变量读取，其次从 .env 文件读取
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """缓存配置对象，避免在运行中重复读取环境变量。"""
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings
