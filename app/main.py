"""FastAPI 应用入口。"""

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging


settings = get_settings()
# 应用启动时统一初始化日志配置。
setup_logging(settings.log_level)

app = FastAPI(title=settings.app_name, debug=settings.app_debug)
# 把 API 路由注册到主应用。
app.include_router(router)
