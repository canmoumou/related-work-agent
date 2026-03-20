"""FastAPI 路由定义。"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import HealthResponse, WorkflowDebugResponse, WorkflowRunRequest, WorkflowRunResponse
from app.workflows.related_work_workflow import RelatedWorkWorkflow


router = APIRouter()
workflow = RelatedWorkWorkflow()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """返回服务健康状态。"""
    settings = get_settings()
    return HealthResponse(status="ok", app_name=settings.app_name)


@router.post("/workflow/run", response_model=WorkflowRunResponse)
async def run_workflow(request: WorkflowRunRequest) -> WorkflowRunResponse:
    """运行标准工作流，只返回最终结果。"""
    result = await workflow.run(topic=request.topic, max_papers=request.max_papers, debug=False)
    return WorkflowRunResponse.model_validate(result.model_dump())


@router.post("/workflow/debug", response_model=WorkflowDebugResponse)
async def run_workflow_debug(request: WorkflowRunRequest) -> WorkflowDebugResponse:
    """运行调试模式工作流，额外返回中间状态。"""
    result = await workflow.run(topic=request.topic, max_papers=request.max_papers, debug=True)
    return WorkflowDebugResponse.model_validate(result.model_dump())
