"""健康检查接口测试。"""

from fastapi.testclient import TestClient

from app.main import app


def test_health_api() -> None:
    """验证健康检查接口可正常返回状态。"""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "app_name" in payload
