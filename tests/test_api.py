import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.anyio
async def test_predict_success():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/predict",
            json={"features": [3.5, 1.2, 4.9]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [7.0, 2.4, 9.8]
    }


@pytest.mark.anyio
async def test_predict_unprocessable_entity():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/predict",
            json={
                "feature1": 3.5,
                "feature2": 1.2,
                "feature3": 4.9,
            },
        )

    assert response.status_code == 422
