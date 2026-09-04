# tests/test_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.anyio
async def test_predict_success():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
            "features": [3.5, 1.2, 4.9]
        })
    assert resp.status_code == 200
    assert {"predictions": [7.0, 2.4, 9.8]} == resp.json()


@pytest.mark.anyio
async def test_predict_unprocessable_entity():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
            "feature1": 3.5,
            "feature2": 1.2,
            "feature3": 4.9
        })
    assert resp.status_code == 422


# --- Exercice 1 ---

@pytest.mark.anyio
async def test_predict_correct_values():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
            "features": [1.0, 2.0, 3.0]
        })
    assert resp.status_code == 200
    assert {"predictions": [2.0, 4.0, 6.0]} == resp.json()


@pytest.mark.anyio
async def test_predict_incorrect_values():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
            "features": [1.0, 2.0, 3.0]
        })
    assert resp.status_code == 200
    assert {"predictions": [3.0, 5.0, 7.0]} != resp.json()


@pytest.mark.anyio
async def test_predict_missing_features_field():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
            "wrong_field": [3.5, 1.2, 4.9]
        })
    assert resp.status_code == 422 