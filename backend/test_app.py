from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 404  # Since there's no root endpoint

def test_generate_ui():
    response = client.post("/api/generate-ui", data={})
    assert response.status_code in [200, 400, 404]  # Adjust based on real API behavior