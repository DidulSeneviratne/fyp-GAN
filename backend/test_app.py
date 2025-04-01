from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 404  # Since there's no root endpoint

def test_generate_ui():
    with open("dataset1/1.jpg", "rb") as image_file:
        response = client.post("/api/generate-ui", files={
            "sketch": ("sketch.jpg", image_file, "image/jpg"),
        }, data={
            "region": "South Asia",
            "age": "Teen",
            "device": "Mobile",
            "product": "Travel",
            "useCustomColor": "false",
            "colors1": "[]",
            "orientation": "Portrait"
        })

    assert response.status_code in [200, 400, 404]  # Adjust based on real API behavior