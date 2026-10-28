from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

sample_laptop = {
    "Company": "Apple",
    "TypeName": "Ultrabook",
    "Inches": 13.3,
    "ScreenResolution": "IPS Panel Retina Display 2560x1600",
    "Cpu": "Intel Core i5 2.3GHz",
    "Ram_GB": 8,
    "Memory": "128GB SSD",
    "Gpu": "Intel Iris Plus Graphics 640",
    "OpSys": "macOS",
    "Weight_kg": 1.37
}


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction():
    response = client.post("/predict", json=sample_laptop)

    assert response.status_code == 200

    result = response.json()

    assert result["currency"] == "EUR"
    assert isinstance(result["predicted_price_euros"], float)
    assert result["predicted_price_euros"] > 0


def test_invalid_input():
    laptop = sample_laptop.copy()
    laptop["Ram_GB"] = -8

    response = client.post("/predict", json=laptop)

    assert response.status_code == 422