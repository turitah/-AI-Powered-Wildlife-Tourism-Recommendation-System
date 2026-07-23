import pytest
from app import app, build_recommendation


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Plan your visit" in response.data


def test_home_page_does_not_offer_removed_animal(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Chimpanzee" not in response.data
    assert b"chimpazee" not in response.data.lower()


def test_about_page(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert b"About the AI-powered wildlife recommendation system" in response.data


def test_prediction_result(client):
    response = client.post(
        "/predict",
        data={"animal": "Elephant", "temperature": "24", "rainfall": "150", "season": "Dry"},
    )
    assert response.status_code == 200
    assert b"Recommendation ready" in response.data


def test_build_recommendation():
    result = build_recommendation("Elephant", 24, 150, "Dry")
    assert "recommended_park" in result
    assert "confidence" in result
    assert result["recommended_animal"] == "Elephant"
