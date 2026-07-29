import pytest
from app import app, build_recommendation


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_auth_page(client):
    response = client.get("/login")
    assert response.status_code == 200
    assert b"Welcome Back!" in response.data


def test_register_page(client):
    response = client.get("/register")
    assert response.status_code == 200
    assert b"Create Account" in response.data


def test_home_requires_login(client):
    response = client.get("/home")
    assert response.status_code == 302
    assert response.location.endswith("/login")


def test_register_and_home_page(client):
    response = client.post(
        "/register",
        data={
            "full_name": "Test Explorer",
            "email": "test@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
            "terms": "on",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Discover the best" in response.data
    assert b"Number of Visitors" not in response.data


def test_home_page_does_not_offer_removed_animal(client):
    client.post(
        "/register",
        data={
            "full_name": "Test Explorer",
            "email": "test2@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
            "terms": "on",
        },
    )
    response = client.get("/home")
    assert response.status_code == 200
    assert b"Chimpanzee" not in response.data
    assert b"chimpazee" not in response.data.lower()


def test_about_page(client):
    client.post(
        "/register",
        data={
            "full_name": "Test Explorer",
            "email": "test3@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
            "terms": "on",
        },
    )
    response = client.get("/about")
    assert response.status_code == 200
    assert b"About the AI-powered wildlife recommendation system" in response.data


def test_prediction_result(client):
    client.post(
        "/register",
        data={
            "full_name": "Test Explorer",
            "email": "test4@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
            "terms": "on",
        },
    )
    response = client.post(
        "/predict",
        data={"animal": "Elephant", "temperature": "24", "rainfall": "150", "season": "Dry"},
    )
    assert response.status_code == 200
    assert b"Recommendation ready" in response.data


def test_prediction_uses_elephant_background_video(client):
    client.post(
        "/register",
        data={
            "full_name": "Test Explorer",
            "email": "test5@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
            "terms": "on",
        },
    )
    response = client.post(
        "/predict",
        data={"animal": "Elephant", "temperature": "24", "rainfall": "150", "season": "Dry"},
    )
    assert response.status_code == 200
    assert b"videos/elephant.mp4" in response.data


def test_build_recommendation():
    result = build_recommendation("Elephant", 24, 150, "Dry")
    assert "recommended_park" in result
    assert "confidence" in result
    assert result["recommended_animal"] == "Elephant"
