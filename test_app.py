import os
import tempfile

import pytest

# Configure a temporary test database BEFORE importing the app.
# This ensures tests never touch the real lambula.db or users.json.
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ["TEST_DATABASE_URI"] = f"sqlite:///{_tmp_db.name}"

from app import app, build_recommendation, db  # noqa: E402
from models import User  # noqa: E402


@pytest.fixture(autouse=True)
def setup_database():
    """Use a fresh in-memory database for each test."""
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["TESTING"] = True

    with app.app_context():
        db.create_all()
        yield
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client():
    return app.test_client()


def _register(client, email="test@example.com", full_name="Test Explorer"):
    return client.post(
        "/register",
        data={
            "full_name": full_name,
            "email": email,
            "password": "secret123",
            "confirm_password": "secret123",
            "terms": "on",
        },
        follow_redirects=True,
    )


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
    response = _register(client)
    assert response.status_code == 200
    assert b"Discover the best" in response.data


def test_home_page_does_not_offer_removed_animal(client):
    _register(client, email="test2@example.com")
    response = client.get("/home")
    assert response.status_code == 200
    assert b"Gorilla" not in response.data
    assert b"Unknown" not in response.data


def test_about_page(client):
    _register(client, email="test3@example.com")
    response = client.get("/about")
    assert response.status_code == 200


def test_prediction_result(client):
    _register(client, email="test4@example.com")
    response = client.post(
        "/predict",
        data={"animal": "Elephant", "temperature": "24", "rainfall": "150", "season": "Dry"},
    )
    assert response.status_code == 200


def test_prediction_uses_elephant_background_video(client):
    _register(client, email="test5@example.com")
    response = client.post(
        "/predict",
        data={"animal": "Elephant", "temperature": "24", "rainfall": "150", "season": "Dry"},
    )
    assert response.status_code == 200
    assert b"/static/videos/Elephant/" in response.data
    assert b"video/mp4" in response.data


def test_build_recommendation():
    result = build_recommendation("Elephant", 24, 150, "Dry")
    assert "recommended_park" in result
    assert "confidence" in result
    assert result["recommended_animal"] == "Elephant"


def test_user_stored_in_database(client):
    """Verify that registration creates a real database record."""
    _register(client, email="dbtest@example.com", full_name="DB Test")
    with app.app_context():
        user = User.query.filter_by(email="dbtest@example.com").first()
        assert user is not None
        assert user.full_name == "DB Test"
        assert user.provider == "local"


def test_duplicate_registration_rejected(client):
    """Registering the same email twice should fail."""
    _register(client, email="dup@example.com")
    client.get("/logout")
    response = client.post(
        "/register",
        data={
            "full_name": "Another",
            "email": "dup@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
            "terms": "on",
        },
    )
    assert b"already exists" in response.data


def test_prediction_history_saved(client):
    """A successful prediction should be saved to prediction_history."""
    _register(client, email="hist@example.com")
    client.post(
        "/predict",
        data={"animal": "Elephant", "temperature": "24", "rainfall": "150", "season": "Dry"},
    )
    with app.app_context():
        user = User.query.filter_by(email="hist@example.com").first()
        assert user is not None
        assert len(user.predictions) == 1
        assert user.predictions[0].animal == "Elephant"


def test_history_requires_login(client):
    response = client.get("/history")
    assert response.status_code == 302
    assert response.location.endswith("/login")


def test_history_page_shows_predictions(client):
    _register(client, email="hist2@example.com")
    client.post(
        "/predict",
        data={"animal": "Elephant", "temperature": "24", "rainfall": "150", "season": "Dry"},
    )
    response = client.get("/history")
    assert response.status_code == 200
    assert b"Elephant" in response.data
    assert b"Prediction History" in response.data


def test_history_page_empty_when_no_predictions(client):
    _register(client, email="hist3@example.com")
    response = client.get("/history")
    assert response.status_code == 200
    assert b"have not made any predictions yet" in response.data