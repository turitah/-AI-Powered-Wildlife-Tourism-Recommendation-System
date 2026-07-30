"""Database models for the LAMBULA Wildlife Tourism Recommendation System.

This module defines the SQLAlchemy database models used by the Flask
application to persist user accounts and prediction history.

Models
------
User
    Represents a registered user (local email/password or Google OAuth).
PredictionHistory
    Stores a user's past wildlife recommendation queries and results.
"""

import datetime

from flask_sqlalchemy import SQLAlchemy

# A single SQLAlchemy instance shared across the application.
db = SQLAlchemy()


class User(db.Model):
    """A registered user of the LAMBULA platform."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(255), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    provider = db.Column(db.String(50), nullable=False, default="local")
    created_at = db.Column(db.DateTime, default=datetime.datetime.now(datetime.timezone.utc))

    # Relationship: a user can have many prediction-history entries.
    predictions = db.relationship(
        "PredictionHistory",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<User {self.email}>"


class PredictionHistory(db.Model):
    """A single recommendation query made by a user."""

    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=False, index=True
    )
    animal = db.Column(db.String(100), nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    season = db.Column(db.String(50), nullable=False)
    recommended_park = db.Column(db.String(255), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now(datetime.timezone.utc))

    def __repr__(self):
        return f"<PredictionHistory {self.animal} -> {self.recommended_park}>"