import os
import sys
import json
from pathlib import Path
from functools import wraps

import joblib
import pandas as pd
from authlib.integrations.flask_client import OAuth
from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from models import PredictionHistory, User, db

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("SECRET_KEY", "lambula-wildlife-tourism-dev-key")

# ----------------------------------------------------
# DATABASE CONFIGURATION (SQLAlchemy + SQLite)
# ----------------------------------------------------
base_dir = Path(__file__).resolve().parent
DB_PATH = base_dir / "lambula.db"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH.as_posix()}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# Legacy JSON path — kept only for one-time migration of existing accounts.
USERS_PATH = base_dir / "Data" / "users.json"


def _migrate_json_users_to_db():
    """Copy any existing users from Data/users.json into the SQLite database.

    This runs once at startup. Users that already exist in the database are
    skipped, so it is safe to call repeatedly.
    """
    if not USERS_PATH.exists():
        return

    try:
        with open(USERS_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        print(f"WARNING: Could not read legacy users.json for migration. Details: {e}", file=sys.stderr)
        return

    if not isinstance(data, dict):
        return

    migrated = 0
    for email, info in data.items():
        if not isinstance(info, dict):
            continue
        # Skip if a DB user with this email already exists.
        if User.query.filter_by(email=email).first() is not None:
            continue

        user = User(
            email=email,
            full_name=info.get("full_name", ""),
            password_hash=info.get("password_hash", generate_password_hash(os.urandom(16).hex())),
            provider=info.get("provider", "local"),
        )
        db.session.add(user)
        migrated += 1

    if migrated:
        db.session.commit()
        print(f"SUCCESS: Migrated {migrated} user(s) from users.json into the database.")


def _init_database():
    with app.app_context():
        print(f"INIT DB: Using database at {DB_PATH.resolve()}", file=sys.stderr)
        if DB_PATH.exists():
            print("INIT DB: Database file exists, verifying tables...", file=sys.stderr)
            try:
                db.create_all()
                print("INIT DB: Tables verified successfully.", file=sys.stderr)
            except Exception as exc:
                print(f"WARNING: Database verification failed ({exc}). Data may be preserved.", file=sys.stderr)
                try:
                    db.create_all()
                except Exception as exc2:
                    print(f"WARNING: Could not create tables. Details: {exc2}", file=sys.stderr)
        else:
            print("INIT DB: Database file does not exist, creating...", file=sys.stderr)
            db.create_all()
            print("INIT DB: Database created successfully.", file=sys.stderr)
        _migrate_json_users_to_db()
        try:
            user_count = User.query.count()
            print(f"INIT DB: Total users in database: {user_count}", file=sys.stderr)
        except Exception as exc:
            print(f"INIT DB: Could not count users. Details: {exc}", file=sys.stderr)


_init_database()


# ----------------------------------------------------
# OAUTH CONFIGURATION
# ----------------------------------------------------
oauth = OAuth(app)
google_oauth = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

DATASET_PATH = base_dir / "Dataset" / "Tourism_Game_Park_Datasets.csv"
MODELS_DIR = base_dir / "Notebook" / "models"
WILDLIFE_INFO_PATH = base_dir / "Data" / "wildlife_info.json.txt"
PARK_INFO_PATH = base_dir / "Data" / "parks_info.json.txt"


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_email"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def get_available_animals():
    excluded = {"gorilla", "unknown"}
    available = sorted([
        animal for animal in animals
        if str(animal).strip().lower() not in excluded
    ])
    if not any(str(animal).strip().lower() == "chimpanzee" for animal in available):
        available.append("Chimpanzee")
        available.sort()
    return available


def google_oauth_configured():
    return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))


def login_google_user(user_info):
    """Create or look up a Google user in the database and set the session."""
    email = user_info.get("email", "").strip().lower()
    if not email:
        return False

    full_name = user_info.get("name") or email.split("@")[0]
    user = User.query.filter_by(email=email).first()

    if user is None:
        user = User(
            email=email,
            full_name=full_name,
            password_hash=generate_password_hash(os.urandom(16).hex()),
            provider="google",
        )
        db.session.add(user)
        db.session.commit()
    elif not user.full_name:
        user.full_name = full_name
        db.session.commit()

    session["user_email"] = email
    session["user_name"] = user.full_name or full_name
    return True


# ----------------------------------------------------
# 1. LOAD DATASET FOR BACKUP & SIGHTING DETAILS
# ----------------------------------------------------
try:
    raw_df = pd.read_csv(DATASET_PATH)
    raw_df["TempC"] = pd.to_numeric(raw_df["TempC"], errors="coerce")
    raw_df["RainfallMM"] = pd.to_numeric(raw_df["RainfallMM"], errors="coerce")
    raw_df["Visitors"] = pd.to_numeric(raw_df["Visitors"], errors="coerce")
    raw_df["Animal"] = raw_df["Animal"].fillna("Unknown").astype(str).str.strip()
    raw_df["Park"] = raw_df["Park"].fillna("Unknown").astype(str).str.strip()
    raw_df["Season"] = raw_df["Season"].fillna("Dry").astype(str).str.strip()

    raw_df["TempC"] = raw_df["TempC"].fillna(raw_df["TempC"].median())
    raw_df["RainfallMM"] = raw_df["RainfallMM"].fillna(raw_df["RainfallMM"].median())
    raw_df["Visitors"] = raw_df["Visitors"].fillna(raw_df["Visitors"].median())
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load dataset from {DATASET_PATH}. Error: {e}", file=sys.stderr)
    raw_df = pd.DataFrame(columns=["Animal", "Park", "Season", "TempC", "RainfallMM", "Visitors"])

# ----------------------------------------------------
# 2. SYSTEM OVERRIDES & SAFE ML MACHINE LOADING
# ----------------------------------------------------
# Initialize globally accessible variables as None
feature_columns = ["Animal", "TempC", "RainfallMM", "Season"]
park_encoder = None
animal_encoder = None
season_encoder = None
model = None
animals = []

try:
    feature_columns = list(joblib.load(MODELS_DIR / "feature_columns.pkl"))
    label_encoders = joblib.load(MODELS_DIR / "label_encoder.pkl")

    park_encoder = label_encoders["Park"]
    animal_encoder = label_encoders["Animal"]
    season_encoder = label_encoders["Season"]
    model = joblib.load(MODELS_DIR / "wildlife_detection_random_forest.pkl")

    print("SUCCESS: All pre-trained model files and encoders loaded successfully.")

    animals = sorted([
        animal for animal in animal_encoder.classes_
        if str(animal).strip().lower() not in {"gorilla", "unknown"}
    ])
except FileNotFoundError as fnf_err:
    print(f"CRITICAL ERROR: Missing core machine learning asset. Details: {fnf_err}", file=sys.stderr)
except Exception as e:
    print(f"CRITICAL ERROR: Failed to parse or load ML pickle architectures. Details: {e}", file=sys.stderr)

# Fallback: always populate the list from the dataset if the model encoders are unavailable
if not animals:
    animals = sorted([
        animal
        for animal in raw_df["Animal"].dropna().astype(str).str.strip().unique().tolist()
        if str(animal).strip().lower() not in {"gorilla", "unknown"}
    ])


def get_animal_image_path(animal):
    if not isinstance(animal, str):
        return "images/fallback.jpg"

    normalized = animal.strip().lower()
    image_mapping = {
        "buffalo": "animals/buffalo.jpg",
        "chimpanzee": "animals/Chimpanzee/image_4.jpg",
        "elephant": "animals/elephant.jpg",
        "gorilla": "animals/Glorilla.jpg",
        "hippo": "animals/Hippo.jpg",
        "leopard": "animals/leopard.jpg",
        "lion": "animals/lion.jpg",
    }

    if normalized in image_mapping:
        return image_mapping[normalized]

    if "buffalo" in normalized:
        return image_mapping["buffalo"]
    if "elephant" in normalized:
        return image_mapping["elephant"]
    if "gorilla" in normalized:
        return image_mapping["gorilla"]
    if "hippo" in normalized:
        return image_mapping["hippo"]
    if "leopard" in normalized:
        return image_mapping["leopard"]
    if "lion" in normalized:
        return image_mapping["lion"]

    return "images/fallback.jpg"


# ----------------------------------------------------
# 3. METADATA LOADING
# ----------------------------------------------------

wildlife_info = {}
park_info = {}

try:
    with open(WILDLIFE_INFO_PATH, "r", encoding="utf-8") as file:
        wildlife_info = json.load(file)
except Exception as e:
    print(f"WARNING: Failed to load wildlife metadata. Details: {e}", file=sys.stderr)

try:
    with open(PARK_INFO_PATH, "r", encoding="utf-8") as file:
        park_info = json.load(file)
except Exception as e:
    print(f"WARNING: Failed to load park metadata. Details: {e}", file=sys.stderr)


def asset_path(value, default="images/fallback.jpg"):
    if not value or not isinstance(value, str):
        return default
    normalized = value.strip().replace("\\", "/").lstrip("/")
    if normalized.lower().startswith("static/"):
        normalized = normalized[7:]
    return normalized or default


def get_wildlife_primary_image(animal):
    animal_data = wildlife_info.get(animal, {})
    image_list = animal_data.get("images", [])
    if image_list and isinstance(image_list, list):
        return asset_path(image_list[0])
    return get_animal_image_path(animal)


def get_wildlife_gallery(animal, limit=6):
    animal_data = wildlife_info.get(animal, {})
    image_list = animal_data.get("images", [])
    if not isinstance(image_list, list):
        return []
    gallery = [asset_path(image) for image in image_list if isinstance(image, str)]
    return gallery[:limit]


def get_wildlife_video(animal):
    animal_data = wildlife_info.get(animal, {})
    video_list = animal_data.get("videos", [])
    if video_list and isinstance(video_list, list):
        return asset_path(video_list[0])

    normalized = animal.strip().lower()
    if "lion" in normalized:
        return "videos/lion.mp4"
    if "buffalo" in normalized:
        return "videos/buffalo.mp4"
    if "elephant" in normalized:
        return "videos/elephant.mp4"
    if "hippo" in normalized:
        return "videos/hippo.mp4"
    if "leopard" in normalized:
        return "videos/leopard.mp4"
    if "gorilla" in normalized or "chimp" in normalized:
        return "videos/gorilla.mp4"
    return "videos/crestedCrane.mp4"


def get_park_image(park_data, park_name=None):
    if not isinstance(park_data, dict):
        return "images/fallback.jpg"

    candidate = asset_path(park_data.get("park_image", ""))
    candidate_path = base_dir / "static" / candidate
    if candidate and candidate_path.exists():
        return candidate

    if park_name and isinstance(park_name, str):
        park_folder = base_dir / "static" / "Parks" / park_name
        if park_folder.exists() and park_folder.is_dir():
            for img_file in sorted(park_folder.iterdir()):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    return str(Path("Parks") / park_name / img_file.name).replace("\\", "/")

    return "images/fallback.jpg"


def get_full_park_name(short_name):
    if not isinstance(short_name, str):
        return short_name
    normalized = short_name.strip().lower()
    for full_name in park_info:
        if normalized == full_name.lower() or normalized in full_name.lower():
            return full_name
    for full_name in park_info:
        if full_name.lower().startswith(normalized):
            return full_name
    return short_name


def get_park_features(park_data, limit=5):
    if not isinstance(park_data, dict):
        return []
    features = park_data.get("activities", []) or park_data.get("main_attractions", [])
    if not isinstance(features, list):
        return []
    return features[:limit]


def get_related_animals(park_data, recommended_animal):
    if not isinstance(park_data, dict):
        return []
    animals = park_data.get("main_wildlife", [])
    if not isinstance(animals, list):
        return []
    return [animal for animal in animals if animal != recommended_animal][:4]

# ----------------------------------------------------
# 4. REVISIONARY RECOMMENDATION LOGIC
# ----------------------------------------------------
def build_recommendation(animal, temperature, rainfall, season):
    # Safety Check: If the system files didn't boot, yield an elegant fallback error dict
    if not all([model, park_encoder, animal_encoder, season_encoder]):
        return {
            "recommended_animal": animal,
            "recommended_park": "System Maintenance Mode",
            "confidence": 0.0,
            "reason": "The backend recommendation system is temporarily offline due to model maintenance.",
            "average_temp": 0.0,
            "average_rainfall": 0.0,
            "dominant_season": "N/A",
            "sightings_count": 0,
        }

    # Verify input categorical items against locked model classes safely
    animal_value = animal if animal in animal_encoder.classes_ else "Unknown"
    season_value = season if season in season_encoder.classes_ else "Dry"

    feature_values = {
        "Animal": animal_encoder.transform([animal_value])[0],
        "TempC": float(temperature),
        "RainfallMM": float(rainfall),
        "Season": season_encoder.transform([season_value])[0],
    }
    input_frame = pd.DataFrame([feature_values], columns=feature_columns)

    predicted_index = int(model.predict(input_frame)[0])
    predicted_park = park_encoder.inverse_transform([predicted_index])[0]
    confidence = round(float(model.predict_proba(input_frame)[0].max() * 100), 1)

    # Process historical data rows cleanly using local dataframe copies if available
    matching_rows = raw_df[
        (raw_df["Animal"].str.lower() == animal_value.lower())
        & (raw_df["Season"].str.lower() == season_value.lower())
    ] if not raw_df.empty else pd.DataFrame()

    if matching_rows.empty and not raw_df.empty:
        matching_rows = raw_df[raw_df["Animal"].str.lower() == animal_value.lower()]
    if matching_rows.empty:
        matching_rows = raw_df

    if not matching_rows.empty:
        park_rows = matching_rows[matching_rows["Park"] == predicted_park]
        if park_rows.empty:
            park_rows = matching_rows

        average_temp = round(float(park_rows["TempC"].mean()), 1)
        average_rainfall = round(float(park_rows["RainfallMM"].mean()), 1)
        dominant_season = park_rows["Season"].mode().iloc[0] if not park_rows["Season"].mode().empty else season_value
        sightings_count = int(len(park_rows))
    else:
        average_temp, average_rainfall, dominant_season, sightings_count = 0.0, 0.0, season_value, 0

    reason = (
        f"Historical records suggest that {animal_value} is most commonly observed in {predicted_park} "
        f"when conditions are close to {temperature}°C and {rainfall}mm of rainfall during the {season_value} season."
    )

    return {
        "recommended_animal": animal_value,
        "recommended_park": predicted_park,
        "confidence": confidence,
        "reason": reason,
        "average_temp": average_temp,
        "average_rainfall": average_rainfall,
        "dominant_season": dominant_season,
        "sightings_count": sightings_count,
    }

# ----------------------------------------------------
# 5. FLASK URL ROUTING STRATEGY
# ----------------------------------------------------
@app.route("/")
def root():
    if session.get("user_email"):
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_email"):
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        print(f"LOGIN ATTEMPT: email={email}, user_found={user is not None}", file=sys.stderr)
        if user is not None:
            print(f"LOGIN DEBUG: stored_hash={user.password_hash[:20]}...", file=sys.stderr)

        if not user or not check_password_hash(user.password_hash, password):
            print(f"LOGIN FAILED: email={email}", file=sys.stderr)
            return render_template("login.html", error="Invalid email or password.")

        session["user_email"] = email
        session["user_name"] = user.full_name or email
        session.permanent = bool(request.form.get("remember"))
        print(f"LOGIN SUCCESS: email={email}", file=sys.stderr)
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_email"):
        return redirect(url_for("home"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not full_name or not email or not password:
            return render_template("register.html", error="Please fill in all required fields.")
        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match.")
        if len(password) < 6:
            return render_template("register.html", error="Password must be at least 6 characters.")

        if User.query.filter_by(email=email).first() is not None:
            return render_template("register.html", error="An account with this email already exists.")

        user = User(
            email=email,
            full_name=full_name,
            password_hash=generate_password_hash(password),
            provider="local",
        )
        db.session.add(user)
        db.session.commit()
        print(f"REGISTER SUCCESS: email={email}, user_id={user.id}", file=sys.stderr)

        session["user_email"] = email
        session["user_name"] = full_name
        return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/auth/google")
def google_login():
    if not google_oauth_configured():
        return render_template(
            "login.html",
            error="Google sign-in is not configured yet. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )
    redirect_uri = url_for("google_callback", _external=True)
    return google_oauth.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def google_callback():
    if not google_oauth_configured():
        return redirect(url_for("login"))

    try:
        token = google_oauth.authorize_access_token()
        user_info = token.get("userinfo")
        if not user_info:
            user_info = google_oauth.get("userinfo").json()
        if not login_google_user(user_info):
            return render_template("login.html", error="Google sign-in did not return a valid email.")
    except Exception as exc:
        print(f"WARNING: Google OAuth callback failed. Details: {exc}", file=sys.stderr)
        return render_template("login.html", error="Google sign-in failed. Please try again.")

    return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.pop("user_email", None)
    session.pop("user_name", None)
    session.permanent = False
    session.modified = True
    return redirect(url_for("login"))


@app.route("/history")
@login_required
def history():
    user = User.query.filter_by(email=session.get("user_email")).first()
    predictions = []
    if user is not None:
        predictions = (
            PredictionHistory.query.filter_by(user_id=user.id)
            .order_by(PredictionHistory.created_at.desc())
            .all()
        )
    return render_template(
        "history.html",
        predictions=predictions,
        user_name=session.get("user_name"),
    )


@app.route("/home")
@login_required
def home():
    available_animals = get_available_animals()
    default_animal = "Elephant" if "Elephant" in available_animals else (available_animals[0] if available_animals else "Unknown")
    return render_template(
        "index.html",
        animals=available_animals,
        selected_animal=default_animal,
        user_name=session.get("user_name"),
    )


@app.route("/about")
@login_required
def about():
    selected_animal = request.args.get("animal", "Elephant")
    return render_template("about.html", selected_animal=selected_animal, user_name=session.get("user_name"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        animal = request.form.get("animal", "Elephant")
        temperature = request.form.get("temperature", 24)
        rainfall = request.form.get("rainfall", 150)
        season = request.form.get("season", "Dry")

        result = build_recommendation(animal, temperature, rainfall, season)
        result["selected_animal"] = animal
        result["animal_image"] = get_animal_image_path(animal)

        wildlife_data = dict(wildlife_info.get(animal, {}))
        wildlife_data["image_path"] = get_wildlife_primary_image(animal)
        wildlife_data["gallery"] = get_wildlife_gallery(animal)
        wildlife_data["video_path"] = get_wildlife_video(animal)
        result["wildlife_data"] = wildlife_data

        full_park_name = get_full_park_name(result["recommended_park"])
        park_data = dict(park_info.get(full_park_name, {}))
        park_data["name"] = full_park_name
        park_data["park_image_path"] = get_park_image(park_data, full_park_name)
        park_data["highlights"] = get_park_features(park_data)
        result["recommended_park"] = full_park_name
        result["park_data"] = park_data
        result["related_animals"] = get_related_animals(park_data, result["recommended_animal"])
        result["user_name"] = session.get("user_name")

        try:
            user = User.query.filter_by(email=session.get("user_email")).first()
            if user is not None:
                history_entry = PredictionHistory(
                    user_id=user.id,
                    animal=animal,
                    temperature=float(temperature),
                    rainfall=float(rainfall),
                    season=season,
                    recommended_park=full_park_name,
                    confidence=result.get("confidence"),
                )
                db.session.add(history_entry)
                db.session.commit()
        except Exception as exc:
            print(f"WARNING: Could not save prediction history. Details: {exc}", file=sys.stderr)
            db.session.rollback()

        return render_template("result.html", **result)

    default_animal = "Elephant" if "Elephant" in get_available_animals() else (get_available_animals()[0] if get_available_animals() else "Unknown")
    return render_template(
        "index.html",
        animals=get_available_animals(),
        selected_animal=default_animal,
        user_name=session.get("user_name"),
    )


# ----------------------------------------------------
# 5. PLAN SAFARI / BOOKING (BUSINESS MODULE)
# ----------------------------------------------------
SAFARI_PACKAGES = [
    {
        "key": "trail_tracker",
        "name": "Trail Tracker",
        "days": 2,
        "price_per_person": 240,
        "description": "Shared guide and a budget lodge — a short, focused trip built around your prediction window.",
    },
    {
        "key": "canopy_explorer",
        "name": "Canopy Explorer",
        "days": 3,
        "price_per_person": 460,
        "description": "Private guide and a mid-tier eco-lodge close to the park entrance.",
    },
    {
        "key": "rangers_reserve",
        "name": "Ranger's Reserve",
        "days": 5,
        "price_per_person": 980,
        "description": "Multi-park route with a premium camp and dedicated ranger support.",
    },
]

GUIDE_FEE = 40
PERMIT_FEE = 70
SERVICE_FEE = 18


@app.route("/plan-safari")
@login_required
def plan_safari():
    animal = request.args.get("animal", "Elephant")
    park = request.args.get("park", "Kibale National Park")
    confidence = request.args.get("confidence", "")
    try:
        travelers = max(1, int(request.args.get("travelers", 2)))
    except (TypeError, ValueError):
        travelers = 2

    packages = []
    for pkg in SAFARI_PACKAGES:
        subtotal = pkg["price_per_person"] * travelers
        total = subtotal + GUIDE_FEE + PERMIT_FEE + SERVICE_FEE
        packages.append({**pkg, "subtotal": subtotal, "total": total})

    return render_template(
        "plan_safari.html",
        selected_animal=animal,
        animal_image=get_wildlife_primary_image(animal),
        park=park,
        confidence=confidence,
        travelers=travelers,
        packages=packages,
        guide_fee=GUIDE_FEE,
        permit_fee=PERMIT_FEE,
        service_fee=SERVICE_FEE,
        user_name=session.get("user_name"),
    )


@app.route("/book-safari", methods=["POST"])
@login_required
def book_safari():
    animal = request.form.get("animal", "Elephant")
    park = request.form.get("park", "Kibale National Park")
    package_name = request.form.get("package_name", "Trail Tracker")
    total = request.form.get("total", "0")
    travelers = request.form.get("travelers", "2")

    return render_template(
        "booking_confirmation.html",
        selected_animal=animal,
        animal_image=get_wildlife_primary_image(animal),
        park=park,
        package_name=package_name,
        total=total,
        travelers=travelers,
        user_name=session.get("user_name"),
    )


if __name__ == "__main__":
    app.run(debug=True)