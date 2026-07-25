import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder="templates", static_folder="static")

base_dir = Path(__file__).resolve().parent
DATASET_PATH = base_dir / "Dataset" / "Tourism_Game_Park_Datasets.csv"
MODELS_DIR = base_dir / "Notebook" / "models"

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
        if str(animal).strip().lower() not in {"chimpanzee", "chimpazee", "unknown"}
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
        if str(animal).strip().lower() not in {"chimpanzee", "chimpazee", "unknown"}
    ])


def get_animal_image_path(animal):
    if not isinstance(animal, str):
        return "images/fallback.jpg"

    normalized = animal.strip().lower()
    image_mapping = {
        "buffalo": "animals/buffalo.jpg",
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
# 3. REVISIONARY RECOMMENDATION LOGIC
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
# 4. FLASK URL ROUTING STRATEGY
# ----------------------------------------------------
@app.route("/")
def home():
    # Graceful fallback selection if animals list failed to build
    default_animal = "Crested Crane" if "Crested Crane" in animals else (animals[0] if animals else "Unknown")
    return render_template("index.html", animals=animals, selected_animal=default_animal)


@app.route("/about")
def about():
    selected_animal = request.args.get("animal", "Elephant")
    return render_template("about.html", selected_animal=selected_animal)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        animal = request.form.get("animal", "Elephant")
        temperature = request.form.get("temperature", 24)
        rainfall = request.form.get("rainfall", 150)
        season = request.form.get("season", "Dry")

        result = build_recommendation(animal, temperature, rainfall, season)
        result["selected_animal"] = animal
        result["animal_image"] = get_animal_image_path(animal)
        return render_template("result.html", **result)

    default_animal = "Crested Crane" if "Crested Crane" in animals else (animals[0] if animals else "Unknown")
    return render_template("index.html", animals=animals, selected_animal=default_animal)


if __name__ == "__main__":
    app.run(debug=True)
