from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder="templates", static_folder="static")

base_dir = Path(__file__).resolve().parent
DATASET_PATH = base_dir / "Dataset" / "Tourism_Game_Park_Datasets.csv"

# Load and prepare the dataset
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

animal_encoder = LabelEncoder().fit(raw_df["Animal"])
park_encoder = LabelEncoder().fit(raw_df["Park"])
season_encoder = LabelEncoder().fit(raw_df["Season"])

features = pd.DataFrame(
    {
        "Animal": animal_encoder.transform(raw_df["Animal"]),
        "TempC": raw_df["TempC"],
        "RainfallMM": raw_df["RainfallMM"],
        "Season": season_encoder.transform(raw_df["Season"]),
    }
)
labels = park_encoder.transform(raw_df["Park"])

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(features, labels)

animals = sorted(
    [
        animal
        for animal in raw_df["Animal"].dropna().unique().tolist()
        if str(animal).strip().lower() not in {"chimpanzee", "chimpazee"}
    ]
)


def build_recommendation(animal, temperature, rainfall, season):
    animal_value = animal if animal in animal_encoder.classes_ else "Unknown"
    season_value = season if season in season_encoder.classes_ else "Dry"

    input_frame = pd.DataFrame(
        [
            {
                "Animal": animal_encoder.transform([animal_value])[0],
                "TempC": float(temperature),
                "RainfallMM": float(rainfall),
                "Season": season_encoder.transform([season_value])[0],
            }
        ]
    )

    predicted_index = int(model.predict(input_frame)[0])
    predicted_park = park_encoder.inverse_transform([predicted_index])[0]
    confidence = round(float(model.predict_proba(input_frame)[0].max() * 100), 1)

    matching_rows = raw_df[
        (raw_df["Animal"].str.lower() == animal_value.lower())
        & (raw_df["Season"].str.lower() == season_value.lower())
    ]
    if matching_rows.empty:
        matching_rows = raw_df[raw_df["Animal"].str.lower() == animal_value.lower()]
    if matching_rows.empty:
        matching_rows = raw_df

    park_rows = matching_rows[matching_rows["Park"] == predicted_park]
    if park_rows.empty:
        park_rows = matching_rows

    average_temp = round(float(park_rows["TempC"].mean()), 1)
    average_rainfall = round(float(park_rows["RainfallMM"].mean()), 1)
    dominant_season = park_rows["Season"].mode().iloc[0] if not park_rows["Season"].mode().empty else season_value
    sightings_count = int(len(park_rows))

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


@app.route("/")
def home():
    return render_template("index.html", animals=animals, selected_animal="Crested Crane")


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
        return render_template("result.html", **result)

    return render_template("index.html", animals=animals, selected_animal="Crested Crane")


if __name__ == "__main__":
    app.run(debug=True)