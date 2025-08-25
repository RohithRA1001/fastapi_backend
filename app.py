from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os

# -------------------------------
# Define Input Schemas
# -------------------------------
class InputData(BaseModel):  # Format A
    device: str
    classification: str
    manufacturer: str
    country: str
    implanted: str

class FeatureInput(BaseModel):  # Format B
    features: list[int]


# -------------------------------
# Load Model
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weakened_ensemble_model.pkl")

app = FastAPI()

# -------------------------------
# Enable CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for now allow all (change later for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def home():
    return {"message": "FastAPI backend is running!"}


@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        features = []

        # -------------------------------
        # Case 1: Raw features array
        # -------------------------------
        if "features" in data:
            features = data["features"]

        # -------------------------------
        # Case 2: Structured input
        # -------------------------------
        else:
            input_data = InputData(**data)

            classification_map = {"Class I": 1, "Class II": 2, "Class III": 3}
            country_map = {"USA": 1, "India": 2, "Germany": 3, "Japan": 4}
            implanted_map = {"yes": 1, "no": 0}

            features = [
                classification_map.get(input_data.classification, 0),
                country_map.get(input_data.country, 0),
                implanted_map.get(input_data.implanted, 0),
                len(input_data.manufacturer),
                len(input_data.device),
            ]

        # -------------------------------
        # Prediction
        # -------------------------------
        prediction = model.predict([features])

        return {
            "input": data,
            "features_used": features,
            "prediction": int(prediction[0]),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
