from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# -------------------------------
# Define Input Schema
# -------------------------------
class InputData(BaseModel):
    device: str
    classification: str
    manufacturer: str
    country: str
    implanted: str


# -------------------------------
# Load Model
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weakened_ensemble_model.pkl")

app = FastAPI()

# ✅ Enable CORS (important for frontend -> backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace "*" with ["https://your-frontend-domain.com"]
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
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Example encodings (adjust to match your preprocessing pipeline)
        classification_map = {"Class I": 1, "Class II": 2, "Class III": 3}
        country_map = {"USA": 1, "India": 2, "Germany": 3, "Japan": 4}
        implanted_map = {"yes": 1, "no": 0}

        # Convert input into numerical features
        features = [
            classification_map.get(data.classification, 0),
            country_map.get(data.country, 0),
            implanted_map.get(data.implanted, 0),
            len(data.manufacturer),  # simple encoding for manufacturer
            len(data.device)         # simple encoding for device name
        ]

        prediction = model.predict([features])

        return {
            "input": data.dict(),
            "features_used": features,
            "prediction": int(prediction[0])
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
