from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import pickle
import pandas as pd

# -------------------------------
# Define Input Schema (must match your training INPUT_FIELDS)
# -------------------------------
class InputData(BaseModel):
    classification: str
    code: str
    implanted: str
    name_device: str
    name_manufacturer: str


# -------------------------------
# Load Model & Encoders
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

app = FastAPI()

# Enable CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for now allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    print("Model and encoders loaded successfully.")
except Exception as e:
    print(f"Error loading model/encoders: {e}")
    model, label_encoders = None, None


# -------------------------------
# Helper: Encode Input Row
# -------------------------------
def encode_input(data: InputData):
    row = {}
    for col, val in data.dict().items():
        if col in label_encoders:
            le = label_encoders[col]
            if val in le.classes_:
                row[col] = int(le.transform([val])[0])
            else:
                # unseen category → fallback to most common
                row[col] = int(pd.Series(le.transform(le.classes_)).mode()[0])
        else:
            row[col] = val
    return pd.DataFrame([row])


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
        # Encode input
        input_df = encode_input(data)

        # Predict
        pred_class = model.predict(input_df)[0]
        pred_probs = model.predict_proba(input_df)[0]

        # Decode prediction if target was label-encoded
        target_name = pred_class
        target_encoder = label_encoders.get("action_classification")  # adjust if target name different
        if target_encoder:
            try:
                target_name = target_encoder.inverse_transform([int(pred_class)])[0]
            except Exception:
                pass

        return {
            "input": data.dict(),
            "prediction_class": str(target_name),
            "prediction_id": int(pred_class),
            "class_probabilities": {
                str(cls): float(p) for cls, p in zip(model.classes_, pred_probs)
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
