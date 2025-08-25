from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import pickle
import pandas as pd

# -------------------------------
# Input Schema (matches your training INPUT_FIELDS)
# -------------------------------
class InputData(BaseModel):
    classification: str
    code: str
    implanted: str
    name_device: str
    name_manufacturer: str
    country: str

# -------------------------------
# Load Model & Encoders
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    print("✅ Model and encoders loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/encoders: {e}")
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
        # 1️⃣ Encode input row
        input_df = encode_input(data)

        # 2️⃣ Predict
        pred_class = model.predict(input_df)[0]
        pred_probs = model.predict_proba(input_df)[0]

        # 3️⃣ Decode class name if label-encoded
        target_encoder = label_encoders.get("action_classification")  # change if your target name differs
        if target_encoder:
            try:
                pred_class_name = target_encoder.inverse_transform([int(pred_class)])[0]
            except Exception:
                pred_class_name = str(pred_class)
        else:
            pred_class_name = str(pred_class)

        # 4️⃣ Return response
        return {
            "input": data.dict(),
            "prediction_class": pred_class_name,
            "prediction_id": int(pred_class),
            "class_probabilities": {
                str(cls): float(p) for cls, p in zip(model.classes_, pred_probs)
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
