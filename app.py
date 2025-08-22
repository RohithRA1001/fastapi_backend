from fastapi import FastAPI
import joblib

# Create FastAPI app
app = FastAPI()

# Full path to your dummy model
MODEL_PATH = r"c:\Users\rohit\Documents\Cognizant\backend\dummy_model.pkl"

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.post("/predict")
def predict(data: dict):
    if not model:
        return {"error": "Model not loaded"}
    
    try:
        features = data.get("features")
        prediction = model.predict([features])[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}
