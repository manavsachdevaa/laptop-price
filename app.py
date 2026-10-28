from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Laptop Price Prediction API",
    description="Predict laptop prices using machine learning",
    version="1.0.0"
)

# Load the saved model and metadata
MODEL_PATH = Path(__file__).parent / "model.joblib"
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]


class LaptopInput(BaseModel):
    Company: str
    TypeName: str
    Inches: float = Field(gt=0, le=30)
    ScreenResolution: str
    Cpu: str
    Ram_GB: float = Field(gt=0, le=1024)
    Memory: str
    Gpu: str
    OpSys: str
    Weight_kg: float = Field(gt=0, le=20)


@app.get("/")
def home():
    return {
        "message": "Laptop Price Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_price(laptop: LaptopInput):
    input_data = pd.DataFrame(
        [laptop.model_dump()],
        columns=features
    )

    predicted_price = float(model.predict(input_data)[0])

    return {
        "predicted_price_euros": round(predicted_price, 2),
        "currency": "EUR"
    }