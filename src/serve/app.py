"""FastAPI model serving endpoint."""

import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Steel Fault Detection API",
    description="Predict steel plate fault types from inspection measurements.",
    version="1.0.0",
)

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

FEATURE_NAMES = scaler.feature_names_in_.tolist()


class PredictionRequest(BaseModel):
    features: list[float]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [
                        42,
                        50,
                        270900,
                        270944,
                        267,
                        17,
                        44,
                        24220,
                        76,
                        108,
                        1687,
                        0,
                        1,
                        100,
                        0.0839,
                        0.6015,
                        0.7781,
                        0.0,
                        0.2893,
                        1.0,
                        0.0,
                        2.4265,
                        -0.3665,
                        1.6439,
                        -0.471,
                        -0.2035,
                        0.3862,
                    ]
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "healthy", "model": "steel-fault-classifier"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(FEATURE_NAMES)} features, got {len(request.features)}",
        )

    input_df = pd.DataFrame([request.features], columns=FEATURE_NAMES)
    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]

    label = label_encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities))

    prob_dict = {
        label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(probabilities)
    }

    return PredictionResponse(
        prediction=label,
        confidence=round(confidence, 4),
        probabilities=prob_dict,
    )
