from datetime import datetime, timezone
from typing import Optional, Literal
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# The trained sklearn pipeline (TF-IDF vectorizers + OneHotEncoder + 
# LogisticRegression) is loaded once at startup. The pipeline expects a
# DataFrame with columns: diagnosis_text, admission_text, history_text,
# age_numeric, gender, unittype.
MODEL_PATH = Path(__file__).parent / "models" / "final_model.joblib"
model = joblib.load(MODEL_PATH)
MODEL_VERSION = "1.0.0"

# Training median age, used as default when age is missing or unparseable.
# This matches the fillna(median) logic in model_development.ipynb.
MEDIAN_AGE = 66.0

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
# These match the API specification from the assessment.
# The API accepts unit_type (with underscore) per the spec,
# but the model expects unittype (no underscore) from the patient table.

class Demographics(BaseModel):
    gender: Optional[str] = None
    age: Optional[str] = None
    unit_type: Optional[str] = None


class PredictRequest(BaseModel):
    patient_id: Optional[str] = None
    diagnosis_text: str = Field(..., min_length=1)
    admission_text: Optional[str] = None
    history_text: Optional[str] = None
    demographics: Optional[Demographics] = None


class Probability(BaseModel):
    alive: float
    expired: float


class PredictResponse(BaseModel):
    patient_id: str
    prediction: Literal["Alive", "Expired"]
    probability: Probability
    model_version: str
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint for container orchestration."""
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Accept patient clinical text and demographics, return mortality
    prediction with probability scores.
    
    Preprocessing mirrors the training pipeline:
      - Age string converted to float, "> 89" mapped to 90,
        missing/invalid defaults to training median (66)
      - Empty optional text fields default to empty string
      - API field unit_type mapped to model column unittype
      - Gender/unittype passed as strings; OneHotEncoder in the
        pipeline handles encoding (handle_unknown="ignore" means
        unseen categories are silently zeroed out)
    """
    # Parse demographics, defaulting to empty Demographics if not provided
    demographics = payload.demographics or Demographics()
    
    # Convert age string to numeric, matching training preprocessing:
    #   "> 89" → 90 (HIPAA de-identification convention)
    #   empty/missing → 66.0 (training median)
    #   invalid string → 66.0 (safe fallback)
    age_str = demographics.age or ""
    if age_str == "> 89":
        age_numeric = 90.0
    elif age_str == "":
        age_numeric = MEDIAN_AGE
    else:
        try:
            age_numeric = float(age_str)
        except ValueError:
            age_numeric = MEDIAN_AGE

    # Build input DataFrame with columns matching the trained pipeline.
    # Note: API spec uses "unit_type" but model expects "unittype".
    input_df = pd.DataFrame([{
        "diagnosis_text": payload.diagnosis_text,
        "admission_text": payload.admission_text or "",
        "history_text": payload.history_text or "",
        "age_numeric": age_numeric,
        "gender": demographics.gender or "",
        "unittype": demographics.unit_type or "",
    }])

    # Run the full pipeline: TF-IDF vectorization → encoding → prediction
    proba = model.predict_proba(input_df)[0]
    p_alive = round(float(proba[0]), 4)
    p_expired = round(float(proba[1]), 4)
    prediction = "Expired" if p_expired > p_alive else "Alive"

    return PredictResponse(
        patient_id=payload.patient_id or "",
        prediction=prediction,
        probability=Probability(alive=p_alive, expired=p_expired),
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )