# Part 2: Model Deployment

## Quick Start
```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## Endpoints

### GET /health
Health check for container orchestration.
```bash
curl http://localhost:8000/health
```

### POST /predict
Returns mortality prediction for a patient.
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "test_001",
    "diagnosis_text": "cardiovascular arrhythmias ventricular disorders",
    "admission_text": "admission diagnosis cardiac arrest",
    "history_text": "Comprehensive Progress notes Organ Systems Cardiovascular",
    "demographics": {
      "gender": "Male",
      "age": "72",
      "unit_type": "Med-Surg ICU"
    }
  }'
```

Only `diagnosis_text` is required. All other fields are optional.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| Port | 8000 | Exposed via docker-compose.yml |
| MODEL_VERSION | 1.0.0 | Set in app.py |