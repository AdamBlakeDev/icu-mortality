# ICU Mortality Prediction Challenge

Submission for the Senior Data Scientist take-home assessment.
Predicts ICU patient mortality (Alive vs. Expired) from clinical text
using the eICU Collaborative Research Database Demo (v2.0.1).

## Project Structure
```
submission/
├── README.md
├── requirements.txt
│
├── part1_model_development/
│   ├── data_exploration.ipynb      # Dataset exploration and feature selection
│   ├── data_exploration.py         # Jupytext-synced script
│   ├── model_development.ipynb     # Feature engineering, training, evaluation
│   └── model_development.py        # Jupytext-synced script
│
├── part2_deployment/
│   ├── app.py                      # FastAPI application
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── README.md                   # API documentation and usage
│   ├── requirements.txt
│   └── models/
│       └── final_model.joblib      # Trained pipeline
│
└── part3_documentation/
    └── MODEL_CARD.md               # Model card (Mitchell et al., 2019)
```

## Part 1: Model Development

Two Jupyter notebooks (paired with `.py` scripts via Jupytext):

- **data_exploration.ipynb:** Dataset profiling, table coverage analysis,
  text column classification, feature set export.
- **model_development.ipynb:** 24h temporal filtering, text aggregation,
  API field mapping, model training (5 models compared), evaluation,
  and final model export.

**Final model:** Separate TF-IDF + Logistic Regression (AUROC ~0.82)

### Setup
```bash
cd submission
pip install -r requirements.txt
```

Notebooks expect the eICU SQLite database at:
`data/raw/eicu/eicu_v2_0_1.sqlite3`

## Part 2: Model Deployment

Dockerized FastAPI service with `/health` and `/predict` endpoints.
```bash
cd submission/part2_deployment
docker compose up --build
```

Test:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"diagnosis_text": "sepsis respiratory failure"}'
```

See `part2_deployment/README.md` for full API documentation.

## Part 3: Model Card

Located at `part3_documentation/MODEL_CARD.md`. Covers model details,
intended use, training data, metrics, ethical considerations, and
known limitations following the Mitchell et al. (2019) framework.

## AI Tool Usage

AI tools (Claude) were used throughout this assessment for:
- Brainstorming project structure, approach, and modeling decisions
- Iterating on feature engineering decisions and aggregation logic
- Debugging SQL queries and sklearn pipeline configuration
- Code generation for boilerplate (API setup, Dockerfile, evaluation loops)
- Drafting and refining markdown documentation and model card
- Clinical domain knowledge (interpreting medical abbreviations, validating
  feature importance against clinical expectations, understanding ICU workflows)
- Reviewing notebook narrative flow

All code, decisions, and documentation reflect my understanding
and were validated against the data before inclusion.
