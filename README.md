# StudentItch — AI/ML-Powered Problem Bank

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20model-F7931E?logo=scikitlearn&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude%20API-generation-D97757)

A curated bank of 60 real-world student project problem statements, plus:

1. A **trained scikit-learn regression model** that predicts a problem's
   *Build Score* (0–100 feasibility rating) from structured features —
   real supervised ML, not an LLM call.
2. A **FastAPI backend** that serves the problem bank, runs that model for
   inference, and proxies the two generative features (project brief +
   "Itch Bot" chat) to the Claude API server-side.
3. The original frontend (vanilla HTML/CSS/JS), now calling the backend
   instead of hitting an LLM API directly from the browser.

## Architecture

```
frontend/ (static HTML/JS)
     │  fetch('/api/...')
     ▼
backend/main.py (FastAPI)
     ├── /api/problems, /api/problems/{id}      → serves data/problems.json
     ├── /api/predict-score                     → ml/artifacts/build_score_model.joblib (scikit-learn inference)
     └── /api/problems/{id}/brief, /api/chat    → Anthropic Claude API (server-side key)

ml/train_build_score_model.py  → trains + evaluates the model, writes ml/artifacts/
notebooks/build_score_model.ipynb → EDA + model comparison walkthrough (same pipeline, annotated)
```

## The ML model — Build Score prediction

**Task:** given a problem's category, difficulty, time window, data
availability, and how many domain tags/stack items it touches, predict its
Build Score (0–100).

**Features:** `category`, `difficulty`, `dataAvailability` (one-hot encoded)
+ `avg_weeks`, `week_span`, `num_domain_tags`, `num_stack_items` (numeric).
Individual domain tags are folded into a *count* rather than one-hot encoded
per-tag — with only 60 rows, one-hot-encoding ~20 distinct tags blew up the
feature space and caused severe overfitting (see the notebook for the
before/after).

**Model selection:** `ml/train_build_score_model.py` compares a
mean-baseline (`DummyRegressor`), `Ridge`, `RandomForestRegressor`, and
`GradientBoostingRegressor` via 5-fold cross-validated R², with a small
`GridSearchCV` hyperparameter sweep for each, and ships whichever real
model generalizes best.

**Results (n=60 problems, 80/20 split, current run):**

| Model | CV R² (mean, 5-fold) |
|---|---|
| baseline (predict the mean) | −0.399 |
| **ridge (selected)** | **−0.384** |
| random forest | −0.458 |
| gradient boosting | −0.503 |

Held-out test split: MAE ≈ 10.6, RMSE ≈ 13.5.

**Honest take:** this is a genuinely small dataset (60 labeled examples),
and Build Score as currently labeled has a real subjective/curatorial
component that these structured features don't fully explain — the
selected model only marginally beats predicting the average. It's shipped
anyway because it's the best available real model, and the API/UI don't
overstate its precision. See `notebooks/build_score_model.ipynb` for the
full EDA and reasoning, and `ml/artifacts/metrics.json` for exact numbers
from your local run (re-training will shift these slightly depending on
random splits). The clear path to a materially better model is more
labeled data — see "Next steps" below.

Full metrics are written to `ml/artifacts/metrics.json` and feature
importance to `ml/artifacts/feature_importance.png` every time you retrain.

## Setup

```bash
git clone <this-repo>
cd studentitch-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY

# train (or retrain) the ML model — writes ml/artifacts/
python ml/train_build_score_model.py

# run the backend (also serves the frontend at http://localhost:8000)
uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000` — the frontend, brief generator, Itch Bot, and
the "Score your own idea" ML tool are all served from the same FastAPI app.

## Project layout

```
data/problems.json                    60 problem statements (id, title, category, difficulty, ...)
ml/train_build_score_model.py         feature engineering + model comparison + training
ml/artifacts/                         trained model, metrics.json, feature_importance.png (generated)
notebooks/build_score_model.ipynb     EDA + model walkthrough, executed with real outputs
backend/main.py                       FastAPI app: problem bank, ML inference, Claude API proxy
frontend/index.html                   the site (unchanged UI, calls /api/* instead of an LLM directly)
requirements.txt
.env.example
```

## Retraining the model

Add more problems to `data/problems.json` (same schema as the existing
entries) and re-run:

```bash
python ml/train_build_score_model.py
```

This overwrites `ml/artifacts/build_score_model.joblib`, which the backend
loads lazily on first request — restart `uvicorn` to pick up a freshly
trained model.

## Next steps

- Collect more labeled problems (100+) — the single highest-leverage
  improvement for model quality at this dataset size.
- Have a second person independently score a held-out set of problems to
  check inter-rater reliability on Build Score itself before trusting the
  model to explain much more variance than it currently does.
- Consider adding a text-based feature (e.g. a TF-IDF or embedding vector
  of the title/context) alongside the structured features — the current
  model only sees metadata, not what the problem is actually about.
