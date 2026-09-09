"""
StudentItch backend — FastAPI service.

Responsibilities:
  1. Serve the problem bank (data/problems.json).
  2. Run real ML inference: predict a Build Score for a new/edited problem
     using the trained scikit-learn model in ml/artifacts/.
  3. Proxy the two AI-generation features (project brief + Itch Bot chat) to
     the Anthropic API using a server-side API key, so the key never has to
     live in browser JS.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Environment:
    ANTHROPIC_API_KEY must be set (see .env.example).
"""
import json
import os
from typing import List, Optional

import joblib
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "problems.json")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "artifacts", "build_score_model.joblib")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

MODEL_NAME = "claude-sonnet-4-6"

app = FastAPI(title="StudentItch API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_client: Optional[Anthropic] = None


def get_anthropic_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY is not set on the server. See .env.example.",
            )
        _client = Anthropic(api_key=api_key)
    return _client


def load_problems() -> List[dict]:
    with open(DATA_PATH) as f:
        return json.load(f)


_model_bundle = None


def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail="Model artifact not found. Run: python ml/train_build_score_model.py",
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


PROBLEMS = load_problems()
PROBLEMS_BY_ID = {p["id"]: p for p in PROBLEMS}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class BriefRequest(BaseModel):
    problem_id: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)


class PredictScoreRequest(BaseModel):
    category: str
    difficulty: str  # Beginner | Intermediate | Advanced
    dataAvailability: str  # e.g. "Public dataset exists", "Needs field collection", "Needs synthetic data"
    min_weeks: int
    max_weeks: int
    num_domain_tags: int = 3
    num_stack_items: int = 4


# ---------------------------------------------------------------------------
# Routes — problem bank
# ---------------------------------------------------------------------------
@app.get("/api/problems")
def list_problems():
    return PROBLEMS


@app.get("/api/problems/{problem_id}")
def get_problem(problem_id: str):
    problem = PROBLEMS_BY_ID.get(problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")
    return problem


# ---------------------------------------------------------------------------
# Route — ML inference (real model, no LLM call)
# ---------------------------------------------------------------------------
@app.post("/api/predict-score")
def predict_score(req: PredictScoreRequest):
    """Predict a Build Score (0-100) for a candidate problem statement using
    the trained scikit-learn regression model — this is genuine ML
    inference, not an LLM call."""
    bundle = get_model_bundle()
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    avg_weeks = (req.min_weeks + req.max_weeks) / 2
    week_span = req.max_weeks - req.min_weeks
    row = {
        "category": req.category,
        "difficulty": req.difficulty,
        "dataAvailability": req.dataAvailability,
        "avg_weeks": avg_weeks,
        "week_span": week_span,
        "num_domain_tags": req.num_domain_tags,
        "num_stack_items": req.num_stack_items,
    }
    X = pd.DataFrame([row])[feature_columns]
    pred = float(pipeline.predict(X)[0])
    pred_clamped = max(0.0, min(100.0, pred))
    return {
        "predicted_build_score": round(pred_clamped, 1),
        "model": bundle.get("model_name", "unknown"),
    }


# ---------------------------------------------------------------------------
# Routes — LLM-backed features (server-side key, proxied from the frontend)
# ---------------------------------------------------------------------------
BRIEF_PROMPT_TEMPLATE = """You are helping a student turn a problem statement into a starter project brief for a college build.

Problem: "{title}"
Category: {category}
Context: {context}
Difficulty: {difficulty}
Suggested stack: {stack}
Time window: {min_weeks}-{max_weeks} weeks
Data availability: {data_availability}

Write a compact starter brief with these sections, using short markdown headers (##) and bullet points, no long paragraphs:
## Problem framing (2-3 sentences)
## MVP scope (bullets, 4-6 items, genuinely buildable in the time window)
## Suggested architecture (bullets, 3-5 items)
## 3 milestones (bullets, one line each)
## Stretch goals (bullets, 2-3 items)

Keep it grounded, specific to this exact problem, and realistic for a 1-4 person student team. No preamble, start directly with the first header."""


@app.post("/api/problems/{problem_id}/brief")
def generate_brief(problem_id: str):
    problem = PROBLEMS_BY_ID.get(problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")

    client = get_anthropic_client()
    prompt = BRIEF_PROMPT_TEMPLATE.format(
        title=problem["title"],
        category=problem["category"],
        context=problem["context"],
        difficulty=problem["difficulty"],
        stack=", ".join(problem["suggestedStack"]),
        min_weeks=problem["estimatedTimeWeeks"][0],
        max_weeks=problem["estimatedTimeWeeks"][1],
        data_availability=problem["dataAvailability"],
    )
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    return {"brief": text}


ITCHBOT_SYSTEM_TEMPLATE = """You are Itch Bot, the recommendation assistant for StudentItch, a bank of real-world student project problem statements.
You may ONLY recommend problems that appear in this JSON list — never invent new ones:
{problems_json}

Rules:
- Be conversational, concise, encouraging. Short paragraphs or a short bulleted list.
- When recommending, name the exact title and give a one-line reason it fits.
- Ask a clarifying question only if you truly can't make a reasonable recommendation without it.
- Use the user's stated skill level, interests, or time available to filter picks.
- Keep responses under ~120 words unless asked for more detail."""


@app.post("/api/chat")
def chat(req: ChatRequest):
    client = get_anthropic_client()
    compact = [
        {
            "id": p["id"],
            "title": p["title"],
            "category": p["category"],
            "difficulty": p["difficulty"],
            "buildScore": p["buildScore"],
            "weeks": p["estimatedTimeWeeks"],
            "stack": p["suggestedStack"],
        }
        for p in PROBLEMS
    ]
    system_prompt = ITCHBOT_SYSTEM_TEMPLATE.format(problems_json=json.dumps(compact))
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": m.role, "content": m.content} for m in req.messages[-10:]],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    return {"reply": text}


@app.get("/api/health")
def health():
    return {"status": "ok", "problems_loaded": len(PROBLEMS)}


# Serve the static frontend (index.html + assets) at the root, after all
# /api routes are registered, so /api/* takes precedence.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
