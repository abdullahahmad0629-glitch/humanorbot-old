"""
app.py - a small web API around our model, used by the React page.

    GET  /          -> {"status": "ok"}  (health check)
    POST /predict   {"text": "...", "model": "base" or "finetuned"}
                 -> {"model": "finetuned", "label": "ai", "ai_probability": 0.97}

The model (Qwen2.5-1.5B-Instruct + our LoRA adapter) is loaded once at start-up.
"base" switches the LoRA adapter off, "finetuned" leaves it on.
"""

import os
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict import Predictor
from preprocess import clean_text
from train import predict_ai_prob

MODELS_DIR = os.environ.get("MODELS_DIR", "model")   # folder with results.json + lora_adapter/

app = FastAPI(title="humanorbot API")

# Allow the React page (hosted on GitHub Pages, a different website) to call this API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = Predictor(MODELS_DIR)
lock = threading.Lock()   # one prediction at a time (switching the adapter on/off is not thread-safe)
last_used = time.time()   # time of the last prediction (used to stop the server when idle)


class PredictRequest(BaseModel):
    text: str
    model: str = "finetuned"


@app.get("/")
def health():
    return {"status": "ok", "models": ["base", "finetuned"]}


@app.post("/predict")
def predict(request: PredictRequest):
    global last_used
    last_used = time.time()
    text = clean_text(request.text)
    if not text:
        raise HTTPException(status_code=400, detail="The text is empty.")
    if request.model not in ("base", "finetuned"):
        raise HTTPException(status_code=400, detail="model must be 'base' or 'finetuned'.")

    with lock:
        if request.model == "base":
            with predictor.model.disable_adapter():     # LoRA off = base model
                prob = predict_ai_prob(predictor.model, predictor.tokenizer, [text],
                                       predictor.max_len, 1, predictor.device)[0]
        else:
            prob = predict_ai_prob(predictor.model, predictor.tokenizer, [text],
                                   predictor.max_len, 1, predictor.device)[0]

    prob = float(prob)
    return {"model": request.model,
            "label": "ai" if prob >= 0.5 else "human",
            "ai_probability": round(prob, 4)}
