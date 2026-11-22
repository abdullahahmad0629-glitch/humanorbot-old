# humanorbot API: runs our model on Kaggle's free GPU and makes it reachable from
# the web page through a free Cloudflare "quick tunnel" (no account needed).
#
# Run with  Save Version -> Save & Run All.  It stops BY ITSELF after
# IDLE_MINUTES without requests, or after MAX_MINUTES in total; the run then
# ends and Kaggle releases the GPU.
#
# Notebook settings: Accelerator = GPU T4, Internet = on,
# Input = the "humanorbot" notebook (its output has the trained model + src/).

IDLE_MINUTES = 20   # stop after this many minutes without a request
MAX_MINUTES = 90    # stop after this many minutes no matter what

import glob, os, re, shutil, subprocess, sys, threading, time
start = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "fastapi", "uvicorn"], check=True)

# 1. Find the trained model (results.json + lora_adapter/) and our 3 scripts
hits = glob.glob("/kaggle/input/**/lora_adapter/adapter_config.json", recursive=True)
hits += glob.glob("/kaggle/working/**/lora_adapter/adapter_config.json", recursive=True)
hits = [os.path.dirname(os.path.dirname(h)) for h in hits]
hits = [h for h in hits if os.path.exists(h + "/results.json") and "/api/" not in h]
hits.sort(key=lambda h: "test_run" in h)          # prefer the full run
assert hits, "No trained model found: add the humanorbot notebook as an input."
models_dir = hits[0]
print("Trained model:", models_dir)

api_dir = "/kaggle/working/api"
os.makedirs(api_dir, exist_ok=True)
for name in ["preprocess.py", "train.py", "predict.py"]:
    candidates = [os.path.join(os.path.dirname(models_dir), "src", name)]
    candidates += glob.glob("/kaggle/input/**/src/" + name, recursive=True)
    found = [c for c in candidates if os.path.exists(c)]
    assert found, f"Could not find {name}"
    shutil.copy(found[0], os.path.join(api_dir, name))

APP_PY = r'''"""
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
'''
with open(os.path.join(api_dir, "app.py"), "w", encoding="utf-8") as f:
    f.write(APP_PY)

# 2. Load the model and start the API server (inside this notebook, port 8000)
os.chdir(api_dir)
sys.path.insert(0, api_dir)
os.environ["MODELS_DIR"] = models_dir
import uvicorn
import app as api                                   # loads Qwen + LoRA (1-2 minutes)
server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=8000, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
time.sleep(3)

# 3. Open a public Cloudflare quick tunnel to the server
if not os.path.exists("cloudflared"):
    subprocess.run(["wget", "-q", "-O", "cloudflared",
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
    os.chmod("cloudflared", 0o755)
tunnel = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
url = None
for line in tunnel.stderr:
    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if match:
        url = match.group(0)
        break
threading.Thread(target=lambda: [None for _ in tunnel.stderr], daemon=True).start()   # keep reading its log

print("\n" + "=" * 70)
print("API ADDRESS:", url)
print("Paste it into the 'Model server address' box on the web page.")
print(f"Stops after {IDLE_MINUTES} min without requests, or {MAX_MINUTES} min in total.")
print("=" * 70 + "\n", flush=True)

# 4. Stay up until idle for too long or the time limit is reached
api.last_used = time.time()
while True:
    time.sleep(30)
    idle = (time.time() - api.last_used) / 60
    total = (time.time() - start) / 60
    if idle >= IDLE_MINUTES:
        reason = f"no requests for {IDLE_MINUTES} minutes"
        break
    if total >= MAX_MINUTES:
        reason = f"time limit of {MAX_MINUTES} minutes reached"
        break
    print(f"[{total:5.1f} min] running, idle for {idle:4.1f} min", flush=True)

print("Stopping:", reason)
server.should_exit = True
tunnel.terminate()
time.sleep(2)
print("Stopped. The GPU is released when this run ends.")
