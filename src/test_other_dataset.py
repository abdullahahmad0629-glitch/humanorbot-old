"""
test_other_dataset.py - test the trained base and fine-tuned models on a
DIFFERENT dataset, to check whether the detector generalises.

Dataset: HC3 (Hello-SimpleAI/HC3 on Hugging Face). Questions from five areas
(Reddit ELI5, finance, medicine, open questions, Wikipedia CS/AI), each
answered by humans and by ChatGPT. Different topics, a different writing
style (answers, not essays) and a different AI model from our training data.

No training happens here: we load the LoRA adapter saved by train.py.
Base model = adapter switched off, fine-tuned model = adapter on.

Kaggle (GPU, Internet ON):
    python src/test_other_dataset.py
    python src/test_other_dataset.py --models-dir /kaggle/working/outputs --per-class 200
"""

import argparse
import glob
import json
import os
import time

import pandas as pd
from huggingface_hub import hf_hub_download

from predict import Predictor
from preprocess import SEED, clean_text
from train import evaluate, predict_ai_prob, print_comparison

DOMAINS = ["reddit_eli5", "finance", "medicine", "open_qa", "wiki_csai"]


def load_hc3(per_class, min_words):
    """Up to `per_class` human and `per_class` ChatGPT answers from EACH domain.
    label: 0 = human, 1 = AI (same as our training data)."""
    parts = []
    for domain in DOMAINS:
        path = hf_hub_download("Hello-SimpleAI/HC3", f"{domain}.jsonl", repo_type="dataset")
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for answer in item["human_answers"]:
                    rows.append((answer, 0))
                for answer in item["chatgpt_answers"]:
                    rows.append((answer, 1))
        df = pd.DataFrame(rows, columns=["text", "label"])
        df["text"] = df["text"].apply(clean_text)
        df = df[df["text"].str.split().str.len() >= min_words]   # skip very short answers
        df = df.drop_duplicates(subset="text")
        for label in [0, 1]:
            rows = df[df["label"] == label]
            sample = rows.sample(min(per_class, len(rows)), random_state=SEED)
            parts.append(sample.assign(domain=domain))
    return pd.concat(parts).reset_index(drop=True)


def find_models_dir():
    """Find the folder that train.py saved (results.json + lora_adapter/).
    Looks in /kaggle/working first, then in notebook outputs added as input.
    The full run ('outputs') is preferred over the short test ('test_run')."""
    hits = (glob.glob("/kaggle/working/**/lora_adapter/adapter_config.json", recursive=True)
            + glob.glob("/kaggle/input/**/lora_adapter/adapter_config.json", recursive=True))
    hits = [os.path.dirname(os.path.dirname(h)) for h in hits]
    hits = [h for h in hits if os.path.exists(os.path.join(h, "results.json"))]
    if not hits:
        raise SystemExit("No trained model found. Run train.py first, or add the notebook "
                         "version that ran it as an input (Add Input -> Your Work).")
    hits.sort(key=lambda h: "test_run" in h)
    return hits[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=None, help="folder with results.json and lora_adapter/")
    parser.add_argument("--out", default="/kaggle/working/hc3")
    parser.add_argument("--per-class", type=int, default=200, help="human and AI answers per domain")
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    models_dir = args.models_dir or find_models_dir()
    os.makedirs(args.out, exist_ok=True)
    print(f"Trained model: {models_dir}")

    # 1. Data
    df = load_hc3(args.per_class, args.min_words)
    print(f"HC3 sample: {len(df)} answers "
          f"(human {(df['label'] == 0).sum()}, AI {(df['label'] == 1).sum()})")
    print(df.groupby(["domain", "label"]).size().unstack().rename(columns={0: "human", 1: "AI"}))
    texts = df["text"].tolist()
    y = df["label"].values

    # 2. Models (one copy; base = adapter off)
    predictor = Predictor(models_dir)
    model, tokenizer, device = predictor.model, predictor.tokenizer, predictor.device

    results = {"settings": {**vars(args), "models_dir": models_dir}}
    t0 = time.time()
    with model.disable_adapter():
        df["base_ai_prob"] = predict_ai_prob(model, tokenizer, texts, predictor.max_len, args.batch_size, device)
    results["base"] = evaluate(y, df["base_ai_prob"].values)
    results["base"]["seconds"] = time.time() - t0

    t0 = time.time()
    df["finetuned_ai_prob"] = predict_ai_prob(model, tokenizer, texts, predictor.max_len, args.batch_size, device)
    results["finetuned"] = evaluate(y, df["finetuned_ai_prob"].values)
    results["finetuned"]["seconds"] = time.time() - t0

    # 3. Accuracy per domain
    per_domain = {}
    for domain, g in df.groupby("domain"):
        per_domain[domain] = {
            "n": len(g),
            "base_accuracy": float(((g["base_ai_prob"] >= 0.5).astype(int) == g["label"]).mean()),
            "finetuned_accuracy": float(((g["finetuned_ai_prob"] >= 0.5).astype(int) == g["label"]).mean()),
        }
    results["per_domain"] = per_domain

    # 4. Save and print
    with open(os.path.join(args.out, "hc3_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    df.to_csv(os.path.join(args.out, "hc3_predictions.csv"), index=False)

    print("\nResults on HC3 (a dataset the model was NOT trained on):")
    print_comparison(results)
    print("\nAccuracy per domain:")
    print(pd.DataFrame(per_domain).T.round(3))
    print(f"\nSaved hc3_results.json and hc3_predictions.csv in '{args.out}/'")


if __name__ == "__main__":
    main()
