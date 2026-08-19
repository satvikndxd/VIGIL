from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.inference import load_model, predict_ecg

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "ml"
MODELS = ML / "models"
CONFIGS = ML / "configs"
METRICS = ML / "metrics"
PREDICTIONS = ML / "predictions"
EXPLANATIONS = ML / "explanations"

BUNDLE = json.loads((CONFIGS / "inference_bundle.json").read_text())
CLASS_NAMES = BUNDLE.get("class_names", ["N", "S", "V", "F", "Q"])
DEFAULT_MODEL = BUNDLE.get("model_type", "RNN")
MODEL_VERSION = BUNDLE.get("model_version", "vigil-arrhythmia-0.1")
MODEL_PATHS = {DEFAULT_MODEL: MODELS / "best_model.pt"}
for name in ["RNN", "LSTM", "GRU", "BiLSTM", "BiLSTM_Attention"]:
    candidate = MODELS / f"{name}_best.pt"
    if candidate.exists():
        MODEL_PATHS[name] = candidate

_loaded_models: dict[str, tuple[Any, dict, str]] = {}

def get_model(name: str = DEFAULT_MODEL):
    if name not in MODEL_PATHS:
        raise HTTPException(status_code=404, detail=f"Model {name} is not available in this repository")
    if name not in _loaded_models:
        _loaded_models[name] = load_model(MODEL_PATHS[name], CONFIGS / "inference_bundle.json")
    return _loaded_models[name]


def load_metrics() -> dict[str, Any]:
    return json.loads((METRICS / "metrics.json").read_text())


def load_predictions() -> pd.DataFrame:
    path = PREDICTIONS / "test_predictions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def waveform_payload(signal: list[float], limit: int = 720) -> list[float]:
    arr = np.asarray(signal, dtype=float).reshape(-1)
    if len(arr) <= limit:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, limit).astype(int)
    return arr[idx].tolist()


class PredictRequest(BaseModel):
    signal: list[float] = Field(..., min_length=180, description="One-dimensional ECG segment")
    model: str = Field(DEFAULT_MODEL)
    record_id: str | None = None
    include_waveform: bool = True


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=128)


app = FastAPI(title="VIGIL ECG Arrhythmia API", version=MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_online": True, "model_version": MODEL_VERSION, "available_models": sorted(MODEL_PATHS)}


@app.get("/model")
def model_info():
    metrics = load_metrics()
    comparison = pd.read_csv(METRICS / "model_comparison.csv")
    if "Unnamed: 0" in comparison.columns:
        comparison = comparison.rename(columns={"Unnamed: 0": "Model"})
    selected = comparison[comparison["Model"] == DEFAULT_MODEL] if "Model" in comparison.columns else pd.DataFrame()
    parameter_count = int(selected["Params"].iloc[0]) if not selected.empty and pd.notna(selected["Params"].iloc[0]) else 0
    return {
        "model_version": MODEL_VERSION,
        "default_model": DEFAULT_MODEL,
        "available_models": sorted(MODEL_PATHS),
        "framework": "PyTorch",
        "classes": CLASS_NAMES,
        "sample_rate_hz": BUNDLE.get("sample_rate_hz", 360),
        "sequence_length": BUNDLE.get("sequence_length", 8),
        "samples_per_beat": BUNDLE.get("beat_pre_samples", 90) + BUNDLE.get("beat_post_samples", 90),
        "parameter_count": parameter_count,
        "metrics_available": list(metrics),
        "research_scope": "Retrospective research prototype; not a medical diagnostic device.",
    }


@app.get("/metrics")
def metrics():
    comparison = pd.read_csv(METRICS / "model_comparison.csv")
    if "Unnamed: 0" in comparison.columns:
        comparison = comparison.rename(columns={"Unnamed: 0": "Model"})
    comparison = comparison.replace({np.nan: None})
    metrics_json = load_metrics()
    return {"models": comparison.to_dict(orient="records"), "raw": metrics_json, "best_neural_model": DEFAULT_MODEL}


@app.get("/sample-signal")
def sample_signal():
    path = ML / "sample_waveform.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No sample waveform is bundled")
    return json.loads(path.read_text())


@app.get("/records")
def records(limit: int = 100, class_filter: str | None = None):
    df = load_predictions()
    if class_filter and class_filter != "All":
        df = df[(df["true_class"] == class_filter) | (df["pred_class"] == class_filter)]
    return {"records": df.head(max(1, min(limit, 500))).replace({np.nan: None}).to_dict(orient="records"), "total": int(len(df))}


@app.get("/records/{record_id}")
def record_detail(record_id: str):
    df = load_predictions()
    if df.empty or "record" not in df.columns:
        raise HTTPException(status_code=404, detail="No prediction records are available")
    selected = df[df["record"].astype(str) == str(record_id)]
    if selected.empty:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"record_id": record_id, "predictions": selected.replace({np.nan: None}).to_dict(orient="records")}


@app.get("/explanations")
def explanations():
    result: dict[str, Any] = {"attention_available": any("Attention" in name for name in MODEL_PATHS), "temporal_attention": [], "integrated_gradients_available": False}
    ig_path = EXPLANATIONS / "integrated_gradients.npy"
    if ig_path.exists():
        ig = np.load(ig_path)
        result["integrated_gradients_available"] = True
        result["integrated_gradients_shape"] = list(ig.shape)
        result["top_signal_regions"] = [
            {"beat_position": int(i), "magnitude": float(np.abs(ig[i]).mean())}
            for i in np.argsort(-np.abs(ig).mean(axis=1))[: min(8, len(ig))]
        ]
    return result


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.perf_counter()
    try:
        result = predict_ecg(req.signal, MODEL_PATHS.get(req.model, MODEL_PATHS[DEFAULT_MODEL]), CONFIGS / "inference_bundle.json")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result.update({
        "record_id": req.record_id,
        "confidence": max(result["probabilities"].values()),
        "model_name": req.model if req.model in MODEL_PATHS else DEFAULT_MODEL,
        "preprocessing_version": "robust-median-mad-v1",
        "latency_ms": (time.perf_counter() - start) * 1000,
        "waveform": waveform_payload(req.signal) if req.include_waveform else [],
        "explanation_note": "Temporal attention is descriptive and is not a causal explanation.",
    })
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    return {"predictions": [predict(item) for item in req.items]}
