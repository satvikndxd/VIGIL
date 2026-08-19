# VIGIL

## Interpretable Temporal Deep Learning for ECG Arrhythmia Classification

VIGIL is a retrospective ECG analytics workstation built around a five-class MIT-BIH arrhythmia classification pipeline. It combines beat-level WFDB preprocessing, grouped record/patient-level evaluation, classical and temporal deep-learning benchmarks, an inference API, and a dense enterprise BI-style dashboard.

> **Research scope.** This is a retrospective research prototype and is not a medical diagnostic device. The outputs must not be used to make treatment decisions or to imply clinical deployment readiness.

## Problem and dataset

The project uses the Kaggle mirror `abdallahwagih/mit-bih-arrhythmia-database`, which contains MIT-BIH WFDB recordings and expert beat annotations. The research target is a five-class AAMI-style grouping:

| Class | Symbols |
|---|---|
| N | N, L, R, e, j |
| S | A, a, J, S |
| V | V, E |
| F | F |
| Q | Q, /, f |

Each beat is represented by 90 samples before and 90 samples after the annotation at 360 Hz. The sequence input contains the current beat plus up to seven preceding beats, giving an input shape of `8 × 180`.

## Leakage prevention

The split is performed by subject/record group before beat construction. Individual beats from a held-out record cannot appear in train or validation. Within each record, sequences preserve chronological order and contain only the current and preceding beats. Exact split logic is implemented in the notebook.

## Benchmark

The repository includes Logistic Regression, Random Forest, RNN, LSTM, GRU, Bi-LSTM, and Bi-LSTM + Temporal Attention checkpoints and metrics. The current measured outputs are stored in `ml/metrics/model_comparison.csv` and `ml/metrics/metrics.json`. The benchmark is not optimized for accuracy alone; it reports Macro-F1, balanced accuracy, AUROC, AUPRC, per-class metrics, calibration, training time, parameter count, and inference latency.

The current DEBUG-mode experiment selected the plain RNN as the best neural model by Macro-F1. The classical baselines remain in the results because the best overall model depends on the metric: Logistic Regression has the strongest Macro-F1 in the current run, while Random Forest has the strongest AUROC and lowest ECE. The dashboard displays these measured differences rather than implying that the attention model wins.

## Architecture

```text
ECG waveform
    ↓
WFDB beat extraction and robust normalization
    ↓
Grouped record/subject split + 8-beat sequence construction
    ↓
Logistic Regression / Random Forest / RNN / LSTM / GRU / Bi-LSTM
    ↓
Bi-LSTM + Temporal Attention benchmark
    ↓
Class probabilities + temporal attention + Integrated Gradients artifacts
    ↙                                  ↘
FastAPI inference API                  Enterprise ECG BI dashboard
```

## Repository layout

| Path | Purpose |
|---|---|
| `notebooks/` | Original and executed MIT-BIH research notebooks. |
| `ml/models/` | `RNN_best.pt`, `LSTM_best.pt`, `GRU_best.pt`, `BiLSTM_best.pt`, `BiLSTM_Attention_best.pt`, and the active `best_model.pt`. |
| `ml/configs/` | Label mapping, preprocessing, inference bundle, and experiment configuration. |
| `ml/metrics/` | Model comparison, full JSON metrics, training histories, and latency summary. |
| `ml/predictions/` | Held-out test predictions used by the analytics UI. |
| `ml/explanations/` | Integrated Gradients arrays and images. |
| `ml/plots/` | Class distribution, waveform, confusion matrix, and reliability plots. |
| `api/` | FastAPI service exposing health, model, metrics, records, explanations, and prediction endpoints. |
| `src/` | React enterprise BI dashboard. |

## Run the API

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | API and model status. |
| `GET /model` | Model version, classes, input contract, and available checkpoints. |
| `GET /metrics` | Real benchmark metrics and comparison rows. |
| `GET /sample-signal` | Bundled real MIT-BIH record-100 waveform segment. |
| `GET /records` | Held-out prediction rows for error analysis. |
| `GET /explanations` | Available Integrated Gradients metadata and attention availability. |
| `POST /predict` | Predict a raw one-dimensional ECG segment. |
| `POST /predict/batch` | Batch inference with Pydantic validation. |

Example request:

```json
{
  "signal": [0.1, 0.12, 0.09],
  "model": "RNN",
  "record_id": "100",
  "include_waveform": true
}
```

The signal must contain at least 180 samples. The response includes the predicted class, probabilities for N/S/V/F/Q, confidence, model version, preprocessing version, latency, waveform payload, and attention when available.

## Run the dashboard

```bash
npm install
npm run dev
```

The dashboard expects the API at `http://localhost:8000`. To change it:

```bash
VITE_API_BASE=http://localhost:8000 npm run dev
```

The frontend follows the supplied enterprise BI reference: compact rectangular cards, white/off-white canvas, thin gray borders, subtle shadows, restrained navy/orange class semantics, dense desktop-first grid, compact charts, and no gradients or glassmorphism. The ECG waveform is the primary visual, followed by probabilities, temporal attention, confusion matrix, model comparison, class distribution, error analysis, and model metadata.

## Serving benchmark

The live FastAPI endpoint was benchmarked with 30 real requests against the bundled MIT-BIH sample signal. The measured results are stored in `ml/metrics/serving_benchmark.json`: p50 latency `12.86 ms`, p95 latency `16.77 ms`, mean latency `13.02 ms`, and throughput `76.82 requests/second` in the sandbox validation environment. These figures are environment-specific and are not production capacity claims.

## Reproducibility

The main notebook is `notebooks/mitbih_arrhythmia_research.ipynb`. It contains the configuration, grouped split, preprocessing, benchmark, ablation, explanations, and exports. The current artifacts are committed so the dashboard can run without retraining. To retrain, mount the KaggleHub MIT-BIH dataset and rerun the notebook in `DEBUG` or `FINAL` mode.

## Limitations

The five-class grouping is a research mapping, not a clinical label system. The current experiment has severe minority-class limitations, especially for F and N predictions under the selected grouped split. Attention is descriptive rather than causal, and Integrated Gradients is an attribution diagnostic rather than a clinical explanation. The results are not external validation and should not be presented as clinical performance.
