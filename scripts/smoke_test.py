import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
base = 'http://127.0.0.1:8000'
required = [
    ROOT / 'ml/models/best_model.pt', ROOT / 'ml/models/RNN_best.pt',
    ROOT / 'ml/models/LSTM_best.pt', ROOT / 'ml/models/GRU_best.pt',
    ROOT / 'ml/models/BiLSTM_best.pt', ROOT / 'ml/models/BiLSTM_Attention_best.pt',
    ROOT / 'ml/configs/inference_bundle.json', ROOT / 'ml/metrics/metrics.json',
    ROOT / 'ml/metrics/model_comparison.csv', ROOT / 'ml/metrics/serving_benchmark.json',
]
assert all(p.exists() for p in required)
health = requests.get(base + '/health', timeout=20).json()
assert health['status'] == 'ok'
assert set(['RNN','LSTM','GRU','BiLSTM','BiLSTM_Attention']).issubset(set(health['available_models']))
model = requests.get(base + '/model', timeout=20).json()
assert model['classes'] == ['N','S','V','F','Q']
metrics = requests.get(base + '/metrics', timeout=20).json()
assert len(metrics['models']) >= 7
sample = requests.get(base + '/sample-signal', timeout=20).json()
for model_name in ['RNN','LSTM','GRU','BiLSTM','BiLSTM_Attention']:
    prediction = requests.post(base + '/predict', json={'signal': sample['signal'], 'model': model_name, 'record_id': sample['record_id']}, timeout=30).json()
    assert prediction['class_name'] in ['N','S','V','F','Q']
    assert abs(sum(prediction['probabilities'].values()) - 1) < 1e-5
print(json.dumps({'health': health, 'model': model, 'metric_models': len(metrics['models']), 'prediction_models_checked': 5}, indent=2))
