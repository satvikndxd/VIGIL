import json
import time
from pathlib import Path
import numpy as np
import requests

root=Path('/home/ubuntu/VIGIL')
signal=json.loads((root/'ml/sample_waveform.json').read_text())['signal']
url='http://127.0.0.1:8000/predict'
requests.post(url,json={'signal':signal,'model':'RNN','record_id':'100'},timeout=30).raise_for_status()
lat=[]
for _ in range(30):
    t=time.perf_counter(); requests.post(url,json={'signal':signal,'model':'RNN','record_id':'100'},timeout=30).raise_for_status(); lat.append((time.perf_counter()-t)*1000)
arr=np.array(lat)
result={'requests':len(lat),'cold_start_ms':None,'p50_ms':float(np.percentile(arr,50)),'p95_ms':float(np.percentile(arr,95)),'throughput_requests_per_second':float(1000/arr.mean()),'mean_ms':float(arr.mean())}
(root/'ml/metrics/serving_benchmark.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
