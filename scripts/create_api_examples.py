import json
from pathlib import Path
import requests
import wfdb

root=Path(open('/home/ubuntu/kaggle_mitbih_path.txt').read().strip())
root=next(root.rglob('*.hea')).parent
out=Path('/home/ubuntu/VIGIL/api/examples'); out.mkdir(parents=True,exist_ok=True)
found={}; candidates=[]
for record in ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124']:
    signal,_=wfdb.rdsamp(str(root/record))
    for start in [0, 900, 1800, 2700]:
        segment=signal[start:start+1800,0].astype(float).tolist()
        if len(segment)<180: continue
        r=requests.post('http://127.0.0.1:8000/predict',json={'signal':segment,'model':'RNN','record_id':record},timeout=30)
        r.raise_for_status(); item=r.json(); candidates.append(item)
        key=item['class_name']
        found.setdefault(key,item)
        if len(found)>=5 and any(max(x['probabilities'].values())<0.5 for x in candidates): break
    if len(found)>=5 and any(max(x['probabilities'].values())<0.5 for x in candidates): break
if 'N' in found: (out/'normal.json').write_text(json.dumps(found['N'],indent=2))
if 'V' in found: (out/'ventricular.json').write_text(json.dumps(found['V'],indent=2))
uncertain=min(candidates,key=lambda x:max(x['probabilities'].values()))
(out/'uncertain.json').write_text(json.dumps(uncertain,indent=2))
print('saved', [p.name for p in out.glob('*.json')])
