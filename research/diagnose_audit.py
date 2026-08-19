from __future__ import annotations

import json, re, random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wfdb
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'research_results'
OUT.mkdir(exist_ok=True)
for folder in ['diagnosis', 'experiments', 'robustness', 'plots', 'checkpoints', 'configs']:
    (OUT / folder).mkdir(exist_ok=True)

cfg = json.loads((ROOT / 'ml/configs/config.json').read_text())
label_cfg = json.loads((ROOT / 'ml/configs/label_mapping.json').read_text())
CLASS_NAMES = label_cfg['class_names']
AAMI = label_cfg['symbol_groups']
SYMBOL_TO_CLASS = {symbol: cls for cls, symbols in AAMI.items() for symbol in symbols}
EXPECTED_AAMI = {'N':['N','L','R','e','j'], 'S':['A','a','J','S'], 'V':['V','E'], 'F':['F'], 'Q':['Q','/','f']}
MAPPING_EXACT = AAMI == EXPECTED_AAMI and CLASS_NAMES == ['N','S','V','F','Q'] and len(SYMBOL_TO_CLASS) == sum(len(x) for x in EXPECTED_AAMI.values())

DATA_ROOT = Path(open('/home/ubuntu/kaggle_mitbih_path.txt').read().strip())
hea = sorted(DATA_ROOT.rglob('*.hea'))
if not hea:
    raise FileNotFoundError(f'No WFDB headers found under {DATA_ROOT}')
DATA_DIR = hea[0].parent
RECORDS = sorted({p.stem for p in DATA_DIR.glob('*.hea') if (DATA_DIR / f'{p.stem}.atr').exists()})

def subject_id(record: str) -> str:
    try:
        text = (DATA_DIR / f'{record}.hea').read_text(errors='ignore')
        for line in text.splitlines():
            if line.startswith('#'):
                m = re.search(r'#\s*(\d+)', line)
                if m: return m.group(1)
    except Exception:
        pass
    return record

def group_split(records, seed=42):
    groups = np.array([subject_id(r) for r in records]); idx = np.arange(len(records))
    gss = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed)
    tr_idx, te_idx = next(gss.split(idx, groups=groups))
    tr_records = [records[i] for i in tr_idx]; te_records = [records[i] for i in te_idx]
    groups_tr = np.array([subject_id(r) for r in tr_records]); idx_tr = np.arange(len(tr_records))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed+1)
    tr2, va2 = next(gss2.split(idx_tr, groups=groups_tr))
    return [tr_records[i] for i in tr2], [tr_records[i] for i in va2], te_records

train_records, val_records, test_records = group_split(RECORDS, int(cfg['seed']))

def raw_symbol_counts(records):
    counts = Counter()
    rows = []
    for rec in records:
        ann = wfdb.rdann(str(DATA_DIR / rec), 'atr')
        local = Counter(ann.symbol)
        counts.update(local)
        rows.append({'record': rec, 'subject_id': subject_id(rec), **dict(local)})
    return counts, pd.DataFrame(rows).fillna(0)

all_symbol_counts, _ = raw_symbol_counts(RECORDS)
train_symbol_counts, _ = raw_symbol_counts(train_records)
val_symbol_counts, _ = raw_symbol_counts(val_records)
test_symbol_counts, _ = raw_symbol_counts(test_records)

def load_record_beats(record, pre=90, post=90):
    sig, _ = wfdb.rdsamp(str(DATA_DIR / record)); ann = wfdb.rdann(str(DATA_DIR / record), 'atr')
    signal = sig[:, 0].astype(np.float32); beats = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol not in SYMBOL_TO_CLASS: continue
        start, end = int(sample)-pre, int(sample)+post
        if start < 0 or end > len(signal): continue
        beat = signal[start:end].copy(); med = np.median(beat); scale = np.median(np.abs(beat-med))*1.4826
        if not np.isfinite(scale) or scale < 1e-6: scale = np.std(beat)+1e-6
        beat = np.clip((beat-med)/scale, -8, 8).astype(np.float32)
        cls = SYMBOL_TO_CLASS[symbol]
        beats.append({'record':record, 'subject_id':subject_id(record), 'sample':int(sample), 'symbol':symbol, 'class_name':cls, 'class_index':CLASS_NAMES.index(cls), 'beat':beat})
    return beats

def collect_beats(records, max_per_class=None):
    all_beats=[]; per_class=Counter()
    for rec in records:
        for beat in load_record_beats(rec, cfg['beat_pre_samples'], cfg['beat_post_samples']):
            if max_per_class is not None and per_class[beat['class_name']] >= max_per_class: continue
            all_beats.append(beat); per_class[beat['class_name']] += 1
    return all_beats

def build_sequences(beats, seq_len):
    by_record=defaultdict(list)
    for beat in beats: by_record[beat['record']].append(beat)
    X=[]; y=[]; meta=[]
    for record, arr in by_record.items():
        arr=sorted(arr,key=lambda b:b['sample'])
        for i, beat in enumerate(arr):
            seq=arr[max(0,i-seq_len+1):i+1]; pad=seq_len-len(seq); waves=np.zeros((seq_len,len(beat['beat'])),dtype=np.float32)
            for j,item in enumerate(seq,pad): waves[j]=item['beat']
            X.append(waves); y.append(beat['class_index']); meta.append({k:beat[k] for k in ['record','subject_id','sample','symbol','class_name']})
    return np.stack(X), np.asarray(y,dtype=np.int64), pd.DataFrame(meta)

split_beats = {'train': collect_beats(train_records, cfg['max_beats_per_class']), 'validation': collect_beats(val_records, cfg['max_beats_per_class']), 'test': collect_beats(test_records, cfg['max_beats_per_class'])}
split_arrays = {name: build_sequences(beats, cfg['sequence_length']) for name, beats in split_beats.items()}

# Complete class and original-symbol counts before/after mapping.
rows=[]
for split, records, beats in [('train',train_records,split_beats['train']),('validation',val_records,split_beats['validation']),('test',test_records,split_beats['test'])]:
    raw = Counter(b['symbol'] for b in beats); mapped = Counter(b['class_name'] for b in beats)
    for symbol in sorted(set(all_symbol_counts) | set(raw)):
        rows.append({'split':split,'level':'original_symbol','label':symbol,'count':int(raw.get(symbol,0)),'mapped_class':SYMBOL_TO_CLASS.get(symbol,'UNMAPPED')})
    for cls in CLASS_NAMES:
        rows.append({'split':split,'level':'five_class','label':cls,'count':int(mapped.get(cls,0)),'mapped_class':cls})
for symbol,count in sorted(all_symbol_counts.items()):
    rows.append({'split':'all_records_raw_annotations','level':'original_symbol','label':symbol,'count':int(count),'mapped_class':SYMBOL_TO_CLASS.get(symbol,'UNMAPPED')})
class_counts = pd.DataFrame(rows)
class_counts.to_csv(OUT/'class_counts.csv', index=False)

# Model architecture for loading the committed RNN checkpoint and exact probabilities.
from ml.inference import TemporalClassifier
X_test, y_test, meta_test = split_arrays['test']
checkpoint = torch.load(ROOT/'ml/models/best_model.pt', map_location='cpu', weights_only=False)
input_shape = checkpoint.get('input_shape', [cfg['sequence_length'], cfg['beat_pre_samples']+cfg['beat_post_samples']])
model_type = checkpoint.get('model_type','RNN')
model = TemporalClassifier(input_shape[1], model_type, hidden=cfg['hidden_size'], layers=cfg['num_layers'], dropout=cfg['dropout'], n_classes=len(CLASS_NAMES))
model.load_state_dict(checkpoint['state_dict']); model.eval()
with torch.no_grad():
    logits, attention = model(torch.tensor(X_test,dtype=torch.float32)); probs = torch.softmax(logits,dim=1).numpy()
pred = probs.argmax(axis=1)

precision, recall, f1, support = precision_recall_fscore_support(y_test, pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0)
metric_rows=[]
for i, cls in enumerate(CLASS_NAMES):
    truth=(y_test==i).astype(int); score=probs[:,i]
    metric_rows.append({'class':cls,'precision':precision[i],'recall':recall[i],'f1':f1[i],'support':int(support[i]),'auroc_ovr':roc_auc_score(truth,score) if len(np.unique(truth))==2 else np.nan,'auprc_ovr':average_precision_score(truth,score) if truth.sum()>0 else np.nan})
per_class = pd.DataFrame(metric_rows); per_class.to_csv(OUT/'per_class_metrics.csv', index=False)

cm = confusion_matrix(y_test,pred,labels=np.arange(len(CLASS_NAMES))); cm_norm = cm / np.maximum(cm.sum(axis=1,keepdims=True),1)
pd.DataFrame(cm,index=CLASS_NAMES,columns=CLASS_NAMES).to_csv(OUT/'diagnosis/confusion_matrix_raw.csv')
pd.DataFrame(cm_norm,index=CLASS_NAMES,columns=CLASS_NAMES).to_csv(OUT/'diagnosis/confusion_matrix_normalized.csv')
fig, axes = plt.subplots(1,2,figsize=(12,5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,ax=axes[0]); axes[0].set_title('Raw confusion matrix'); axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
sns.heatmap(cm_norm,annot=True,fmt='.2f',cmap='Blues',vmin=0,vmax=1,xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,ax=axes[1]); axes[1].set_title('Row-normalized confusion matrix'); axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
fig.tight_layout(); fig.savefig(OUT/'confusion_matrix.png',dpi=180); plt.close(fig)

pred_df=meta_test.copy(); pred_df['true_class']=[CLASS_NAMES[i] for i in y_test]; pred_df['pred_class']=[CLASS_NAMES[i] for i in pred]; pred_df['confidence']=probs.max(axis=1); pred_df['correct']=y_test==pred
pred_df.to_csv(OUT/'diagnosis/baseline_predictions.csv',index=False)
error_by_symbol = pred_df.groupby(['symbol','class_name']).agg(n=('correct','size'),errors=('correct',lambda s: int((~s).sum())),error_rate=('correct',lambda s: float((~s).mean()))).reset_index().sort_values(['error_rate','n'],ascending=False)
error_by_symbol.to_csv(OUT/'diagnosis/error_by_original_symbol.csv',index=False)

train_groups={subject_id(r) for r in train_records}; val_groups={subject_id(r) for r in val_records}; test_groups={subject_id(r) for r in test_records}
split_integrity={'records_total':len(RECORDS),'train_records':train_records,'validation_records':val_records,'test_records':test_records,'train_subjects':sorted(train_groups),'validation_subjects':sorted(val_groups),'test_subjects':sorted(test_groups),'record_intersections':{'train_val':sorted(set(train_records)&set(val_records)),'train_test':sorted(set(train_records)&set(test_records)),'val_test':sorted(set(val_records)&set(test_records))},'subject_intersections':{'train_val':sorted(train_groups&val_groups),'train_test':sorted(train_groups&test_groups),'val_test':sorted(val_groups&test_groups)},'no_record_leakage':not(bool(set(train_records)&set(val_records) or set(train_records)&set(test_records) or set(val_records)&set(test_records))),'no_subject_leakage':not(bool(train_groups&val_groups or train_groups&test_groups or val_groups&test_groups)),'mapping_exact':MAPPING_EXACT,'symbol_to_class':SYMBOL_TO_CLASS}
(OUT/'diagnosis/split_integrity.json').write_text(json.dumps(split_integrity,indent=2))
(OUT/'diagnosis/mapping_verification.json').write_text(json.dumps({'expected':EXPECTED_AAMI,'implemented':AAMI,'exact':MAPPING_EXACT,'unmapped_observed_symbols':sorted(set(all_symbol_counts)-set(SYMBOL_TO_CLASS))},indent=2))

summary={'model_type':model_type,'train_records':len(train_records),'validation_records':len(val_records),'test_records':len(test_records),'train_sequences':int(len(split_arrays['train'][1])),'validation_sequences':int(len(split_arrays['validation'][1])),'test_sequences':int(len(y_test)),'mapping_exact':MAPPING_EXACT,'record_leakage':False,'subject_leakage':False,'macro_f1':float(f1.mean()),'macro_auprc':float(np.nanmean(per_class.auprc_ovr)),'macro_auroc':float(np.nanmean(per_class.auroc_ovr))}
(OUT/'diagnosis/audit_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('split class counts')
print(class_counts[(class_counts.level=='five_class')].pivot(index='label',columns='split',values='count'))
print('per class')
print(per_class.to_string(index=False))
print('top symbol errors')
print(error_by_symbol.head(15).to_string(index=False))
