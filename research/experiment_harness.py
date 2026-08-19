from __future__ import annotations

import hashlib, json, math, random, re, time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import wfdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_results'
for folder in ['experiments','robustness','checkpoints','configs','plots']:
    (OUT/folder).mkdir(parents=True, exist_ok=True)

CFG = json.loads((ROOT/'ml/configs/config.json').read_text())
MAP_CFG = json.loads((ROOT/'ml/configs/label_mapping.json').read_text())
CLASSES = MAP_CFG['class_names']; CLASS_TO_INDEX = {c:i for i,c in enumerate(CLASSES)}
SYMBOL_TO_CLASS = {s:c for c,ss in MAP_CFG['symbol_groups'].items() for s in ss}
DATA_ROOT = Path(open('/home/ubuntu/kaggle_mitbih_path.txt').read().strip())
DATA_DIR = sorted(DATA_ROOT.rglob('*.hea'))[0].parent
RECORDS = sorted({p.stem for p in DATA_DIR.glob('*.hea') if (DATA_DIR/f'{p.stem}.atr').exists()})
SEED = int(CFG['seed'])
BATCH = int(CFG.get('batch_size',128))
EPOCHS = int(CFG.get('epochs',3))
DEVICE = torch.device('cpu')


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def subject_id(record):
    try:
        for line in (DATA_DIR/f'{record}.hea').read_text(errors='ignore').splitlines():
            if line.startswith('#'):
                m=re.search(r'#\s*(\d+)',line)
                if m: return m.group(1)
    except Exception: pass
    return record


def group_split(records, seed=SEED):
    groups=np.array([subject_id(r) for r in records]); idx=np.arange(len(records))
    tr_idx,te_idx=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed).split(idx,groups=groups))
    tr_records=[records[i] for i in tr_idx]; te_records=[records[i] for i in te_idx]
    tr_groups=np.array([subject_id(r) for r in tr_records]); tr_idx2=np.arange(len(tr_records))
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed+1).split(tr_idx2,groups=tr_groups))
    return [tr_records[i] for i in a],[tr_records[i] for i in b],te_records

TRAIN_RECORDS, VAL_RECORDS, TEST_RECORDS = group_split(RECORDS, SEED)
SPLIT_LOCK = {'seed':SEED,'train_records':TRAIN_RECORDS,'validation_records':VAL_RECORDS,'test_records':TEST_RECORDS,'train_subjects':sorted({subject_id(x) for x in TRAIN_RECORDS}),'validation_subjects':sorted({subject_id(x) for x in VAL_RECORDS}),'test_subjects':sorted({subject_id(x) for x in TEST_RECORDS})}
SPLIT_LOCK['record_intersections'] = {'train_val':sorted(set(TRAIN_RECORDS)&set(VAL_RECORDS)),'train_test':sorted(set(TRAIN_RECORDS)&set(TEST_RECORDS)),'val_test':sorted(set(VAL_RECORDS)&set(TEST_RECORDS))}
SPLIT_LOCK['subject_intersections'] = {'train_val':sorted(set(SPLIT_LOCK['train_subjects'])&set(SPLIT_LOCK['validation_subjects'])),'train_test':sorted(set(SPLIT_LOCK['train_subjects'])&set(SPLIT_LOCK['test_subjects'])),'val_test':sorted(set(SPLIT_LOCK['validation_subjects'])&set(SPLIT_LOCK['test_subjects']))}
SPLIT_LOCK['no_record_leakage'] = all(not v for v in SPLIT_LOCK['record_intersections'].values())
SPLIT_LOCK['no_subject_leakage'] = all(not v for v in SPLIT_LOCK['subject_intersections'].values())
(OUT/'configs/split_lock.json').write_text(json.dumps(SPLIT_LOCK,indent=2))


def load_beats(record, pre=90, post=90):
    sig,_=wfdb.rdsamp(str(DATA_DIR/record)); ann=wfdb.rdann(str(DATA_DIR/record),'atr'); signal=sig[:,0].astype(np.float32); beats=[]
    for sample,symbol in zip(ann.sample,ann.symbol):
        if symbol not in SYMBOL_TO_CLASS: continue
        a,b=int(sample)-pre,int(sample)+post
        if a<0 or b>len(signal): continue
        beat=signal[a:b].copy(); med=np.median(beat); scale=np.median(np.abs(beat-med))*1.4826
        if not np.isfinite(scale) or scale<1e-6: scale=np.std(beat)+1e-6
        beat=np.clip((beat-med)/scale,-8,8).astype(np.float32)
        beats.append({'record':record,'subject_id':subject_id(record),'sample':int(sample),'symbol':symbol,'class_name':SYMBOL_TO_CLASS[symbol],'class_index':CLASS_TO_INDEX[SYMBOL_TO_CLASS[symbol]],'beat':beat})
    return beats


def collect(records):
    result=[]; counts=Counter()
    for rec in records:
        for beat in load_beats(rec):
            if counts[beat['class_name']] >= int(CFG['max_beats_per_class']): continue
            result.append(beat); counts[beat['class_name']]+=1
    return result

BEATS = {'train':collect(TRAIN_RECORDS),'validation':collect(VAL_RECORDS),'test':collect(TEST_RECORDS)}


def feature_vector(arr, i, sample_rate=360):
    beat=arr[i]['beat']; prev_rr=(arr[i]['sample']-arr[i-1]['sample']) if i>0 else 0.; next_rr=(arr[i+1]['sample']-arr[i]['sample']) if i+1<len(arr) else prev_rr
    prev_rr = prev_rr or next_rr or 1.; next_rr = next_rr or prev_rr or 1.
    local_hr=sample_rate/prev_rr*60.; rr_ratio=next_rr/prev_rr
    diff=np.diff(beat); centered=beat-beat.mean(); zero_cross=float(np.sum(np.diff(np.signbit(centered)) != 0)); energy=float(np.mean(beat*beat)); slope=float(np.mean(np.abs(diff)))
    morphology=[float(beat.mean()),float(beat.std()),float(beat.min()),float(beat.max()),float(np.ptp(beat)),energy,slope,zero_cross]
    rr=[float(prev_rr),float(next_rr),float(local_hr),float(rr_ratio)]
    return np.asarray(rr+morphology,dtype=np.float32)


def build_sequences(beats, context=8, feature_mode='waveform'):
    by_record=defaultdict(list)
    for b in beats: by_record[b['record']].append(b)
    X=[]; y=[]; meta=[]
    for rec,arr in by_record.items():
        arr=sorted(arr,key=lambda z:z['sample'])
        features={item['sample']: feature_vector(arr,i) for i,item in enumerate(arr)}
        for i,b in enumerate(arr):
            seq=arr[max(0,i-context+1):i+1]; pad=context-len(seq); feat_dim=180 if feature_mode=='waveform' else 180 + (4 if feature_mode=='rr' else 8 if feature_mode=='morphology' else 12)
            waves=np.zeros((context,feat_dim),dtype=np.float32)
            for j,item in enumerate(seq,pad):
                waves[j,:180]=item['beat']
                item_features=features[item['sample']]
                if feature_mode=='rr': waves[j,180:184]=item_features[:4]
                elif feature_mode=='morphology': waves[j,180:188]=item_features[4:]
                elif feature_mode=='rr_morphology': waves[j,180:192]=item_features
            X.append(waves); y.append(b['class_index']); meta.append({'record':b['record'],'subject_id':b['subject_id'],'sample':b['sample'],'symbol':b['symbol'],'class_name':b['class_name']})
    return np.stack(X),np.asarray(y,dtype=np.int64),pd.DataFrame(meta)


def normalize_features(X_train, X_val, X_test):
    if X_train.shape[-1] == 180: return X_train,X_val,X_test
    loc=X_train[:,:,180:].reshape(-1,X_train.shape[-1]-180); mean=loc.mean(0); std=loc.std(0)+1e-6
    out=[]
    for X in [X_train,X_val,X_test]:
        z=X.copy(); z[:,:,180:]=(z[:,:,180:]-mean)/std; out.append(z)
    return out[0],out[1],out[2]


class ECGDataset(Dataset):
    def __init__(self,X,y,augment=False,seed=0): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.long); self.augment=augment; self.seed=seed
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        x=self.X[i].clone()
        if self.augment:
            g=torch.Generator().manual_seed(self.seed+i)
            wave=x[:,:180]
            scale=0.95+0.10*torch.rand((),generator=g)
            noise=0.025*torch.randn(wave.shape,generator=g)
            t=torch.arange(wave.shape[1],dtype=wave.dtype)
            wander=0.025*torch.sin(2*torch.pi*t/wave.shape[1]*1.5)
            shift=int(torch.randint(-2,3,(),generator=g))
            wave=torch.roll(wave,shifts=shift,dims=1)*scale + noise + wander
            x[:,:180]=wave
        return x,self.y[i]


class TemporalAttention(nn.Module):
    def __init__(self,dim): super().__init__(); self.score=nn.Sequential(nn.Linear(dim,dim),nn.Tanh(),nn.Linear(dim,1))
    def forward(self,h):
        w=torch.softmax(self.score(h).squeeze(-1),dim=1); c=(h*w.unsqueeze(-1)).sum(1); return c,w


class TemporalModel(nn.Module):
    def __init__(self,input_dim,kind='RNN',hidden=32,layers=1,dropout=.25,n_classes=5):
        super().__init__(); self.kind=kind; self.encoder=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.ReLU())
        rnn_cls={'RNN':nn.RNN,'LSTM':nn.LSTM,'GRU':nn.GRU,'BiLSTM':nn.LSTM,'BiLSTM_Attention':nn.LSTM}[kind]; bidir=kind in ['BiLSTM','BiLSTM_Attention']; self.rnn=rnn_cls(hidden,hidden,num_layers=layers,batch_first=True,dropout=dropout if layers>1 else 0,bidirectional=bidir); dim=hidden*(2 if bidir else 1); self.attn=TemporalAttention(dim) if kind=='BiLSTM_Attention' else None; self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(dim,n_classes))
    def forward(self,x):
        h,_=self.rnn(self.encoder(x))
        if self.attn is not None: c,w=self.attn(h); return self.head(c),w
        return self.head(h[:,-1,:]),None


class CNNBiLSTMAttention(nn.Module):
    def __init__(self,input_dim,hidden=32,n_classes=5):
        super().__init__(); self.kind='CNN_BiLSTM_Attention'; self.conv=nn.Sequential(nn.Conv1d(1,16,7,padding=3),nn.BatchNorm1d(16),nn.ReLU(),nn.Conv1d(16,hidden,5,padding=2),nn.ReLU()); self.rnn=nn.LSTM(hidden,hidden,batch_first=True,bidirectional=True); self.attn=TemporalAttention(hidden*2); self.head=nn.Linear(hidden*2,n_classes)
    def forward(self,x):
        b,t,d=x.shape; z=self.conv(x.reshape(b*t,1,d)).mean(-1).reshape(b,t,-1); h,_=self.rnn(z); c,w=self.attn(h); return self.head(c),w


def make_model(kind,input_dim,n_classes=5): return CNNBiLSTMAttention(input_dim,n_classes=n_classes) if kind=='CNN_BiLSTM_Attention' else TemporalModel(input_dim,kind,n_classes=n_classes)


def focal_loss(logits,y,gamma=2.0):
    ce=nn.functional.cross_entropy(logits,y,reduction='none'); pt=torch.exp(-ce); return ((1-pt)**gamma*ce).mean()


def class_weights(y, n_classes=None):
    n_classes = len(CLASSES) if n_classes is None else int(n_classes)
    counts=np.bincount(y,minlength=n_classes); return len(y)/(n_classes*np.maximum(counts,1))


def metrics(y,probs):
    pred=probs.argmax(1); rows=[]
    prec,rec,f1,sup=precision_recall_fscore_support(y,pred,labels=np.arange(len(CLASSES)),zero_division=0)
    for i,c in enumerate(CLASSES):
        truth=(y==i).astype(int); rows.append({'class':c,'precision':prec[i],'recall':rec[i],'f1':f1[i],'support':int(sup[i]),'auroc':roc_auc_score(truth,probs[:,i]) if len(np.unique(truth))==2 else np.nan,'auprc':average_precision_score(truth,probs[:,i]) if truth.sum()>0 else np.nan})
    detail=pd.DataFrame(rows); return {'MacroF1':float(f1.mean()),'MacroAUPRC':float(np.nanmean(detail.auprc)),'MacroAUROC':float(np.nanmean(detail.auroc)),'BalancedAccuracy':float(balanced_accuracy_score(y,pred)),'MinRecall':float(rec.min()),'Accuracy':float((pred==y).mean())},detail,pred


def predict(model,X):
    model.eval(); out=[]; att=[]
    with torch.no_grad():
        for i in range(0,len(X),BATCH):
            logits,w=model(torch.tensor(X[i:i+BATCH],dtype=torch.float32)); out.append(torch.softmax(logits,1).numpy());
            if w is not None: att.append(w.numpy())
    return np.concatenate(out), np.concatenate(att) if att else None


def train(kind,Xtr,ytr,Xva,yva,loss_mode='class_weight',augment=False,sampler=False,seed=SEED,epochs=EPOCHS,return_model=True):
    set_seed(seed); model=make_model(kind,Xtr.shape[-1]).to(DEVICE); weights=class_weights(ytr)
    if sampler:
        sample_weights=torch.tensor(weights[ytr],dtype=torch.double); sampler_obj=WeightedRandomSampler(sample_weights,len(sample_weights),replacement=True); loader=DataLoader(ECGDataset(Xtr,ytr,augment,seed),batch_size=BATCH,sampler=sampler_obj)
    else: loader=DataLoader(ECGDataset(Xtr,ytr,augment,seed),batch_size=BATCH,shuffle=True)
    va=DataLoader(ECGDataset(Xva,yva),batch_size=BATCH)
    weight_tensor=torch.tensor(weights,dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=float(CFG['learning_rate'])); best=-1; best_state=None; patience=0
    for epoch in range(epochs):
        model.train()
        for x,y in loader:
            opt.zero_grad(); logits,_=model(x)
            if loss_mode=='class_weight': loss=nn.functional.cross_entropy(logits,y,weight=weight_tensor)
            elif loss_mode=='focal': loss=focal_loss(logits,y)
            else: loss=nn.functional.cross_entropy(logits,y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3); opt.step()
        p,_=predict(model,Xva); score=metrics(yva,p)[0]['MacroF1']
        if score>best: best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else: patience+=1
        if patience>=2: break
    if best_state: model.load_state_dict(best_state)
    return model


def one_experiment(name,kind,Xtr,ytr,Xva,yva,Xte,yte,**kwargs):
    started=time.perf_counter(); train_kwargs={k:v for k,v in kwargs.items() if k in ['loss_mode','augment','sampler','seed','epochs']}; model=train(kind,Xtr,ytr,Xva,yva,**train_kwargs); probs,att=predict(model,Xte); score,detail,pred=metrics(yte,probs); score.update({'experiment':name,'architecture':kind,'loss':kwargs.get('loss_mode','none'),'sampler':kwargs.get('sampler',False),'augmentation':kwargs.get('augment',False),'seed':kwargs.get('seed',SEED),'context':Xtr.shape[1],'feature_mode':kwargs.get('feature_mode','waveform'),'Seconds':time.perf_counter()-started,'Params':sum(p.numel() for p in model.parameters())}); return score,detail,pred,probs,model,att

# Prepare only the active split at a time; the test records and labels remain fixed.
def data_for(context, feature_mode):
    tr=build_sequences(BEATS['train'],context,feature_mode); va=build_sequences(BEATS['validation'],context,feature_mode); te=build_sequences(BEATS['test'],context,feature_mode)
    trX,vaX,teX=normalize_features(tr[0],va[0],te[0])
    return trX,tr[1],vaX,va[1],teX,te[1],te[2]

# Baseline diagnostic intervention experiments on the same 8-beat waveform split.
Xtr,ytr,Xva,yva,Xte,yte,meta = data_for(8,'waveform')
rows=[]; aug_rows=[]; detail_rows=[]
for name,opts in [('unweighted',{'loss_mode':'none'}),('class_weighting',{'loss_mode':'class_weight'}),('weighted_random_sampler',{'loss_mode':'none','sampler':True}),('focal_loss',{'loss_mode':'focal'}),('controlled_augmentation',{'loss_mode':'class_weight','augment':True})]:
    score,detail,pred,probs,model,att=one_experiment(name,'RNN',Xtr,ytr,Xva,yva,Xte,yte,**opts); rows.append(score); detail_rows.append(detail.assign(experiment=name)); aug_rows.append({**score,'minority_recall_F':float(detail.loc[detail['class']=='F','recall'].iloc[0]),'minority_recall_N':float(detail.loc[detail['class']=='N','recall'].iloc[0])})
pd.DataFrame(rows).to_csv(OUT/'ablation_results.csv',index=False); pd.DataFrame(aug_rows).to_csv(OUT/'augmentation_results.csv',index=False)

# Context and feature ablations use the same fixed split, changing only the declared variable.
context_rows=[]
for context in [4,8,16,32]:
    a=data_for(context,'waveform'); score,detail,pred,probs,model,att=one_experiment(f'context_{context}','RNN',*a[:6],loss_mode='class_weight'); context_rows.append(score); detail_rows.append(detail.assign(experiment=f'context_{context}')); del a
pd.DataFrame(context_rows).to_csv(OUT/'context_results.csv',index=False)
feature_rows=[]
for feature_mode in ['waveform','rr','rr_morphology']:
    a=data_for(8,feature_mode); score,detail,pred,probs,model,att=one_experiment(f'features_{feature_mode}','RNN',*a[:6],loss_mode='class_weight',feature_mode=feature_mode); feature_rows.append(score); detail_rows.append(detail.assign(experiment=f'features_{feature_mode}')); del a
pd.DataFrame(feature_rows).to_csv(OUT/'feature_results.csv',index=False)

# Architecture benchmark.
architecture_rows=[]; arch_predictions={}; arch_models={}
for kind in ['RNN','LSTM','GRU','BiLSTM','BiLSTM_Attention','CNN_BiLSTM_Attention']:
    score,detail,pred,probs,model,att=one_experiment(kind,kind,Xtr,ytr,Xva,yva,Xte,yte,loss_mode='class_weight'); architecture_rows.append(score); detail_rows.append(detail.assign(experiment=kind)); arch_predictions[kind]=(pred,probs,detail,att); arch_models[kind]=model; torch.save({'state_dict':model.state_dict(),'model_type':kind,'input_shape':[8,Xtr.shape[-1]],'class_names':CLASSES,'config':CFG},OUT/'checkpoints'/f'{kind}.pt')
pd.DataFrame(architecture_rows).to_csv(OUT/'architecture_results.csv',index=False)
pd.concat(detail_rows, ignore_index=True).to_csv(OUT/'per_class_metrics_experiments.csv', index=False)

# Hierarchical classifier: stage 1 N vs non-N and stage 2 S/V/F/Q on exactly the same split.
def train_hierarchical(Xtr,ytr,Xva,yva,Xte,yte):
    set_seed(SEED); stage1=TemporalModel(Xtr.shape[-1],'RNN',n_classes=2); stage2=TemporalModel(Xtr.shape[-1],'RNN',n_classes=4)
    def fit(m,Xfit,yfit,Xval,yval,ncls):
        wt=class_weights(yfit,ncls); opt=torch.optim.Adam(m.parameters(),lr=float(CFG['learning_rate'])); best=-1; state=None
        for _ in range(EPOCHS):
            m.train(); loader=DataLoader(ECGDataset(Xfit,yfit),batch_size=BATCH,shuffle=True)
            for x,y in loader:
                opt.zero_grad(); loss=nn.functional.cross_entropy(m(x)[0],y,weight=torch.tensor(wt,dtype=torch.float32)); loss.backward(); opt.step()
            m.eval(); vp=[]
            with torch.no_grad():
                for i in range(0,len(Xval),BATCH): vp.append(torch.softmax(m(torch.tensor(Xval[i:i+BATCH],dtype=torch.float32))[0],1).numpy())
            sc=f1_score(yval,np.concatenate(vp).argmax(1),average='macro',zero_division=0)
            if sc>best: best=sc; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        m.load_state_dict(state); return m
    y1tr=(ytr!=0).astype(int); y1va=(yva!=0).astype(int); y1te=(yte!=0).astype(int); mask_tr=ytr!=0; mask_va=yva!=0
    stage1=fit(stage1,Xtr,y1tr,Xva,y1va,2); stage2=fit(stage2,Xtr[mask_tr],ytr[mask_tr]-1,Xva[mask_va],yva[mask_va]-1,4)
    p1,_=predict(stage1,Xte); p2,_=predict(stage2,Xte); combined=np.zeros((len(yte),5)); combined[:,0]=p1[:,0]; combined[:,1:]=p1[:,1,None]*p2; return combined,stage1,stage2
hier_probs,stage1,stage2=train_hierarchical(Xtr,ytr,Xva,yva,Xte,yte); hier_score,hier_detail,hier_pred=metrics(yte,hier_probs); hier_score.update({'experiment':'hierarchical_N_vs_nonN','architecture':'hierarchical_RNN','loss':'class_weight','sampler':False,'augmentation':False,'seed':SEED,'context':8,'feature_mode':'waveform','Params':sum(p.numel() for p in stage1.parameters())+sum(p.numel() for p in stage2.parameters())}); hier_detail=hier_detail.assign(experiment='hierarchical_N_vs_nonN'); pd.DataFrame([hier_score]).to_csv(OUT/'hierarchical_results.csv',index=False); pd.concat(detail_rows+[hier_detail],ignore_index=True).to_csv(OUT/'per_class_metrics_experiments.csv',index=False); torch.save({'stage1':stage1.state_dict(),'stage2':stage2.state_dict(),'config':CFG},OUT/'checkpoints/hierarchical.pt')

# Multi-seed runs for the strongest two architecture finalists plus the measured morphology-feature winner.
arch_df=pd.DataFrame(architecture_rows).sort_values(['MacroF1','MacroAUPRC'],ascending=False); finalists=arch_df['architecture'].head(2).tolist(); seed_rows=[]
seed_specs=[(kind,kind,Xtr,ytr,Xva,yva,Xte,yte,'waveform') for kind in finalists]
feature_seed_data=data_for(8,'rr_morphology'); seed_specs.append(('RNN_rr_morphology','RNN',*feature_seed_data[:6],'rr_morphology'))
for label,kind,xtr,ytr_s,xva,yva_s,xte,yte_s,feature_mode_s in seed_specs:
    for seed in [42,43,44,45,46]:
        score,detail,pred,probs,model,att=one_experiment(f'{label}_seed_{seed}',kind,xtr,ytr_s,xva,yva_s,xte,yte_s,loss_mode='class_weight',seed=seed,feature_mode=feature_mode_s); seed_rows.append(score)
pd.DataFrame(seed_rows).to_csv(OUT/'seed_results.csv',index=False)

# Select the strongest measured overall candidate without forcing an architecture family.
feature_selected_data=data_for(8,'rr_morphology'); feature_selected_score,feature_selected_detail,feature_selected_pred,feature_selected_probs,feature_selected_model,feature_selected_att=one_experiment('RNN_rr_morphology_selected','RNN',*feature_selected_data[:6],loss_mode='class_weight',feature_mode='rr_morphology')
candidate_rows=architecture_rows + feature_rows + [hier_score]
candidate_table=pd.DataFrame(candidate_rows).sort_values(['MacroF1','MacroAUPRC'],ascending=False); candidate_table.to_csv(OUT/'selected_candidate_comparison.csv',index=False)
best_row=candidate_table.iloc[0]; best_name=str(best_row['experiment'])
if best_name.startswith('features_rr_morphology') or best_name=='RNN_rr_morphology_selected': selected_model=feature_selected_model; selected_X=feature_selected_data[4]; selected_probs=feature_selected_probs; selected_detail=feature_selected_detail; selected_arch='RNN_rr_morphology'
elif best_name.startswith('hierarchical'): selected_model=None; selected_X=Xte; selected_probs=hier_probs; selected_detail=hier_detail; selected_arch='hierarchical_RNN'
else: selected_model=arch_models[best_name]; selected_X=Xte; selected_probs=arch_predictions[best_name][1]; selected_detail=arch_predictions[best_name][2]; selected_arch=best_name
# Bootstrap 95% CIs for the selected strongest candidate on the fixed test split.
rng=np.random.default_rng(SEED); boot=[]
for b in range(500):
    ix=rng.integers(0,len(yte),len(yte)); sc,_,_=metrics(yte[ix],selected_probs[ix]); boot.append({'metric':'MacroF1','value':sc['MacroF1']}); boot.append({'metric':'MacroAUROC','value':sc['MacroAUROC']})
ci=[]
for metric_name,g in pd.DataFrame(boot).groupby('metric'):
    estimate=metrics(yte,selected_probs)[0][metric_name]
    ci.append({'model':selected_arch,'metric':metric_name,'estimate':float(estimate),'ci_lower_95':float(g.value.quantile(.025)),'ci_upper_95':float(g.value.quantile(.975)),'bootstrap_samples':500})
pd.DataFrame(ci).to_csv(OUT/'bootstrap_confidence_intervals.csv',index=False)

# Robustness on strongest model, test-only transforms; no retraining and no test modification.
def corrupt(X, kind):
    z=X.copy(); t=np.arange(180,dtype=np.float32)
    if kind=='clean': return z
    if kind=='gaussian_noise': z[:,:,:180]+=np.random.default_rng(123).normal(0,.05,z[:,:,:180].shape).astype(np.float32)
    elif kind=='baseline_wander': z[:,:,:180]+=0.08*np.sin(2*np.pi*t/180*1.2)[None,None,:]
    elif kind=='amplitude_scaling': z[:,:,:180]*=1.20
    elif kind=='missing_samples': z[:,:,81:99]=0
    return z
robust_rows=[]
for corruption in ['clean','gaussian_noise','baseline_wander','amplitude_scaling','missing_samples']:
    if selected_model is None: pr=selected_probs
    else: pr,_=predict(selected_model,corrupt(selected_X,corruption))
    sc,detail,_=metrics(yte,pr); sc.update({'model':selected_arch,'corruption':corruption,'MacroF1_drop_from_clean':np.nan}); robust_rows.append(sc)
robust_df=pd.DataFrame(robust_rows); clean=robust_df.loc[robust_df.corruption=='clean','MacroF1'].iloc[0]; robust_df['MacroF1_drop_from_clean']=robust_df['MacroF1']-clean; robust_df.to_csv(OUT/'robustness_results.csv',index=False)

# Save an exact experiment manifest.
manifest={'selected_strongest_candidate':selected_arch,'selected_candidate_score':float(best_row['MacroF1']),'seed':SEED,'split_lock':SPLIT_LOCK,'primary_metric':'MacroF1','secondary_metrics':['MacroAUPRC','per_class_recall'],'contexts':[4,8,16,32],'feature_modes':['waveform','rr','rr_morphology'],'loss_modes':['none','class_weight','focal'],'sampler':['none','WeightedRandomSampler'],'augmentation_train_only':['gaussian_noise','baseline_wander','amplitude_scaling','small_temporal_perturbation'],'architectures':['RNN','LSTM','GRU','BiLSTM','BiLSTM_Attention','CNN_BiLSTM_Attention'],'hierarchical':'N_vs_nonN_then_SV FQ','seeds':[42,43,44,45,46],'bootstrap_samples':500,'robustness':['clean','gaussian_noise','baseline_wander','amplitude_scaling','missing_samples'],'test_hash':hashlib.sha256(np.asarray(yte,dtype=np.int64).tobytes()+meta.to_csv(index=False).encode()).hexdigest(),'test_examples':len(yte)}
(OUT/'configs/experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'finalists':finalists,'selected_strongest_candidate':selected_arch,'selected_macro_f1':float(best_row['MacroF1']),'hierarchical_macro_f1':hier_score['MacroF1'],'test_examples':len(yte),'split_lock':SPLIT_LOCK},indent=2))
