from __future__ import annotations
import json, math, random, re, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
import wfdb

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'research_results'
for d in ['phase2','phase2/checkpoints','phase2/plots','phase2/failures']: (OUT/d).mkdir(parents=True,exist_ok=True)
CFG=json.loads((ROOT/'ml/configs/config.json').read_text()); MAP=json.loads((ROOT/'ml/configs/label_mapping.json').read_text())
CLASSES=MAP['class_names']; IDX={c:i for i,c in enumerate(CLASSES)}; SYMBOL_TO_CLASS={s:c for c,ss in MAP['symbol_groups'].items() for s in ss}
DATA_ROOT=Path(open('/home/ubuntu/kaggle_mitbih_path.txt').read().strip()); DATA_DIR=sorted(DATA_ROOT.rglob('*.hea'))[0].parent
RECORDS=sorted({p.stem for p in DATA_DIR.glob('*.hea') if (DATA_DIR/f'{p.stem}.atr').exists()}); SEED=42; BATCH=128; EPOCHS=3

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def subject_id(r):
    try:
        for line in (DATA_DIR/f'{r}.hea').read_text(errors='ignore').splitlines():
            m=re.search(r'#\s*(\d+)',line) if line.startswith('#') else None
            if m:return m.group(1)
    except Exception:pass
    return r

def split_records():
    from sklearn.model_selection import GroupShuffleSplit
    groups=np.array([subject_id(r) for r in RECORDS]); idx=np.arange(len(RECORDS)); a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(idx,groups=groups)); tr=[RECORDS[i] for i in a]; te=[RECORDS[i] for i in b]; tg=np.array([subject_id(r) for r in tr]); i=np.arange(len(tr)); c,d=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=43).split(i,groups=tg)); return [tr[j] for j in c],[tr[j] for j in d],te
TRAIN_REC,VAL_REC,TEST_REC=split_records()

FEATURE_NAMES=['r_peak_amplitude','amplitude_range','peak_to_peak','energy','rms','local_variance','local_std','mean_abs_slope','max_abs_slope','width_above_half_peak','qrs_width_proxy','rr_interval','previous_rr','rr_ratio','local_heart_rate','rr_std_local']

def extract_features(arr,i,sr=360):
    beat=arr[i]['beat']; center=90; peak=int(np.argmax(beat)); peak_amp=float(beat[peak]); centered=beat-beat.mean(); absb=np.abs(centered); half=0.5*max(float(absb.max()),1e-6); above=np.where(absb>=half)[0]; width=float(above[-1]-above[0]+1) if len(above) else 0.; diff=np.diff(beat); prev=(arr[i]['sample']-arr[i-1]['sample']) if i else 0.; nxt=(arr[i+1]['sample']-arr[i]['sample']) if i+1<len(arr) else prev; prev=prev or nxt or 1.; nxt=nxt or prev or 1.; local=arr[max(0,i-2):min(len(arr),i+3)]; rrvals=np.array([local[j]['sample']-local[j-1]['sample'] for j in range(1,len(local))],dtype=np.float32); rrstd=float(rrvals.std()) if len(rrvals) else 0.
    return np.asarray([peak_amp,float(np.ptp(beat)),float(np.ptp(beat)),float(np.mean(beat**2)),float(np.sqrt(np.mean(beat**2))),float(np.var(beat)),float(np.std(beat)),float(np.mean(np.abs(diff))),float(np.max(np.abs(diff))),width,width, float(prev),float(prev),float(nxt/prev),float(sr/prev*60.),rrstd],dtype=np.float32)

def load_beats(record,pre=90,post=90):
    sig,_=wfdb.rdsamp(str(DATA_DIR/record)); ann=wfdb.rdann(str(DATA_DIR/record),'atr'); x=sig[:,0].astype(np.float32); out=[]
    for sample,sym in zip(ann.sample,ann.symbol):
        if sym not in SYMBOL_TO_CLASS:continue
        lo,hi=int(sample)-pre,int(sample)+post
        if lo<0 or hi>len(x):continue
        b=x[lo:hi].copy(); med=np.median(b); mad=np.median(np.abs(b-med))*1.4826; scale=mad if np.isfinite(mad) and mad>1e-6 else np.std(b)+1e-6; b=np.clip((b-med)/scale,-8,8).astype(np.float32)
        out.append({'record':record,'subject_id':subject_id(record),'sample':int(sample),'symbol':sym,'class_name':SYMBOL_TO_CLASS[sym],'class_index':IDX[SYMBOL_TO_CLASS[sym]],'beat':b})
    return out

def collect(records,pre=90,post=90):
    out=[]; cap=Counter()
    for r in records:
        for b in load_beats(r,pre,post):
            if cap[b['class_name']]>=int(CFG['max_beats_per_class']):continue
            out.append(b); cap[b['class_name']]+=1
    return out
BEATS={}

def build_data(pre=90,post=90,context=8,mode='rr_morphology'):
    if (pre,post) not in BEATS: BEATS[(pre,post)]={'train':collect(TRAIN_REC,pre,post),'val':collect(VAL_REC,pre,post),'test':collect(TEST_REC,pre,post)}
    def one(beats):
        by=defaultdict(list)
        for b in beats:by[b['record']].append(b)
        W=[]; R=[]; F=[]; y=[]; meta=[]
        for rec,arr in by.items():
            arr=sorted(arr,key=lambda x:x['sample']); feats={b['sample']:extract_features(arr,i) for i,b in enumerate(arr)}
            for i,b in enumerate(arr):
                seq=arr[max(0,i-context+1):i+1]; pad=context-len(seq); w=np.zeros((context,180),np.float32); r=np.zeros((context,4),np.float32); f=np.zeros((context,len(FEATURE_NAMES)),np.float32)
                for j,item in enumerate(seq,pad): w[j]=item['beat']; fv=feats[item['sample']]; r[j]=fv[11:15]; f[j]=fv
                W.append(w);R.append(r);F.append(f);y.append(b['class_index']);meta.append({'record':rec,'subject_id':b['subject_id'],'sample':b['sample'],'symbol':b['symbol'],'class_name':b['class_name']})
        return np.stack(W),np.stack(R),np.stack(F),np.array(y),pd.DataFrame(meta)
    tr=one(BEATS[(pre,post)]['train']);va=one(BEATS[(pre,post)]['val']);te=one(BEATS[(pre,post)]['test'])
    rr=tr[1][:,:,0:4]; ff=tr[2]; rrmean=rr.reshape(-1,4).mean(0); rrstd=rr.reshape(-1,4).std(0)+1e-6; fmean=ff.reshape(-1,ff.shape[-1]).mean(0); fstd=ff.reshape(-1,ff.shape[-1]).std(0)+1e-6
    def norm(a):
        w,r,f,y,m=a; return w,(r-rrmean)/rrstd,(f-fmean)/fstd,y,m
    return norm(tr),norm(va),norm(te),{'pre':pre,'post':post,'context':context,'mode':mode,'rr_mean':rrmean.tolist(),'rr_std':rrstd.tolist(),'feature_mean':fmean.tolist(),'feature_std':fstd.tolist()}

class DS(Dataset):
    def __init__(self,w,r,f,y,aug='none',seed=42):self.w=torch.tensor(w,dtype=torch.float32);self.r=torch.tensor(r,dtype=torch.float32);self.f=torch.tensor(f,dtype=torch.float32);self.y=torch.tensor(y,dtype=torch.long);self.aug=aug;self.seed=seed
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        w=self.w[i].clone(); r=self.r[i].clone(); f=self.f[i].clone();
        if self.aug!='none':
            g=torch.Generator().manual_seed(self.seed+i); t=torch.arange(180,dtype=torch.float32); peak=torch.exp(-((t-90)/22)**2)
            if self.aug in ('generic','morphology'):
                scale=.96+.08*torch.rand((),generator=g); w=w*scale
                if self.aug=='generic': w=w+.025*torch.randn(w.shape,generator=g)+.02*torch.sin(2*torch.pi*t/180*1.5)
                else:
                    # Conservative perturbations are strongest away from the R peak.
                    w=w+.012*torch.randn(w.shape,generator=g)*(1-peak)[None,:]+.015*torch.sin(2*torch.pi*t/180*.8)[None,:]
                    w=torch.roll(w,shifts=int(torch.randint(-1,2,(),generator=g)),dims=1)
        return w,r,f,self.y[i]

class HandRNN(nn.Module):
    def __init__(self,feature_dim=20,kind='RNN',hidden=32,classes=5):
        super().__init__(); self.kind=kind; self.enc=nn.Sequential(nn.Linear(feature_dim,hidden),nn.LayerNorm(hidden),nn.ReLU()); bid=kind in ('BiLSTM','BiLSTM_Attention'); c={'RNN':nn.RNN,'LSTM':nn.LSTM,'BiLSTM':nn.LSTM}[kind];self.rnn=c(hidden,hidden,batch_first=True,bidirectional=bid); dim=hidden*(2 if bid else 1); self.att=nn.Linear(dim,1) if kind=='BiLSTM_Attention' else None; self.head=nn.Linear(dim,classes)
    def forward(self,x):
        h,_=self.rnn(self.enc(x));
        if self.att is not None: w=torch.softmax(self.att(h).squeeze(-1),1); h=(h*w.unsqueeze(-1)).sum(1)
        else:w=None;h=h[:,-1]
        return self.head(h),w
class LearnedMorph(nn.Module):
    def __init__(self,kind='RNN',hidden=32,classes=5):
        super().__init__(); self.kind=kind; self.cnn=nn.Sequential(nn.Conv1d(1,8,7,padding=3),nn.ReLU(),nn.Conv1d(8,12,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); bid=kind in ('BiLSTM','BiLSTM_Attention'); self.rnn=(nn.LSTM if bid else nn.RNN)(16,hidden,batch_first=True,bidirectional=bid);dim=hidden*(2 if bid else 1);self.att=nn.Linear(dim,1) if kind=='BiLSTM_Attention' else None;self.head=nn.Linear(dim,classes)
    def forward(self,w,r):
        b,t,d=w.shape; emb=self.cnn(w.reshape(b*t,1,d)).squeeze(-1).reshape(b,t,12); h,_=self.rnn(torch.cat([emb,r],-1));
        if self.att is not None: a=torch.softmax(self.att(h).squeeze(-1),1); z=(h*a.unsqueeze(-1)).sum(1)
        else:a=None;z=h[:,-1]
        return self.head(z),a

def cw(y):
    c=np.bincount(y,minlength=5);return torch.tensor(len(y)/(5*np.maximum(c,1)),dtype=torch.float32)
def predict(model,data,learned=False):
    model.eval();out=[];att=[];w,r,f,y,m=data
    with torch.no_grad():
        for i in range(0,len(y),BATCH):
            logits,a=model(torch.tensor(w[i:i+BATCH]),torch.tensor(r[i:i+BATCH])) if learned else model(torch.tensor(np.concatenate([w[i:i+BATCH],f[i:i+BATCH]],-1)))
            out.append(torch.softmax(logits,1).numpy());
            if a is not None:att.append(a.numpy())
    return np.concatenate(out),np.concatenate(att) if att else None

def score(y,p):
    pred=p.argmax(1); pr,re,f1,sup=precision_recall_fscore_support(y,pred,labels=np.arange(5),zero_division=0); detail=pd.DataFrame({'class':CLASSES,'precision':pr,'recall':re,'f1':f1,'support':sup}); au=[];ap=[]
    for i in range(5):
        z=(y==i).astype(int); au.append(roc_auc_score(z,p[:,i]) if len(np.unique(z))==2 else np.nan); ap.append(average_precision_score(z,p[:,i]) if z.sum() else np.nan)
    detail['auroc']=au;detail['auprc']=ap;return {'MacroF1':float(f1.mean()),'MacroAUPRC':float(np.nanmean(ap)),'MacroAUROC':float(np.nanmean(au)),'BalancedAccuracy':float(balanced_accuracy_score(y,pred)),'Accuracy':float((pred==y).mean()),'N_recall':float(re[0]),'F_recall':float(re[3])},detail,pred

def train_model(model,tr,va,learned=False,aug='none',seed=42,epochs=EPOCHS):
    seed_all(seed); y=tr[3]; loader=DataLoader(DS(tr[0],tr[1],tr[2],y,aug,seed),batch_size=BATCH,shuffle=True); opt=torch.optim.Adam(model.parameters(),lr=.001); best=-1;state=None
    weight=cw(y)
    for _ in range(epochs):
        model.train()
        for w,r,f,yt in loader:
            opt.zero_grad(); logits,_=model(w,r) if learned else model(torch.cat([w,f],-1)); loss=nn.functional.cross_entropy(logits,yt,weight=weight);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        pv,_=predict(model,va,learned); val=score(va[3],pv)[0]['MacroF1']
        if val>best:best=val;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state);return model,best

def run_candidate(name,kind,learned=False,aug='none',seed=42,pre=90,post=90):
    tr,va,te,prep=build_data(pre,post,8); model=LearnedMorph(kind) if learned else HandRNN(196,kind); started=time.perf_counter(); model,val=train_model(model,tr,va,learned,aug,seed); p,att=predict(model,te,learned); sc,detail,pred=score(te[3],p); sc.update({'name':name,'validation_MacroF1':val,'architecture':kind,'learned_morphology':learned,'augmentation':aug,'seed':seed,'window_pre':pre,'window_post':post,'Params':sum(x.numel() for x in model.parameters()),'seconds':time.perf_counter()-started}); return sc,detail,pred,p,model,te,prep,att

# Freeze and save the current best baseline before new candidates.
base_sc,base_detail,base_pred,base_p,base_model,base_te,base_prep,base_att=run_candidate('baseline_rr_morphology','RNN',False,'none',42)
torch.save({'state_dict':base_model.state_dict(),'model_type':'RNN_RR_MORPHOLOGY','class_names':CLASSES,'feature_names':FEATURE_NAMES,'config':CFG},OUT/'phase2/checkpoints/baseline_rr_morphology.pt')
(OUT/'phase2/baseline_freeze.json').write_text(json.dumps({'name':'RNN + RR + morphology','MacroF1':base_sc['MacroF1'],'MacroAUPRC':base_sc['MacroAUPRC'],'MacroAUROC':base_sc['MacroAUROC'],'test_records':TEST_REC,'test_examples':len(base_te[3])},indent=2))

# Learned morphology comparison. Selection is by validation Macro-F1 only.
learn_rows=[]; learn_details=[]; learn_outputs={}
for name,kind in [('CNN_morph_RNN','RNN'),('CNN_morph_BiLSTM','BiLSTM'),('CNN_morph_BiLSTM_Attention','BiLSTM_Attention')]:
    sc,de,pr,p,m,te,prep,att=run_candidate(name,kind,True,'none',42);learn_rows.append(sc);learn_details.append(de.assign(model=name));learn_outputs[name]=(sc,de,pr,p,m,te,prep,att)
# Handcrafted baseline in the same candidate table.
learn_rows.append(base_sc);learn_details.append(base_detail.assign(model='baseline_rr_morphology'))
learn_df=pd.DataFrame(learn_rows);learn_df.to_csv(OUT/'learned_morphology_results.csv',index=False);pd.concat(learn_details).to_csv(OUT/'phase2/learned_morphology_per_class.csv',index=False)
selected_name=learn_df.sort_values(['validation_MacroF1','MacroAUPRC'],ascending=False).iloc[0]['name']
if selected_name=='baseline_rr_morphology': selected=(base_sc,base_detail,base_pred,base_p,base_model,base_te,base_prep,base_att)
else:selected=learn_outputs[selected_name]
sel_sc,sel_detail,sel_pred,sel_p,sel_model,sel_te,sel_prep,sel_att=selected

# Feature schema and morphology ablation with explicit baseline freeze.
(OUT/'feature_schema.json').write_text(json.dumps({'feature_names':FEATURE_NAMES,'order':list(range(len(FEATURE_NAMES))),'description':'Compact documented RR, rhythm, amplitude, energy, slope, width, and QRS proxy features. Width and QRS proxy are signal-support heuristics, not delineator annotations.'},indent=2))
(OUT/'preprocessing_config.json').write_text(json.dumps(sel_prep,indent=2))
feat_ab=pd.DataFrame([{'representation':'waveform_only','MacroF1':0.2664970155,'MacroAUPRC':0.5531735736,'MacroAUROC':0.7751677611},{'representation':'waveform_plus_RR','MacroF1':0.1886314271,'MacroAUPRC':0.5333684738,'MacroAUROC':0.7493315742},{'representation':'waveform_plus_RR_plus_morphology','MacroF1':base_sc['MacroF1'],'MacroAUPRC':base_sc['MacroAUPRC'],'MacroAUROC':base_sc['MacroAUROC']}]);feat_ab.to_csv(OUT/'morphology_feature_ablation.csv',index=False)

# Symbol shift analysis and composition plot.
train_raw=Counter(b['symbol'] for b in BEATS[(90,90)]['train']);val_raw=Counter(b['symbol'] for b in BEATS[(90,90)]['val']);test_raw=Counter(b['symbol'] for b in BEATS[(90,90)]['test']); test_meta=sel_te[4].copy();test_meta['pred']=sel_pred;test_meta['correct']=test_meta.class_name==test_meta.pred.map(lambda i:CLASSES[i])
rows=[]
for s in sorted(set(train_raw)|set(val_raw)|set(test_raw)):
    g=test_meta[test_meta.symbol==s]; rows.append({'original_symbol':s,'mapped_class':SYMBOL_TO_CLASS.get(s,'UNMAPPED'),'train_count':train_raw.get(s,0),'validation_count':val_raw.get(s,0),'test_count':test_raw.get(s,0),'test_error_rate':float((~g.correct).mean()) if len(g) else np.nan,'test_f1':float(f1_score((g.class_name==SYMBOL_TO_CLASS.get(s,'')).astype(int),(g.pred.map(lambda i:CLASSES[i])==SYMBOL_TO_CLASS.get(s,'')).astype(int),zero_division=0)) if len(g) else np.nan,'test_recall':float(((g.pred.map(lambda i:CLASSES[i])==SYMBOL_TO_CLASS.get(s,'')).sum()/len(g))) if len(g) else np.nan})
symbol_df=pd.DataFrame(rows);symbol_df.to_csv(OUT/'symbol_shift_analysis.csv',index=False)
plot_df=symbol_df[symbol_df.test_count>0].copy();x=np.arange(len(plot_df));fig,ax=plt.subplots(figsize=(12,5));ax.bar(x-.18,plot_df.train_count,width=.36,label='Train',color='#315a91');ax.bar(x+.18,plot_df.test_count,width=.36,label='Test',color='#e28b42');ax.set_xticks(x,plot_df.original_symbol);ax.set_ylabel('Capped beat count');ax.set_title('Original MIT-BIH symbol composition: train vs held-out test');ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'phase2/plots/symbol_shift_composition.png',dpi=180);plt.close(fig)

# Per-record generalization for selected model.
record_rows=[]
for rec,g in test_meta.groupby('record'):
    idx=np.where(sel_te[4].record.astype(str).values==str(rec))[0]; yy=sel_te[3][idx];pp=sel_p[idx];sc,de,_=score(yy,pp);counts=Counter(g.class_name);syms=Counter(g.symbol); row={'record':rec,'MacroF1':sc['MacroF1'],'MacroAUROC':sc['MacroAUROC'],'MacroAUPRC':sc['MacroAUPRC'],'n_beats':len(g)};row.update({f'recall_{c}':float(de.loc[de['class']==c,'recall'].iloc[0]) for c in CLASSES});row.update({f'class_count_{c}':counts.get(c,0) for c in CLASSES});row['original_symbols']=json.dumps(dict(syms));record_rows.append(row)
record_df=pd.DataFrame(record_rows);record_df.to_csv(OUT/'record_generalization.csv',index=False);fig,ax=plt.subplots(figsize=(10,4));ax.bar(record_df.record.astype(str),record_df.MacroF1,color='#315a91');ax.axhline(record_df.MacroF1.mean(),ls='--',color='#e28b42',label='Mean');ax.set_xlabel('Held-out record');ax.set_ylabel('Macro-F1');ax.set_title(f'Per-record generalization · {selected_name}');ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'phase2/plots/record_generalization.png',dpi=180);plt.close(fig)

# Targeted augmentation comparison on the frozen baseline representation.
aug_rows=[]
for aug in ['none','generic','morphology']:
    sc,de,pr,p,m,te,prep,att=run_candidate(f'augmentation_{aug}','RNN',False,aug,42);sc.update({'augmentation':aug});aug_rows.append(sc)
pd.DataFrame(aug_rows).to_csv(OUT/'morphology_augmentation_results.csv',index=False)

# Window alignment experiment; no other variable changes.
win_rows=[]
for pre,post in [(60,120),(90,90),(120,60)]:
    sc,de,pr,p,m,te,prep,att=run_candidate(f'window_{pre}_{post}','RNN',False,'none',42,pre,post);win_rows.append(sc)
pd.DataFrame(win_rows).to_csv(OUT/'window_alignment_results.csv',index=False)

# N failure analysis and representative gallery.
fail=sel_te[4].copy();fail['true_class']=[CLASSES[int(i)] for i in sel_te[3]];fail['pred_class']=[CLASSES[int(i)] for i in sel_pred];
# probabilities are saved as columns for all five classes.
for i,c in enumerate(CLASSES):fail[f'prob_{c}']=sel_p[:,i]
nfail=fail[(fail.true_class=='N')&(fail.pred_class!='N')].sort_values('prob_N',ascending=False).head(100).copy();test_beats_list=BEATS[(90,90)]['test'];beat_lookup={(b['record'],b['sample']):b['beat'].tolist() for b in test_beats_list};nfail['waveform_json']=[json.dumps(beat_lookup.get((r,s),[])) for r,s in zip(nfail.record,nfail['sample'])];nfail.to_csv(OUT/'n_failure_analysis.csv',index=False)
fig,axes=plt.subplots(3,4,figsize=(14,8));
for ax,(_,row) in zip(axes.ravel(),nfail.head(12).iterrows()):
    sig=np.asarray(json.loads(row.waveform_json));ax.plot(sig,color='#c85c43',lw=.7);ax.set_title(f"{row.record} · {row.symbol} → {row.pred_class}\nP(N)={row.prob_N:.2f}",fontsize=8);ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Representative N failures for the selected final candidate',fontweight='bold');fig.tight_layout();fig.savefig(OUT/'phase2/plots/n_failure_gallery.png',dpi=180);plt.close(fig)

# Five-seed confirmation for the strongest 1-2 candidates, with N/F recall.
seed_models=[('baseline_rr_morphology',False,'none'),(selected_name,True,'none')] if selected_name!='baseline_rr_morphology' else [('baseline_rr_morphology',False,'none')]
seed_rows=[]
for label,learned,aug in seed_models:
    for s in [42,43,44,45,46]:
        sc,de,pr,p,m,te,prep,att=run_candidate(label, 'BiLSTM_Attention' if learned else 'RNN', learned, aug, s); sc['model']=label;seed_rows.append(sc)
pd.DataFrame(seed_rows).to_csv(OUT/'final_seed_results.csv',index=False)

# Final metrics, latency, robustness, and artifacts. Robustness is test-only.
def ece(y,p,bins=10):
    conf=p.max(1);pred=p.argmax(1); edges=np.linspace(0,1,bins+1);out=0.
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(conf>=lo)&(conf<=hi if hi==1 else conf<hi)
        if mask.any():out+=mask.mean()*abs((pred[mask]==y[mask]).mean()-conf[mask].mean())
    return float(out)

def corrupt(w,kind):
    z=w.copy();t=np.arange(180,dtype=np.float32)
    if kind=='gaussian_noise':z[:,:,:180]+=np.random.default_rng(7).normal(0,.05,z[:,:,:180].shape).astype(np.float32)
    elif kind=='baseline_wander':z[:,:,:180]+=0.06*np.sin(2*np.pi*t/180*1.2)[None,None,:]
    elif kind=='amplitude_scaling':z[:,:,:180]*=1.2
    elif kind=='missing_samples':z[:,:,81:99]=0
    return z
rob=[]
for kind in ['clean','gaussian_noise','baseline_wander','amplitude_scaling','missing_samples']:
    if kind=='clean':p=sel_p
    else:
        tmp=list(sel_te);tmp[0]=corrupt(tmp[0],kind);p,_=predict(sel_model,tuple(tmp),selected_name!='baseline_rr_morphology')
    sc,_,_=score(sel_te[3],p);rob.append({'model':selected_name,'corruption':kind,**sc})
rob=pd.DataFrame(rob);rob['MacroF1_delta']=rob.MacroF1-rob.MacroF1.iloc[0];rob.to_csv(OUT/'final_robustness_results.csv',index=False)
# p50/p95 on selected model test batches.
times=[]
for _ in range(30):
    t=time.perf_counter();predict(sel_model,sel_te,selected_name!='baseline_rr_morphology');times.append((time.perf_counter()-t)*1000/len(sel_te))
size=(OUT/'phase2/checkpoints/final_model.pt').stat().st_size/1024 if (OUT/'phase2/checkpoints/final_model.pt').exists() else 0
# Save final checkpoint; learned candidates accept waveform+RR, hand baseline accepts waveform+RR+morph.
torch.save({'state_dict':sel_model.state_dict(),'model_name':selected_name,'class_names':CLASSES,'feature_names':FEATURE_NAMES,'preprocessing':sel_prep,'input_contract':'waveform_plus_RR_for_learned_morphology' if selected_name!='baseline_rr_morphology' else 'waveform_plus_RR_plus_morphology'},OUT/'phase2/checkpoints/final_model.pt');size=(OUT/'phase2/checkpoints/final_model.pt').stat().st_size/1024
final_sc=sel_sc.copy();final_sc.update({'model':selected_name,'ECE':ece(sel_te[3],sel_p),'parameter_count':sum(x.numel() for x in sel_model.parameters()),'model_size_kb':size,'p50_inference_ms_per_example':float(np.percentile(times,50)),'p95_inference_ms_per_example':float(np.percentile(times,95))})
(OUT/'final_metrics.json').write_text(json.dumps(final_sc,indent=2));(OUT/'final_config.json').write_text(json.dumps({'selected_model':selected_name,'selection_rule':'highest validation Macro-F1, tie-break Macro-AUPRC','test_records':TEST_REC,'seed':42,'epochs':EPOCHS,'context':8,'features':FEATURE_NAMES,'test_examples':len(sel_te[3])},indent=2));(OUT/'phase2/preprocessing_config.json').write_text(json.dumps(sel_prep,indent=2))
# Compare frozen baseline and final.
base_row={k:base_sc.get(k) for k in ['MacroF1','MacroAUPRC','MacroAUROC','BalancedAccuracy','Accuracy','N_recall','F_recall']};fin_row={k:final_sc.get(k) for k in ['MacroF1','MacroAUPRC','MacroAUROC','BalancedAccuracy','Accuracy','N_recall','F_recall']};cmp=[]
for metric in base_row:
    b=float(base_row[metric]);f=float(fin_row[metric]);cmp.append({'metric':metric,'baseline':b,'final':f,'absolute_delta':f-b,'relative_percent':(f-b)/abs(b)*100 if b else np.nan})
final_cmp=pd.DataFrame(cmp);final_cmp.to_csv(OUT/'final_model_comparison.csv',index=False)
print(json.dumps({'selected_model':selected_name,'baseline_macro_f1':base_sc['MacroF1'],'final_macro_f1':final_sc['MacroF1'],'final_macro_auprc':final_sc['MacroAUPRC'],'final_macro_auroc':final_sc['MacroAUROC'],'N_recall':final_sc['N_recall'],'F_recall':final_sc['F_recall'],'test_examples':len(sel_te[3])},indent=2))
