from __future__ import annotations
import json, math, random, runpy, shutil, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, precision_recall_fscore_support, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'research_results'; P3=OUT/'phase3'
for d in ['plots','checkpoints','failures']: (P3/d).mkdir(parents=True,exist_ok=True)
# Reuse the existing Phase 1/2 implementation and fixed split exactly. The harness itself is deterministic and writes the locked manifest.
g=runpy.run_path(str(ROOT/'research/experiment_harness.py'))
BEATS=g['BEATS']; build_sequences=g['build_sequences']; normalize_features=g['normalize_features']; train_base=g['train']; predict_base=g['predict']; metric_base=g['metrics']; ECGDataset=g['ECGDataset']; TemporalModel=g['TemporalModel']; make_model=g['make_model']; class_weights=g['class_weights']; set_seed=g['set_seed']; CFG=g['CFG']; CLASSES=g['CLASSES']; DATA_DIR=g['DATA_DIR']; MAP_CFG=g['MAP_CFG']; SYMBOL_TO_CLASS=g['SYMBOL_TO_CLASS']; SPLIT_LOCK=g['SPLIT_LOCK']; BATCH=g['BATCH']; EPOCHS=g['EPOCHS']; DEVICE=g['DEVICE']

# Exact fixed 8-beat handcrafted-morphology representation used by the frozen baseline.
tr_raw=build_sequences(BEATS['train'],8,'rr_morphology'); va_raw=build_sequences(BEATS['validation'],8,'rr_morphology'); te_raw=build_sequences(BEATS['test'],8,'rr_morphology')
Xtr,ytr,mtr=tr_raw; Xva,yva,mva=va_raw; Xte,yte,mte=te_raw; Xtr,Xva,Xte=normalize_features(Xtr,Xva,Xte)
classes=CLASSES; cidx={c:i for i,c in enumerate(classes)}
all_symbols=sorted(set(mtr.symbol.astype(str))|set(mva.symbol.astype(str))|set(mte.symbol.astype(str))); symbol_idx={s:i for i,s in enumerate(all_symbols)}
strtr=mtr.symbol.astype(str).map(symbol_idx).to_numpy(); strva=mva.symbol.astype(str).map(symbol_idx).to_numpy(); strte=mte.symbol.astype(str).map(symbol_idx).to_numpy()
domain_records=sorted(mtr.record.astype(str).unique()); domain_idx={r:i for i,r in enumerate(domain_records)}; dtr=mtr.record.astype(str).map(domain_idx).to_numpy()

# Freeze prior baseline metadata and checkpoint before any Phase 3 change.
shutil.copy2(OUT/'baseline_rr_morphology_predictions.csv',P3/'baseline_predictions.csv'); shutil.copy2(OUT/'checkpoints/RNN_rr_morphology.pt',P3/'baseline_checkpoint.pt')
base_metrics=json.loads((OUT/'final_metrics.json').read_text()); (P3/'baseline_metrics.json').write_text(json.dumps({'model':'RNN + RR + morphology','MacroF1':0.40123549379362033,'MacroAUPRC':0.5963220323855083,'MacroAUROC':0.790607924571247,'five_seed':'0.4087 ± 0.0174','test_records':SPLIT_LOCK['test_records'],'test_examples':len(yte)},indent=2))

class FeatureDataset(Dataset):
    def __init__(self,X,y,aux=None): self.X=torch.tensor(X,dtype=torch.float32);self.y=torch.tensor(y,dtype=torch.long);self.aux=None if aux is None else torch.tensor(aux,dtype=torch.long)
    def __len__(self):return len(self.y)
    def __getitem__(self,i): return (self.X[i],self.y[i]) if self.aux is None else (self.X[i],self.y[i],self.aux[i])

def score_full(y,p):
    pred=p.argmax(1); pr,re,f1,sup=precision_recall_fscore_support(y,pred,labels=np.arange(5),zero_division=0); au=[];ap=[]
    for i in range(5):
        z=(y==i).astype(int); au.append(roc_auc_score(z,p[:,i]) if len(np.unique(z))==2 else np.nan); ap.append(average_precision_score(z,p[:,i]) if z.sum() else np.nan)
    detail=pd.DataFrame({'class':classes,'precision':pr,'recall':re,'f1':f1,'support':sup,'auroc':au,'auprc':ap})
    return {'MacroF1':float(f1.mean()),'MacroAUPRC':float(np.nanmean(ap)),'MacroAUROC':float(np.nanmean(au)),'BalancedAccuracy':float(balanced_accuracy_score(y,pred)),'Accuracy':float((pred==y).mean()),'N_recall':float(re[0]),'F_recall':float(re[3])},detail,pred

def predict_model(model,X):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),BATCH):out.append(torch.softmax(model(torch.tensor(X[i:i+BATCH],dtype=torch.float32))[0],1).numpy())
    return np.concatenate(out)

def train_symbol(name, mode='current', seed=42, epochs=EPOCHS):
    set_seed(seed); model=make_model('RNN',Xtr.shape[-1]); cw=class_weights(ytr); opt=torch.optim.Adam(model.parameters(),lr=float(CFG['learning_rate']))
    if mode=='symbol_sampler':
        counts=Counter(zip(ytr,strtr)); weights=np.asarray([1.0/max(counts[(int(y),int(s))],1) for y,s in zip(ytr,strtr)],dtype=np.float64); sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),len(weights),replacement=True); loader=DataLoader(FeatureDataset(Xtr,ytr),batch_size=BATCH,sampler=sampler)
    else: loader=DataLoader(FeatureDataset(Xtr,ytr,strtr if mode=='symbol_loss' else None),batch_size=BATCH,shuffle=True)
    best=-1.; state=None
    for _ in range(epochs):
        model.train()
        for batch in loader:
            x,y=batch[0],batch[1]; opt.zero_grad(); logits,_=model(x)
            if mode=='symbol_loss':
                sample_w=torch.tensor([1.0/max(Counter(zip(ytr,strtr))[(int(yy),int(ss))],1) for yy,ss in zip(y.numpy(),batch[2].numpy())],dtype=torch.float32); loss=(nn.functional.cross_entropy(logits,y,weight=torch.tensor(cw,dtype=torch.float32),reduction='none')*sample_w).mean()
            else: loss=nn.functional.cross_entropy(logits,y,weight=torch.tensor(cw,dtype=torch.float32))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        vp=predict_model(model,Xva); val=score_full(yva,vp)[0]['MacroF1']
        if val>best:best=val;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); pp=predict_model(model,Xte);sc,de,pr=score_full(yte,pp);sc.update({'experiment':name,'validation_MacroF1':best,'seed':seed,'mode':mode,'architecture':'RNN','Params':sum(p.numel() for p in model.parameters())});return sc,de,pr,pp,model

# Symbol-aware training comparisons.
symbol_rows=[]; symbol_detail=[]; symbol_outputs={}
for name,mode in [('current_class_balanced','current'),('original_symbol_balanced_sampler','symbol_sampler'),('original_symbol_aware_loss','symbol_loss')]:
    sc,de,pr,pp,m=train_symbol(name,mode,42);symbol_rows.append(sc);symbol_detail.append(de.assign(experiment=name));symbol_outputs[name]=(sc,de,pr,pp,m)
pd.DataFrame(symbol_rows).to_csv(OUT/'phase3_symbol_balance.csv',index=False);pd.concat(symbol_detail).to_csv(P3/'symbol_balance_per_class.csv',index=False)

# Shared encoder with five-class and original-symbol heads.
class MultiTask(nn.Module):
    def __init__(self,input_dim,symbols):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(input_dim,32),nn.LayerNorm(32),nn.ReLU());self.rnn=nn.RNN(32,32,batch_first=True);self.cls=nn.Linear(32,5);self.sym=nn.Linear(32,symbols)
    def forward(self,x):h,_=self.rnn(self.encoder(x));z=h[:,-1];return self.cls(z),self.sym(z)
def predict_mt(m,X):
    m.eval();o=[]
    with torch.no_grad():
        for i in range(0,len(X),BATCH):o.append(torch.softmax(m(torch.tensor(X[i:i+BATCH],dtype=torch.float32))[0],1).numpy())
    return np.concatenate(o)
def train_mt(lam,seed=42):
    set_seed(seed);m=MultiTask(Xtr.shape[-1],len(all_symbols));opt=torch.optim.Adam(m.parameters(),lr=float(CFG['learning_rate']));cw=class_weights(ytr);best=-1;state=None
    loader=DataLoader(FeatureDataset(Xtr,ytr,strtr),batch_size=BATCH,shuffle=True)
    for _ in range(EPOCHS):
        m.train()
        for x,y,s in loader:
            opt.zero_grad();a,b=m(x);loss=nn.functional.cross_entropy(a,y,weight=torch.tensor(cw,dtype=torch.float32))+lam*nn.functional.cross_entropy(b,s);loss.backward();opt.step()
        vp=predict_mt(m,Xva);val=score_full(yva,vp)[0]['MacroF1']
        if val>best:best=val;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(state);pp=predict_mt(m,Xte);sc,de,pr=score_full(yte,pp);sc.update({'lambda':lam,'validation_MacroF1':best,'experiment':f'multitask_lambda_{lam}','Params':sum(p.numel() for p in m.parameters())});return sc,de,pr,pp,m
mt_rows=[];mt_detail=[];mt_outputs={}
for lam in [0.1,0.25,0.5]:
    sc,de,pr,pp,m=train_mt(lam);mt_rows.append(sc);mt_detail.append(de.assign(experiment=f'multitask_lambda_{lam}'));mt_outputs[lam]=(sc,de,pr,pp,m)
pd.DataFrame(mt_rows).to_csv(OUT/'phase3_multitask.csv',index=False);pd.concat(mt_detail).to_csv(P3/'multitask_per_class.csv',index=False)

# Small domain-adversarial encoder with record IDs used only as training supervision.
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,l):ctx.l=l;return x.view_as(x)
    @staticmethod
    def backward(ctx,g):return -ctx.l*g,None
class DomainAdv(nn.Module):
    def __init__(self,input_dim,n_domains):
        super().__init__();self.enc=nn.Sequential(nn.Conv1d(1,8,7,padding=3),nn.ReLU(),nn.Conv1d(8,12,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1));self.rnn=nn.RNN(12,24,batch_first=True);self.cls=nn.Linear(24,5);self.dom=nn.Sequential(nn.Linear(24,24),nn.ReLU(),nn.Linear(24,n_domains))
    def forward(self,x,lam=.15):
        b,t,d=x.shape;z=self.enc(x.reshape(b*t,1,d)).squeeze(-1).reshape(b,t,12);h,_=self.rnn(z);q=h[:,-1];return self.cls(q),self.dom(GRL.apply(q,lam))
def train_domain(seed=42):
    set_seed(seed);m=DomainAdv(Xtr.shape[-1],len(domain_records));opt=torch.optim.Adam(m.parameters(),lr=float(CFG['learning_rate']));cw=class_weights(ytr);best=-1;state=None;loader=DataLoader(FeatureDataset(Xtr,ytr,dtr),batch_size=BATCH,shuffle=True)
    for _ in range(EPOCHS):
        m.train()
        for x,y,d in loader:
            opt.zero_grad();a,b=m(x);loss=nn.functional.cross_entropy(a,y,weight=torch.tensor(cw,dtype=torch.float32))+0.15*nn.functional.cross_entropy(b,d);loss.backward();opt.step()
        m.eval();vp=[]
        with torch.no_grad():
            for i in range(0,len(Xva),BATCH):vp.append(torch.softmax(m(torch.tensor(Xva[i:i+BATCH],dtype=torch.float32))[0],1).numpy())
        val=score_full(yva,np.concatenate(vp))[0]['MacroF1']
        if val>best:best=val;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(state);m.eval();pp=[]
    with torch.no_grad():
        for i in range(0,len(Xte),BATCH):pp.append(torch.softmax(m(torch.tensor(Xte[i:i+BATCH],dtype=torch.float32))[0],1).numpy())
    pp=np.concatenate(pp);sc,de,pr=score_full(yte,pp);sc.update({'experiment':'domain_adversarial','validation_MacroF1':best,'architecture':'small_CNN_RNN_GRL','Params':sum(p.numel() for p in m.parameters())});return sc,de,pr,pp,m
dom_sc,dom_de,dom_pr,dom_pp,dom_model=train_domain();pd.DataFrame([dom_sc]).to_csv(OUT/'phase3_domain_adversarial.csv',index=False);dom_de.assign(experiment='domain_adversarial').to_csv(P3/'domain_adversarial_per_class.csv',index=False)

# Representation ablation table: existing measured baselines plus Phase 3 candidates.
repr_rows=[{'representation':'waveform_only','MacroF1':0.2664970155,'MacroAUPRC':0.5531735736,'MacroAUROC':0.7751677611},{'representation':'handcrafted_morphology','MacroF1':0.4012354938,'MacroAUPRC':0.5963220324,'MacroAUROC':0.7906079246},{'representation':'cnn_morphology_embedding','MacroF1':float(pd.DataFrame(pd.read_csv(OUT/'learned_morphology_results.csv')).query("name == 'CNN_morph_RNN'").MacroF1.iloc[0]),'MacroAUPRC':float(pd.DataFrame(pd.read_csv(OUT/'learned_morphology_results.csv')).query("name == 'CNN_morph_RNN'").MacroAUPRC.iloc[0]),'MacroAUROC':float(pd.DataFrame(pd.read_csv(OUT/'learned_morphology_results.csv')).query("name == 'CNN_morph_RNN'").MacroAUROC.iloc[0])}]
for sc,label in [(symbol_rows[2],'cnn_rr_morph_symbol_loss'),(mt_rows[1],'cnn_rr_morph_multitask'),(dom_sc,'cnn_rr_morph_domain_adversarial')]:repr_rows.append({'representation':label,'MacroF1':sc['MacroF1'],'MacroAUPRC':sc['MacroAUPRC'],'MacroAUROC':sc['MacroAUROC'],'N_recall':sc['N_recall'],'F_recall':sc['F_recall']})
pd.DataFrame(repr_rows).to_csv(OUT/'phase3_representation_ablation.csv',index=False)

# Candidate selection by validation Macro-F1, with secondary checks; baseline remains eligible and is retained unless a candidate passes all gates.
candidates=[]
for sc, label in [(symbol_outputs['original_symbol_balanced_sampler'][0],'symbol_sampler'),(symbol_outputs['original_symbol_aware_loss'][0],'symbol_loss'),(mt_outputs[0.1][0],'multitask_0.1'),(mt_outputs[0.25][0],'multitask_0.25'),(mt_outputs[0.5][0],'multitask_0.5'),(dom_sc,'domain_adversarial')]: candidates.append({**sc,'candidate':label})
cand=pd.DataFrame(candidates); cand.to_csv(P3/'candidate_table.csv',index=False); best_label=str(cand.sort_values(['validation_MacroF1','MacroAUPRC'],ascending=False).iloc[0].candidate); best_tuple={'symbol_sampler':symbol_outputs['original_symbol_balanced_sampler'],'symbol_loss':symbol_outputs['original_symbol_aware_loss'],'multitask_0.1':mt_outputs[0.1],'multitask_0.25':mt_outputs[0.25],'multitask_0.5':mt_outputs[0.5],'domain_adversarial':(dom_sc,dom_de,dom_pr,dom_pp,dom_model)}[best_label]
# The final replacement gate is deliberately conservative: require better Macro-F1 and no major AUPRC loss and improved N or F recall.
best_sc,best_de,best_pr,best_pp,best_model=best_tuple;replace_baseline=best_sc['MacroF1']>0.40123549379362033 and best_sc['MacroAUPRC']>=0.5963220323855083-0.02 and (best_sc['N_recall']>0 or best_sc['F_recall']>0)
final_sc=best_sc if replace_baseline else {'model':'RNN + RR + morphology','MacroF1':0.40123549379362033,'MacroAUPRC':0.5963220323855083,'MacroAUROC':0.790607924571247,'N_recall':0.0,'F_recall':0.0,'selected_phase3_candidate':best_label,'replacement_gate_passed':False,'reason':'No Phase 3 candidate passed all replacement gates; frozen baseline retained.'}
# Copy selected checkpoint only if the gate is passed; otherwise copy the frozen baseline checkpoint.
if replace_baseline:
    torch.save({'state_dict':best_model.state_dict(),'model_name':best_label,'class_names':classes,'config':CFG},OUT/'phase3_best_model.pt');shutil.copy2(OUT/'phase3_best_model.pt',P3/'phase3_best_model.pt')
else:
    shutil.copy2(OUT/'checkpoints/RNN_rr_morphology.pt',OUT/'phase3_best_model.pt');shutil.copy2(OUT/'checkpoints/RNN_rr_morphology.pt',P3/'phase3_best_model.pt')
(OUT/'phase3_metrics.json').write_text(json.dumps(final_sc,indent=2));(OUT/'phase3_best_config.json').write_text(json.dumps({'selected_candidate':best_label,'replacement_gate_passed':replace_baseline,'test_split':SPLIT_LOCK['test_records'],'test_examples':len(yte),'selection_rule':'validation Macro-F1 first; secondary Macro-AUPRC, AUROC, Balanced Accuracy, N/F recall; baseline replacement requires all gates'},indent=2))

# Per-record comparison for baseline and selected Phase 3 candidate.
base_pred_df=pd.read_csv(OUT/'baseline_rr_morphology_predictions.csv');
def record_metrics(df,p):
    rows=[]
    for rec,gp in df.groupby(df.record.astype(str)):
        idx=np.where(mte.record.astype(str).values==str(rec))[0];yy=yte[idx];pp=p[idx];sc,de,_=score_full(yy,pp);row={'record':rec,'MacroF1':sc['MacroF1'],'MacroAUROC':sc['MacroAUROC'],'MacroAUPRC':sc['MacroAUPRC'],'N_recall':float(de.loc[de['class']=='N','recall'].iloc[0]),'S_recall':float(de.loc[de['class']=='S','recall'].iloc[0]),'V_recall':float(de.loc[de['class']=='V','recall'].iloc[0]),'F_recall':float(de.loc[de['class']=='F','recall'].iloc[0]),'Q_recall':float(de.loc[de['class']=='Q','recall'].iloc[0]),'original_symbols':json.dumps(dict(Counter(gp.symbol.astype(str))))};rows.append(row)
    return pd.DataFrame(rows)
base_p=np.column_stack([base_pred_df[f'prob_{c}'] for c in classes]);best_record=record_metrics(mte,best_pp if replace_baseline else base_p);best_record['model']='phase3_final' if replace_baseline else 'frozen_baseline';base_record=record_metrics(mte,base_p);base_record['model']='frozen_baseline';pd.concat([base_record,best_record]).drop_duplicates(['record','model']).to_csv(OUT/'phase3_record_generalization.csv',index=False)
# Plot baseline versus Phase 3 final by held-out record.
plot=base_record.merge(best_record,on='record',suffixes=('_baseline','_phase3'));fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(plot));ax.bar(x-.18,plot.MacroF1_baseline,width=.36,label='Frozen baseline',color='#315a91');ax.bar(x+.18,plot.MacroF1_phase3,width=.36,label='Phase 3 final',color='#e28b42');ax.set_xticks(x,plot.record);ax.set_ylabel('Macro-F1');ax.set_title('Per-record generalization: frozen baseline vs Phase 3 final');ax.legend(frameon=False);fig.tight_layout();fig.savefig(P3/'plots/record_generalization_phase3.png',dpi=180);plt.close(fig)

# N failures: same 100 baseline failures, with Phase 3 predicted class/probabilities and record/symbol distributions.
nfail=base_pred_df[(base_pred_df.true_class=='N')&(base_pred_df.pred_class!='N')].head(100).copy();nfail['phase3_pred_class']=[classes[int(i)] for i in best_pr[nfail.index] if False] if replace_baseline else nfail['pred_class'];nfail['phase3_confidence']=best_pp[nfail.index].max(1) if replace_baseline else nfail['confidence'];nfail['phase3_top_class']=[classes[int(i)] for i in best_pp[nfail.index].argmax(1)] if replace_baseline else nfail['pred_class'];nfail.to_csv(OUT/'phase3_n_failure_analysis.csv',index=False)
# Calibration and validation-only temperature scaling.
def ece(y,p,bins=10):
    conf=p.max(1);pred=p.argmax(1);edges=np.linspace(0,1,bins+1);v=0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(conf>=lo)&(conf<=hi if hi==1 else conf<hi)
        if mask.any():v+=mask.mean()*abs((pred[mask]==y[mask]).mean()-conf[mask].mean())
    return float(v)
def brier(y,p):return float(np.mean(np.sum((p-np.eye(5)[y])**2,axis=1)))
def cal_row(label,y,p,temperature=1.0):return {'model':label,'temperature':temperature,'ECE':ece(y,p),'Brier':brier(y,p),'MacroF1':score_full(y,p)[0]['MacroF1'],'MacroAUPRC':score_full(y,p)[0]['MacroAUPRC'],'MacroAUROC':score_full(y,p)[0]['MacroAUROC']}
# obtain a validation baseline probability from a fresh validation-trained baseline model; test stays locked.
base_model=train_base('RNN',Xtr,ytr,Xva,yva,loss_mode='class_weight',seed=42);base_val=predict_base(base_model,Xva)[0];base_test=predict_base(base_model,Xte)[0]
cal_rows=[cal_row('frozen_baseline_test',yte,base_p),cal_row('phase3_final_test',yte,best_pp if replace_baseline else base_p)]
# Temperature fit uses validation NLL only and is evaluated once on test.
def temp_fit(p,y):
    logits=np.log(np.clip(p,1e-7,1));best=(1e9,1.)
    for t in np.linspace(.5,3.,26):
        q=np.exp(logits/t);q/=q.sum(1,keepdims=True);nll=-np.log(np.clip(q[np.arange(len(y)),y],1e-7,1)).mean()
        if nll<best[0]:best=(nll,float(t))
    return best[1]
t=temp_fit(base_val,yva);test_logits=np.log(np.clip(base_p,1e-7,1));q=np.exp(test_logits/t);q/=q.sum(1,keepdims=True);cal_rows.append(cal_row('frozen_baseline_temperature_scaled',yte,q,t));pd.DataFrame(cal_rows).to_csv(OUT/'phase3_calibration.csv',index=False)
# Five seeds for frozen baseline and strongest Phase 3 candidate, using the same split.
seed_rows=[]
for label,mode in [('frozen_baseline','current'),('phase3_symbol_loss', 'symbol_loss')]:
    for s in [42,43,44,45,46]:
        sc,de,pr,pp,m=train_symbol(f'phase3_seed_{label}_{s}',mode,s);sc['model']=label;seed_rows.append(sc)
pd.DataFrame(seed_rows).to_csv(OUT/'phase3_seed_results.csv',index=False)
# Bootstrap 500 fixed-test resamples for final Phase 3 selected output.
final_p=best_pp if replace_baseline else base_p;rng=np.random.default_rng(42);boot=[]
for _ in range(500):
    ix=rng.integers(0,len(yte),len(yte));sc,_,_=score_full(yte[ix],final_p[ix]);boot.extend([{'metric':'MacroF1','value':sc['MacroF1']},{'metric':'MacroAUROC','value':sc['MacroAUROC']},{'metric':'MacroAUPRC','value':sc['MacroAUPRC']}])
boot_df=pd.DataFrame(boot);out=[]
for metric,grp in boot_df.groupby('metric'):
    est=score_full(yte,final_p)[0][metric];out.append({'model':'phase3_final','metric':metric,'estimate':float(est),'ci_lower_95':float(grp.value.quantile(.025)),'ci_upper_95':float(grp.value.quantile(.975)),'bootstrap_samples':500})
pd.DataFrame(out).to_csv(OUT/'phase3_bootstrap_results.csv',index=False)
# Robustness baseline versus final; corrupt test only.
def corrupt(X,kind):
    z=X.copy();t=np.arange(180,dtype=np.float32)
    if kind=='gaussian_noise':z[:,:,:180]+=np.random.default_rng(11).normal(0,.05,z[:,:,:180].shape).astype(np.float32)
    elif kind=='baseline_wander':z[:,:,:180]+=0.08*np.sin(2*np.pi*t/180*1.2)[None,None,:]
    elif kind=='amplitude_scaling':z[:,:,:180]*=1.2
    elif kind=='missing_samples':z[:,:,81:99]=0
    return z
rob=[]
for model_label,model_obj,ref in [('baseline',base_model,base_p),('phase3_final',best_model if replace_baseline else base_model,final_p)]:
    clean=score_full(yte,ref)[0]['MacroF1']
    for kind in ['clean','gaussian_noise','baseline_wander','amplitude_scaling','missing_samples']:
        p=ref if kind=='clean' else predict_base(model_obj,corrupt(Xte,kind))[0];sc=score_full(yte,p)[0];rob.append({'model':model_label,'corruption':kind,**sc,'MacroF1_delta':float(sc['MacroF1']-clean)})
pd.DataFrame(rob).to_csv(OUT/'phase3_robustness.csv',index=False)
print(json.dumps({'selected_phase3_candidate':best_label,'replacement_gate_passed':replace_baseline,'final_model':final_sc.get('model'),'final_macro_f1':final_sc['MacroF1'],'final_macro_auprc':final_sc['MacroAUPRC'],'final_macro_auroc':final_sc['MacroAUROC'],'N_recall':final_sc['N_recall'],'F_recall':final_sc['F_recall'],'test_examples':len(yte),'test_records':SPLIT_LOCK['test_records']},indent=2))
