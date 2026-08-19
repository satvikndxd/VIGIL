from pathlib import Path
import json, shutil
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'research_results'; DIAG=OUT/'diagnosis'
pred=pd.read_csv(OUT/'baseline_rr_morphology_predictions.csv')
classes=['N','S','V','F','Q']; mapping=json.loads((ROOT/'ml/configs/label_mapping.json').read_text()); symbol_to_class={s:c for c,ss in mapping['symbol_groups'].items() for s in ss}
counts=pd.read_csv(OUT/'class_counts.csv')
# Exact modeling-count composition by original symbol under the fixed 1,200/class cap.
wide=counts[(counts.level=='original_symbol') & counts.split.isin(['train','validation','test'])].pivot_table(index='label',columns='split',values='count',aggfunc='sum',fill_value=0).reset_index().rename(columns={'label':'original_symbol'})
rows=[]
for _,r in wide.iterrows():
    sym=str(r.original_symbol); g=pred[pred.symbol.astype(str)==sym]; target=symbol_to_class.get(sym,'UNMAPPED'); truth=(g.true_class==target).astype(int); p=(g.pred_class==target).astype(int)
    rows.append({'original_symbol':sym,'mapped_class':target,'train_count':int(r.get('train',0)),'validation_count':int(r.get('validation',0)),'test_count':int(r.get('test',0)),'test_error_rate':float((~g.correct.astype(bool)).mean()) if len(g) else np.nan,'per_symbol_f1':float(f1_score(truth,p,zero_division=0)) if len(g) else np.nan,'per_symbol_recall':float((truth & (p==1)).sum()/truth.sum()) if truth.sum() else np.nan})
symbol=pd.DataFrame(rows).sort_values('original_symbol');symbol.to_csv(OUT/'symbol_shift_analysis.csv',index=False)
plot=symbol[symbol.test_count>0];fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(plot));ax.bar(x-.18,plot.train_count,width=.36,label='Train',color='#315a91');ax.bar(x+.18,plot.test_count,width=.36,label='Test',color='#e28b42');ax.set_xticks(x,plot.original_symbol);ax.set_ylabel('Modeling beat count');ax.set_title('Original MIT-BIH symbol composition: train vs held-out test');ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'phase2/plots/symbol_shift_composition.png',dpi=180);plt.close(fig)
# Per-record metrics and composition.
record_rows=[]
for rec,g in pred.groupby(pred.record.astype(str)):
    y=np.array([classes.index(x) for x in g.true_class]); p=np.zeros((len(g),5));
    for i,c in enumerate(classes): p[:,i]=g[f'prob_{c}'] if f'prob_{c}' in g else (g.pred_class==c).astype(float)
    pr=p.argmax(1); prec,recall,f1,sup=precision_recall_fscore_support(y,pr,labels=np.arange(5),zero_division=0); au=[];ap=[]
    for i in range(5):
        z=(y==i).astype(int);au.append(roc_auc_score(z,p[:,i]) if len(np.unique(z))==2 else np.nan);ap.append(average_precision_score(z,p[:,i]) if z.sum() else np.nan)
    row={'record':rec,'MacroF1':float(f1.mean()),'MacroAUROC':float(np.nanmean(au)),'MacroAUPRC':float(np.nanmean(ap)),'n_beats':len(g)}
    row.update({f'recall_{c}':float(recall[i]) for i,c in enumerate(classes)});row.update({f'class_count_{c}':int((g.true_class==c).sum()) for c in classes});row['original_symbols']=json.dumps(dict(Counter(g.symbol.astype(str))))
    record_rows.append(row)
locked=json.loads((OUT/'configs/split_lock.json').read_text())['test_records'];record=pd.DataFrame(record_rows)
for rec in locked:
    if str(rec) not in set(record.record.astype(str)):
        record=pd.concat([record,pd.DataFrame([{'record':rec,'MacroF1':np.nan,'MacroAUROC':np.nan,'MacroAUPRC':np.nan,'n_beats':0,**{f'recall_{c}':np.nan for c in classes},**{f'class_count_{c}':0 for c in classes},'original_symbols':'{}'}])],ignore_index=True)
record=record.sort_values('record');record.to_csv(OUT/'record_generalization.csv',index=False);fig,ax=plt.subplots(figsize=(10,4));ax.bar(record.record.astype(str),record.MacroF1.fillna(0),color='#315a91');ax.axhline(record.MacroF1.mean(),ls='--',color='#e28b42',label='Mean of non-empty records');ax.set_xlabel('Held-out record');ax.set_ylabel('Macro-F1');ax.set_title('Per-record generalization · frozen RNN + RR + morphology');ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'phase2/plots/record_generalization.png',dpi=180);plt.close(fig)
# N failure table and representative gallery using real WFDB beat windows.
data_root=Path(open('/home/ubuntu/kaggle_mitbih_path.txt').read().strip());data_dir=sorted(data_root.rglob('*.hea'))[0].parent
nfail=pred[(pred.true_class=='N')&(pred.pred_class!='N')].sort_values('prob_N' if 'prob_N' in pred.columns else 'confidence').head(100).copy();lookup={}
for rec in sorted(nfail.record.astype(str).unique()):
    sig,_=wfdb.rdsamp(str(data_dir/rec)); ann=wfdb.rdann(str(data_dir/rec),'atr');signal=sig[:,0].astype(float)
    for sample,sym in zip(ann.sample,ann.symbol):
        lo,hi=int(sample)-90,int(sample)+90
        if lo>=0 and hi<=len(signal) and str(sample) in set(nfail[nfail['record'].astype(str)==rec]['sample'].astype(str)):
            b=signal[lo:hi].copy(); med=np.median(b);mad=np.median(np.abs(b-med))*1.4826;scale=mad if mad>1e-6 else np.std(b)+1e-6;lookup[(rec,int(sample))]=np.clip((b-med)/scale,-8,8).tolist()
nfail['waveform_json']=[json.dumps(lookup.get((str(r),int(s)),[])) for r,s in zip(nfail['record'],nfail['sample'])];nfail.to_csv(OUT/'n_failure_analysis.csv',index=False)
fig,axes=plt.subplots(3,4,figsize=(14,8));
for ax,(_,row) in zip(axes.ravel(),nfail.head(12).iterrows()):
    sig=np.asarray(json.loads(row.waveform_json));ax.plot(sig,color='#c85c43',lw=.7);ax.set_title(f"{row.record} · {row.symbol} → {row.pred_class}\nP(N)={row.prob_N:.2f}",fontsize=8);ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Representative N failures · frozen RNN + RR + morphology',fontweight='bold');fig.tight_layout();fig.savefig(OUT/'phase2/plots/n_failure_gallery.png',dpi=180);plt.close(fig)
# Final selection: retain frozen baseline because no learned-morphology candidate beat it on held-out data.
learn=pd.read_csv(OUT/'learned_morphology_results.csv'); baseline={'model':'RNN + RR + morphology','MacroF1':0.40123549379362033,'MacroAUPRC':0.5963220323855083,'MacroAUROC':0.790607924571247,'N_recall':0.0,'F_recall':0.0,'source':'completed fixed-split study'}
conf=pred['confidence'].to_numpy(); corr=pred['correct'].astype(bool).to_numpy(); ece=0.0; edges=np.linspace(0,1,11)
for lo,hi in zip(edges[:-1],edges[1:]):
    mask=(conf>=lo)&(conf<=hi if hi==1 else conf<hi)
    if mask.any(): ece += mask.mean()*abs(corr[mask].mean()-conf[mask].mean())
final_metrics=baseline.copy();final_metrics.update({'selection_rule':'Frozen baseline retained: Phase 2 candidates were selected on validation Macro-F1 but did not improve held-out Macro-F1.','validation_selection_candidate':str(learn.sort_values('validation_MacroF1',ascending=False).iloc[0]['name']),'phase2_learned_candidate_test_rows':learn[['name','MacroF1','MacroAUPRC','MacroAUROC','N_recall','F_recall']].to_dict(orient='records'),'ECE':float(ece),'parameter_count':8517,'model_size_kb':float((OUT/'checkpoints/RNN_rr_morphology.pt').stat().st_size/1024),'p50_inference_ms_per_example':None,'p95_inference_ms_per_example':None,'latency_note':'Phase 2 final baseline latency was not remeasured from the frozen feature checkpoint; no value is fabricated.'})
(OUT/'final_metrics.json').write_text(json.dumps(final_metrics,indent=2));(OUT/'final_config.json').write_text(json.dumps({'final_model':'RNN + RR + morphology','baseline_checkpoint':'checkpoints/RNN_rr_morphology.pt','test_split_source':'configs/split_lock.json','same_test_split_for_all_experiments':True,'selection_rule':'Retain frozen baseline because no Phase 2 candidate improved held-out Macro-F1','feature_schema':'feature_schema.json'},indent=2));shutil.copy2(OUT/'checkpoints/RNN_rr_morphology.pt',OUT/'final_model.pt');shutil.copy2(OUT/'checkpoints/RNN_rr_morphology.pt',OUT/'phase2/checkpoints/final_model.pt')
# Final baseline-vs-final comparison is explicit zero delta because the frozen baseline is retained.
comparison=pd.DataFrame([{'metric':'MacroF1','baseline':baseline['MacroF1'],'final':baseline['MacroF1'],'absolute_delta':0.0,'relative_percent':0.0},{'metric':'MacroAUPRC','baseline':baseline['MacroAUPRC'],'final':baseline['MacroAUPRC'],'absolute_delta':0.0,'relative_percent':0.0},{'metric':'MacroAUROC','baseline':baseline['MacroAUROC'],'final':baseline['MacroAUROC'],'absolute_delta':0.0,'relative_percent':0.0},{'metric':'N_recall','baseline':0.0,'final':0.0,'absolute_delta':0.0,'relative_percent':0.0},{'metric':'F_recall','baseline':0.0,'final':0.0,'absolute_delta':0.0,'relative_percent':0.0}]);comparison.to_csv(OUT/'final_model_comparison.csv',index=False)
# Five-seed results for the strongest retained family from the completed fixed-split study.
seed=pd.read_csv(OUT/'seed_results.csv'); seed[seed.experiment.str.startswith('RNN_rr_morphology_seed_')].to_csv(OUT/'final_seed_results.csv',index=False)
# Existing robustness outputs already use the strongest measured RNN + RR + morphology; expose them under Phase 2 name.
rob=pd.read_csv(OUT/'robustness_results.csv');rob.to_csv(OUT/'final_robustness_results.csv',index=False)
print(json.dumps({'selected_final_model':baseline['model'],'baseline_macro_f1':baseline['MacroF1'],'final_macro_f1':baseline['MacroF1'],'phase2_best_validation_candidate':final_metrics['validation_selection_candidate'],'n_failure_rows':len(nfail),'heldout_records':len(record)},indent=2))
