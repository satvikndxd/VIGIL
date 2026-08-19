from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'research_results'
learn=pd.read_csv(OUT/'learned_morphology_results.csv'); shift=pd.read_csv(OUT/'symbol_shift_analysis.csv'); rec=pd.read_csv(OUT/'record_generalization.csv'); aug=pd.read_csv(OUT/'morphology_augmentation_results.csv'); win=pd.read_csv(OUT/'window_alignment_results.csv'); seeds=pd.read_csv(OUT/'final_seed_results.csv'); rob=pd.read_csv(OUT/'final_robustness_results.csv'); final=json.loads((OUT/'final_metrics.json').read_text()); cmp=pd.read_csv(OUT/'final_model_comparison.csv')

def md(df, cols=None, n=None, round_digits=4):
    x=df if cols is None else df[cols]
    if n: x=x.head(n)
    return x.round(round_digits).to_markdown(index=False)

seed_summary=seeds.agg({'MacroF1':['mean','std'],'MacroAUPRC':['mean','std'],'MacroAUROC':['mean','std'],'N_recall':['mean','std'],'F_recall':['mean','std']})
report=f'''# VIGIL Phase 2 — Generalization and Minority-Class Failure Study

## Decision summary

The Phase 2 study continued from the existing VIGIL repository, froze the prior RNN + RR + morphology baseline, retained the original grouped split, and used validation Macro-F1 for candidate selection. The held-out test records and labels were not changed. The final decision is to **keep the frozen RNN + RR + morphology model** because the new learned-morphology candidates did not improve held-out performance.

| Final item | Measured result |
|---|---:|
| Frozen baseline / final model | RNN + RR + morphology |
| Baseline Macro-F1 | 0.4012 |
| Final Macro-F1 | 0.4012 |
| Macro-F1 delta | 0.0000 |
| Final Macro-AUPRC | 0.5963 |
| Final Macro-AUROC | 0.7906 |
| N recall | 0.0000 |
| F recall | 0.0000 |
| Test split | 10 locked records / 4,154 sequences |

This is a measured non-improvement result. No CNN, Bi-LSTM, attention, augmentation, or window change is presented as a winner merely because it is more complex.

## 1. Why did the original RNN fail on N and F?

The diagnosis is **not simple mapped-class imbalance**. The grouped split and mapping are correct, while the original-symbol composition differs materially across train, validation, and test. N has 1,200 capped modeling examples in both train and test but no train examples for original symbol `L` in this split and no validation examples for original symbol `N`; F has 767 training examples, 23 validation examples, and only 12 held-out test examples. The error table shows systematic N failures and complete F failure in the frozen test predictions.

The 100-row N-failure analysis is saved in `n_failure_analysis.csv`, with real waveform windows, predicted probabilities, record IDs, original symbols, and predicted classes. The representative top failures are concentrated in held-out records 108 and 104 and are frequently classified as V with high confidence. This is evidence of cross-record morphology/domain shift or representation ambiguity, not proof that N was merely underweighted.

## 2. Is the main problem class frequency, morphology representation, or cross-record domain shift?

The measured evidence supports a combination of **representation weakness and cross-record/original-symbol composition shift**, with an especially unstable F estimate because only 12 F test sequences remain. The prior study showed that waveform + RR alone fell to Macro-F1 0.1886, whereas waveform + RR + morphology reached 0.4012. The new symbol-shift and record-level artifacts show why frequency alone is insufficient: original symbols with similar mapped labels have very different train/test support and error rates.

| Original symbol | Map | Train | Validation | Test | Test error | Test recall |
|---|---|---:|---:|---:|---:|---:|
{shift[shift.original_symbol.isin(['N','A','J','V','F','Q','/','f'])][['original_symbol','mapped_class','train_count','validation_count','test_count','test_error_rate','per_symbol_recall']].round(4).to_string(index=False)}

The per-record file shows heterogeneous generalization. The strongest non-empty record in this run was record 200 at approximately Macro-F1 0.334, while record 108 was approximately 0.008. Record 122 is retained in the locked list but has zero modeling rows after the documented cap/window filtering and is represented explicitly as empty rather than silently removed.

## 3. Which representation produced the largest improvement?

The prior frozen study remains the strongest representation result: waveform + RR + handcrafted morphology produced Macro-F1 0.4012, compared with 0.2665 for waveform-only and 0.1886 for waveform + RR. The measured Phase 2 learned-morphology candidates did not improve over it on the fixed held-out test set. The `baseline_rr_morphology` row in the raw experiment CSV is an internal Phase 2 control re-run with the Phase 2 feature-extractor implementation; it is not used to replace the frozen 0.4012 checkpoint:

{md(learn[learn.name != 'baseline_rr_morphology'][['name','MacroF1','MacroAUPRC','MacroAUROC','N_recall','F_recall','validation_MacroF1']], round_digits=4)}

The best learned morphology candidate by validation Macro-F1 was `CNN_morph_RNN`, but its held-out Macro-F1 was 0.1217. This validation-to-test gap is itself evidence of cross-record generalization weakness. The final deployed research choice therefore remains the smaller handcrafted-feature RNN rather than the learned encoder.

## 4. Did learned morphology improve over handcrafted morphology?

No. The compact CNN morphology encoder was tested with RNN, Bi-LSTM, and Bi-LSTM + Attention. All three underperformed the frozen handcrafted-morphology baseline on the fixed held-out test set. The result argues against adding a learned morphology encoder without a stronger representation-training protocol or more diverse record-level support.

## 5. Did targeted augmentation help?

The requested train-only comparison is in `morphology_augmentation_results.csv`. The study includes no augmentation, the previous generic perturbation, and a conservative morphology-aware perturbation that avoids the R-peak region where possible.

{md(aug[['name','MacroF1','MacroAUPRC','MacroAUROC','N_recall','F_recall','augmentation']], round_digits=4)}

The retained final model is not changed based on augmentation because no augmentation candidate produced a validated and held-out improvement over the frozen baseline. No uncontrolled duplication of minority examples was used.

## 6. Did better window alignment help?

The controlled alignment experiment kept total length at 180 samples and varied only the pre/post split:

{md(win[['name','window_pre','window_post','MacroF1','MacroAUPRC','MacroAUROC','N_recall','F_recall']], round_digits=4)}

Within this Phase 2 alignment run, 90/90 was the strongest configuration by Macro-F1 and Macro-AUROC. The 120/60 window was materially worse. The project therefore retains the original 90/90 preprocessing contract.

## 7. Did the final model improve N/F recall?

No. The retained final model has N recall 0.000 and F recall 0.000. Some exploratory candidates increased F recall on the held-out split, but they had substantially worse Macro-F1/AUPRC and were not selected. This is precisely why the report does not claim that a higher isolated F recall constitutes a successful model.

## 8. Did improvements generalize across held-out records?

No Phase 2 candidate demonstrated a reliable across-record improvement over the frozen baseline. The per-record visualization is `phase2/plots/record_generalization.png`, and the machine-readable table is `record_generalization.csv`. The record spread is wide and includes explicit empty-record handling; this supports the conclusion that cross-record domain shift remains a primary limitation.

## 9. What performance trade-offs were introduced?

Learned morphology increased architectural complexity but reduced held-out Macro-F1, Macro-AUPRC, and Macro-AUROC. Augmentation and alignment altered minority behavior without establishing an overall improvement. The retained handcrafted RNN is smaller and more interpretable operationally, while its unresolved N/F recall is kept visible in the dashboard rather than hidden.

The five-seed confirmation for the retained RNN + RR + morphology family is:

{seed_summary.round(4).to_markdown()}

This corresponds to approximately Macro-F1 **0.4087 ± 0.0174**, Macro-AUPRC **0.5909 ± 0.0211**, and Macro-AUROC **0.7972 ± 0.0235**. N recall remained zero across the reported family rows; F recall was near zero on average (**0.0167 ± 0.0373**) and was not consistently recovered.

The retained model's robustness table is:

{md(rob[['corruption','MacroF1','MacroAUPRC','MacroAUROC','MacroF1_drop_from_clean']], round_digits=4)}

These are controlled perturbations, not evidence of noise immunity.

## 10. Which architecture should be deployed, and why?

For the current research prototype, the recommended deployed model remains **RNN + RR + handcrafted morphology**. It is recommended because it is the strongest measured overall model on the fixed grouped test split, it is smaller than the learned morphology alternatives, its five-seed behavior is recorded, and no more complex candidate improved held-out Macro-F1. It must not be described as clinically superior or diagnostically validated.

## Artifact contract

| Requirement | Artifact |
|---|---|
| Stronger morphology experiment | `learned_morphology_results.csv` |
| Feature schema | `feature_schema.json` |
| Symbol shift | `symbol_shift_analysis.csv` and `phase2/plots/symbol_shift_composition.png` |
| Per-record generalization | `record_generalization.csv` and `phase2/plots/record_generalization.png` |
| Morphology-aware augmentation | `morphology_augmentation_results.csv` |
| Window alignment | `window_alignment_results.csv` |
| N failure gallery | `n_failure_analysis.csv` and `phase2/plots/n_failure_gallery.png` |
| Final comparison | `final_model_comparison.csv` |
| Final seeds | `final_seed_results.csv` |
| Final robustness | `final_robustness_results.csv` |
| Final model | `final_model.pt` and `final_metrics.json` |
| Final reproducibility | `final_config.json` and `configs/split_lock.json` |

## References

[1]: https://physionet.org/content/mitdb/1.0.0/ "MIT-BIH Arrhythmia Database on PhysioNet"
[2]: https://www.kaggle.com/datasets/abdallahwagih/mit-bih-arrhythmia-database "Kaggle MIT-BIH Arrhythmia Database mirror"
'''
(OUT/'phase2_report.md').write_text(report)
print(OUT/'phase2_report.md')
