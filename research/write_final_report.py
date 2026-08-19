from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'research_results'
counts=pd.read_csv(OUT/'class_counts.csv'); per=pd.read_csv(OUT/'diagnosis/../per_class_metrics.csv')
with open(OUT/'diagnosis/split_integrity.json') as f: split=json.load(f)
with open(OUT/'diagnosis/mapping_verification.json') as f: mapping=json.load(f)
errors=pd.read_csv(OUT/'diagnosis/error_by_original_symbol.csv')
ab=pd.read_csv(OUT/'ablation_results.csv'); aug=pd.read_csv(OUT/'augmentation_results.csv'); ctx=pd.read_csv(OUT/'context_results.csv'); feat=pd.read_csv(OUT/'feature_results.csv'); arch=pd.read_csv(OUT/'architecture_results.csv'); hier=pd.read_csv(OUT/'hierarchical_results.csv'); seeds=pd.read_csv(OUT/'seed_results.csv'); ci=pd.read_csv(OUT/'bootstrap_confidence_intervals.csv'); rob=pd.read_csv(OUT/'robustness_results.csv'); candidates=pd.read_csv(OUT/'selected_candidate_comparison.csv')

class_order=['N','S','V','F','Q']
model_counts=counts[(counts.level=='five_class') & counts.split.isin(['train','validation','test'])].pivot(index='label',columns='split',values='count').reindex(class_order)
model_counts['total']=model_counts.sum(axis=1); model_counts.loc['TOTAL']=model_counts.sum(axis=0)
raw=counts[counts.level=='all_records_raw_annotations'].sort_values('label')
seed_summary=seeds.assign(family=seeds['experiment'].str.rsplit('_seed_',n=1).str[0]).groupby('family').agg(seeds=('seed','count'),MacroF1_mean=('MacroF1','mean'),MacroF1_std=('MacroF1','std'),MacroAUPRC_mean=('MacroAUPRC','mean'),MacroAUPRC_std=('MacroAUPRC','std'),MacroAUROC_mean=('MacroAUROC','mean'),MacroAUROC_std=('MacroAUROC','std')).reset_index()

def table(df, cols=None, index=False):
    x=df if cols is None else df[cols]
    return x.to_markdown(index=index)

base=per.copy()
selected=per[per['class'].isin(class_order)].copy()
report=f'''# VIGIL Fixed-Split Minority-Class and Generalization Study

## Executive conclusion

This study diagnosed VIGIL before changing the model or README disclaimer. The split is record- and subject-level leakage-safe, the five-class mapping is implemented exactly, and the test set was held fixed across all experiments. The failures are therefore not explained by record leakage or by a mislabeled AAMI-style mapping alone.

The strongest measured overall approach in this study was **RNN + RR interval + morphology features**. On the fixed test split it improved Macro-F1 from **{float(candidates.loc[candidates.experiment=='RNN','MacroF1'].iloc[0]):.3f}** for the matched flat waveform RNN to **{float(candidates.loc[candidates.experiment=='features_rr_morphology','MacroF1'].iloc[0]):.3f}** (+{float(candidates.loc[candidates.experiment=='features_rr_morphology','MacroF1'].iloc[0]-candidates.loc[candidates.experiment=='RNN','MacroF1'].iloc[0]):.3f}) and Macro-AUPRC from **{float(candidates.loc[candidates.experiment=='RNN','MacroAUPRC'].iloc[0]):.3f}** to **{float(candidates.loc[candidates.experiment=='features_rr_morphology','MacroAUPRC'].iloc[0]):.3f}**. However, it did **not** recover N or F recall in this run. The hierarchical classifier improved Macro-F1 to **{float(hier.MacroF1.iloc[0]):.3f}** and recovered F recall to **0.250**, but N recall remained **0.000**. This is a measurable improvement with a clear limitation, not clinical superiority.

## 1. Diagnosis before intervention

### Fixed split and leakage audit

The exact split contains **31 train records, 7 validation records, and 10 test records**, with 5,567, 2,570, and 4,154 modeling sequences respectively. Record intersections and subject intersections are empty. The audit writes the exact lists and subject IDs to `diagnosis/split_integrity.json`.

{table(model_counts.reset_index(), ['label','train','validation','test','total'])}

The split lock reports `no_record_leakage: true` and `no_subject_leakage: true`. A deterministic test hash is recorded in `configs/experiment_manifest.json`, and every requested experiment uses the same test sequence labels and metadata.

### Mapping verification

The implemented mapping is exact: `{str(mapping['exact']).lower()}`. The mapping is:

{table(pd.DataFrame([{'class':k,'symbols':', '.join(v)} for k,v in mapping['implemented'].items()]))}

Observed annotation symbols outside the five-class mapping are intentionally excluded from beat targets: `{', '.join(mapping['unmapped_observed_symbols'])}`. The full comparison between expected and implemented mappings is in `diagnosis/mapping_verification.json`.

### Original MIT-BIH annotation symbols before mapping

The following table is the raw annotation-symbol count across all records before five-class mapping. The split-specific capped symbol counts are in `class_counts.csv`; the modeling cap is 1,200 beats per mapped class, so split-level modeling counts and raw annotation totals must not be conflated.

{table(raw[['label','count','mapped_class']], ['label','count','mapped_class'])}

### Baseline per-class metrics

The baseline audit evaluated the committed RNN checkpoint on the fixed 4,154-sequence test split.

{table(selected[['class','precision','recall','f1','support','auroc_ovr','auprc_ovr']].round(4))}

![Raw and normalized confusion matrices](confusion_matrix.png)

The baseline errors are concentrated in original symbols and mapped classes rather than being uniform across all symbols. The highest observed test-symbol error rates are:

{table(errors.head(12).round(4))}

This identifies a **representation and cross-record generalization problem** in addition to class-frequency effects. For example, the mapping is correct, but original `N`, `Q`, `/`, `f`, `F`, and `J` symbols have high test error rates in the held-out records. The original symbols represented in a class differ sharply between splits, which creates a morphology/domain-shift problem that class weighting alone cannot solve.

## 2. Controlled interventions on the same test split

### Weighting, sampling, focal loss, and augmentation

{table(aug[['experiment','MacroF1','MacroAUPRC','MacroAUROC','BalancedAccuracy','minority_recall_N','minority_recall_F']].round(4))}

Class weighting improved probability-oriented metrics relative to the unweighted run but did not recover N or F recall. WeightedRandomSampler reduced Macro-F1 to **{float(ab.loc[ab.experiment=='weighted_random_sampler','MacroF1'].iloc[0]):.3f}**, indicating that oversampling without enough morphology/generalization support can increase overfitting or decision instability. Focal loss also left N and F recall at zero. Controlled train-only ECG augmentation produced the best intervention-family Macro-F1 (**{float(ab.loc[ab.experiment=='controlled_augmentation','MacroF1'].iloc[0]):.3f}**) but still did not recover N or F recall.

### Temporal context ablation

{table(ctx[['experiment','context','MacroF1','MacroAUPRC','MacroAUROC','BalancedAccuracy','MinRecall']].round(4))}

The eight-beat context was best among the tested RNN contexts on Macro-F1, but 16 and 32 beats did not improve it, and all contexts retained a zero minimum recall. The limitation is therefore not solved by simply increasing temporal context.

### Feature ablation

{table(feat[['experiment','feature_mode','MacroF1','MacroAUPRC','MacroAUROC','BalancedAccuracy','MinRecall']].round(4))}

RR-only features hurt performance, whereas **RR + morphology** improved Macro-F1 by **{float(feat.loc[feat.feature_mode=='rr_morphology','MacroF1'].iloc[0]-feat.loc[feat.feature_mode=='waveform','MacroF1'].iloc[0]):.3f}** and Macro-AUPRC by **{float(feat.loc[feat.feature_mode=='rr_morphology','MacroAUPRC'].iloc[0]-feat.loc[feat.feature_mode=='waveform','MacroAUPRC'].iloc[0]):.3f}**. This is the clearest evidence that the dominant bottleneck is not just class count: adding explicit morphology and rhythm descriptors helps the classifier separate held-out record patterns. N and F recall nevertheless remain unresolved.

### Architecture benchmark

{table(arch[['architecture','MacroF1','MacroAUPRC','MacroAUROC','BalancedAccuracy','MinRecall','Params']].round(4))}

Bi-LSTM + Attention was the strongest architecture-only model by Macro-F1 in this harness, but it did not beat the feature-enhanced RNN overall. The CNN → Bi-LSTM → Attention model was worse in this run, so it is not presented as a forced winner.

### Hierarchical classifier

{table(hier[['architecture','MacroF1','MacroAUPRC','MacroAUROC','BalancedAccuracy','MinRecall']].round(4))}

The hierarchical N-versus-non-N then S/V/F/Q classifier reached Macro-F1 **{float(hier.MacroF1.iloc[0]):.3f}**, below the feature-enhanced RNN but above the flat waveform RNN. Its per-class rows are in `per_class_metrics_experiments.csv`: N recall remained 0.000, while F recall reached 0.250 and Q recall reached 0.659. The hierarchy changes the error trade-off; it does not solve all classes.

## 3. Multi-seed stability and confidence intervals

The strongest two architecture finalists and the morphology-feature RNN were each run with seeds 42–46 on the same fixed split.

{table(seed_summary.round(4))}

Bootstrap confidence intervals use 500 resamples of the fixed test predictions for the selected strongest candidate, RNN + RR + morphology:

{table(ci.round(4))}

The interval is an uncertainty estimate for this held-out sample, not a clinical confidence claim and not a substitute for external validation.

## 4. Robustness of the strongest measured candidate

{table(rob[['model','corruption','MacroF1','MacroAUPRC','MacroAUROC','BalancedAccuracy','MacroF1_drop_from_clean']].round(4))}

The tested perturbation magnitudes did not materially reduce Macro-F1 for this candidate, although amplitude scaling reduced it by **{abs(float(rob.loc[rob.corruption=='amplitude_scaling','MacroF1_drop_from_clean'].iloc[0])):.3f}** and Gaussian noise, baseline wander, and the 10% missing-sample mask were near the clean result. This is a limited controlled robustness result, not evidence of real-world noise immunity.

## 5. Direct answer: cause, intervention, and trade-off

**Cause.** The audit rules out record/subject leakage and mapping implementation error. The dominant measured failure pattern is a combination of split-specific symbol composition, held-out morphology/domain shift, extreme effective support scarcity for F (12 test sequences), and a representation that is insufficient for N/Q/F distinctions under the chosen grouped split. Training capping makes the mapped training set look balanced, but it cannot create new morphology diversity or make the test symbol composition match training.

**What improved it.** The strongest fixed-split improvement came from adding RR and morphology descriptors to the waveform sequence: Macro-F1 rose from 0.266 to 0.401 and Macro-AUPRC from 0.553 to 0.596. The hierarchical formulation was the next strongest by Macro-F1 at 0.396 and materially increased F recall to 0.250, but it left N recall at zero. Controlled augmentation improved the plain RNN to Macro-F1 0.291, but still did not recover N/F.

**Trade-off.** The feature-enhanced RNN increases parameter count from 8,133 to 8,517 and improves overall class-balanced metrics, but it still has zero N and F recall in the measured run. The hierarchical model increases parameters and complexity and changes the false-error structure; it is not uniformly better. Attention is useful as a diagnostic model but was not the strongest overall. No intervention justifies claiming clinical superiority.

## Reproducibility and artifacts

All requested outputs are exported under `research_results/`:

| Artifact | Location |
|---|---|
| Class counts | `class_counts.csv` |
| Baseline per-class metrics | `per_class_metrics.csv` |
| Raw and normalized confusion matrices | `diagnosis/confusion_matrix_raw.csv`, `diagnosis/confusion_matrix_normalized.csv`, `confusion_matrix.png` |
| Loss/sampler/augmentation ablations | `ablation_results.csv`, `augmentation_results.csv` |
| Context ablation | `context_results.csv` |
| Feature ablation | `feature_results.csv` |
| Architecture benchmark | `architecture_results.csv` |
| Hierarchical comparison | `hierarchical_results.csv` |
| Robustness | `robustness_results.csv` |
| Five-seed stability | `seed_results.csv` |
| Bootstrap intervals | `bootstrap_confidence_intervals.csv` |
| Symbol-level error attribution | `diagnosis/error_by_original_symbol.csv` |
| Split and mapping verification | `diagnosis/split_integrity.json`, `diagnosis/mapping_verification.json` |
| Exact experiment manifest | `configs/experiment_manifest.json` |

The harness is `research/experiment_harness.py`; the diagnosis script is `research/diagnose_audit.py`. Both preserve the fixed split and write exact configurations.
'''
report += "\n\n## References\n\n[1]: https://physionet.org/content/mitdb/1.0.0/ \\\"MIT-BIH Arrhythmia Database on PhysioNet\\\"\n[2]: https://www.kaggle.com/datasets/abdallahwagih/mit-bih-arrhythmia-database \\\"Kaggle MIT-BIH Arrhythmia Database mirror\\\"\n"
(OUT/'final_report.md').write_text(report)
print(OUT/'final_report.md')
