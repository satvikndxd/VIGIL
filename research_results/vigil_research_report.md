# VIGIL: Interpretable Temporal Deep Learning for ECG Arrhythmia Classification

## Executive Summary

### TL;DR

This study evaluated whether progressively more expressive, imbalance-aware, and domain-generalized models could improve five-class MIT-BIH beat classification under a locked record-level split. The final decision is to **retain the compact RNN + RR + handcrafted morphology baseline**: no Phase 2 or Phase 3 candidate passed the predefined replacement gate requiring a higher overall Macro-F1, no major Macro-AUPRC degradation, improved N or F recall, and stable behavior across seeds. The result must be interpreted against the **SOTA illusion** created by random beat-level splits. When beats from the same patient record appear in both training and test data, patient-specific baseline morphology, electrode configuration, noise signature, and annotation context leak across the split; this can produce apparently spectacular 98%+ accuracy without demonstrating out-of-distribution generalization. VIGIL instead treats the patient record as the statistical unit and reports its **0.40 Macro-F1 as an unvarnished OOD baseline**, not as a low score relative to leakage-contaminated benchmarks. In comic-book terms, random splitting is like recognizing a character by their color palette; record-level splitting forces the model to recognize them when drawn in a completely different art style.

| Final-model result | Estimate | Evaluation condition |
|---|---:|---|
| Macro-F1 | **0.4012** | Locked held-out test split |
| Macro-AUPRC | **0.5963** | Locked held-out test split |
| Macro-AUROC | **0.7906** | Locked held-out test split |
| N recall | **0.0000** | Locked held-out test split |
| F recall | **0.0000** | Locked held-out test split |
| Parameters | **8,517** | Final checkpoint |

The central scientific finding is that the unresolved minority-class failures are not explained by mapped-class frequency alone. The evidence instead supports a combination of **representation weakness** and **cross-record/original-symbol composition shift**: held-out N beats from specific records are systematically mapped to V with high confidence, while F performance is estimated from a very small test support. Symbol-aware sampling and symbol-aware loss degraded performance, learned CNN morphology encoders underperformed the handcrafted representation, and domain-adversarial training collapsed under the tested optimization budget. Multi-task symbol supervision produced an isolated F-recall improvement, but not a clean overall improvement.

The report describes these negative results as useful evidence about the limits of the current data, representation, and training protocol. The model remains a research prototype for interpretable temporal modeling and must not be interpreted as clinically validated or diagnostically capable.

## Study Objective, Data Contract, and Reproducibility

The objective was to determine whether interventions targeting class imbalance, temporal context, morphology representation, model architecture, hierarchical decision structure, original-symbol supervision, and record-domain invariance could improve held-out five-class performance without sacrificing ranking quality or reproducibility. This objective is deliberately stricter than the common random beat-level evaluation protocol, in which neighboring beats from the same record can cross the train/test boundary and make the task resemble patient identification rather than OOD morphology recognition. Prior inter-patient ECG work similarly treats separation across patients as a distinct generalization setting rather than an interchangeable split convention [3]. The fixed record-level split is therefore not a cosmetic evaluation choice: it is the central experimental control that converts apparently high in-distribution performance into a more difficult but scientifically meaningful test of transfer across patients and recording conditions.

| Study component | Locked specification |
|---|---|
| Dataset | MIT-BIH Arrhythmia Database, accessed through a Kaggle mirror [1] [2] |
| Mapped classes | N, S, V, F, Q |
| Split unit | Record, with subject-level leakage audit |
| Train/validation/test records | 31 / 7 / 10 |
| Train/validation/test sequences | 5,567 / 2,570 / 4,154 |
| Beat window | 90 samples before and 90 samples after the target beat |
| Temporal context | 8-beat context for the retained baseline |
| Normalization | Robust median/MAD normalization |
| Handcrafted schema | RR and morphology features, including amplitude, energy, RMS, slope, width, QRS proxy, RR interval, prior RR, RR ratio, local heart rate, and local RR variability |
| Primary metric | Macro-F1 |
| Secondary metrics | Macro-AUPRC, Macro-AUROC, balanced accuracy, per-class recall |
| Test-set policy | Fixed and unchanged throughout Phases 1–3 |
| Leakage audit | Record leakage: absent; subject leakage: absent; mapping verification: exact |

The evaluation contract intentionally prioritizes Macro-F1 and per-class recall rather than accuracy, because the research question concerns minority-class and cross-record behavior. The locked split, exact mapping audit, and exported predictions provide the basis for reproducing every reported comparison.

[Insert Figure 1: VIGIL architecture and inference pathway, showing the 90/90 beat-centered waveform window, eight-beat temporal context, RR features, handcrafted morphology branch, recurrent encoder, five-class output, and optional explanation artifacts.]

### Mapped-Class Composition

| Split | N | S | V | F | Q | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,200 | 1,200 | 1,200 | 767 | 1,200 | 5,567 |
| Validation | 1,200 | 143 | 1,200 | 23 | 4 | 2,570 |
| Test | 1,200 | 542 | 1,200 | 12 | 1,200 | 4,154 |

The mapped counts reveal a nonstandard generalization problem: the test distribution is not simply a conventional long-tailed sample. Several mapped classes are capped or balanced at the modeling level, yet the model still fails on N and F. In particular, the validation support for S, F, and Q is highly uneven, making validation-only selection noisy for the rare failure modes that motivated the study.

### Baseline Per-Class Performance

| Class | Precision | Recall | F1 | Support | One-vs-rest AUROC | One-vs-rest AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| N | 0.0000 | 0.0000 | 0.0000 | 1,200 | 0.5966 | 0.3701 |
| S | 0.4441 | 0.9373 | 0.6026 | 542 | 0.9743 | 0.9167 |
| V | 0.5887 | 0.9733 | 0.7337 | 1,200 | 0.9105 | 0.7368 |
| F | 0.0010 | 0.0833 | 0.0021 | 12 | 0.4846 | 0.0029 |
| Q | 1.0000 | 0.0483 | 0.0922 | 1,200 | 0.8938 | 0.7805 |

The aggregate Macro-F1 conceals strongly asymmetric behavior. S and V are recalled well, whereas N is never recovered, F has near-zero utility with only a small number of test examples, and Q is recognized only when predicted with very high precision. This pattern is consistent with a decision boundary shaped by record-specific morphology and class composition rather than a uniform inability to separate all five labels.

## Final Decision and Replacement Gate

### Gate Definition

A Phase 2 or Phase 3 candidate could replace the frozen baseline only if it satisfied all of the following conditions: higher held-out Macro-F1, no major degradation in Macro-AUPRC, improvement in N or F recall, and stable behavior across repeated seeds. The gate is deliberately conjunctive because optimizing a single aggregate score could conceal catastrophic failure on the classes under investigation.

| Replacement-gate criterion | Required condition | Final status |
|---|---|---|
| Overall discrimination | Macro-F1 strictly higher than the frozen baseline | **Not satisfied by a fully acceptable candidate** |
| Ranking quality | No major Macro-AUPRC degradation | **Not satisfied by the λ=0.50 multi-task trade-off** |
| Minority recovery | Improved N or F recall | **F recall improved only for selected exploratory candidates; N recall remained 0** |
| Seed stability | Stable across repeated seeds | **No candidate established a superior stable profile** |
| Final decision | Replace the frozen checkpoint only if all conditions pass | **Gate failed; baseline retained** |

The λ=0.50 multi-task model exceeded the baseline on Macro-F1, but its Macro-AUPRC and Macro-AUROC were lower and its F-recall improvement remained partial. Therefore it is a trade-off candidate, not a clean replacement, and the retained model remains the scientifically defensible choice under the stated policy.

### Final Model and Candidate Comparison

| Candidate | Macro-F1 | Macro-AUPRC | Macro-AUROC | N recall | F recall | Gate interpretation |
|---|---:|---:|---:|---:|---:|---|
| Frozen RNN + RR + morphology | **0.4012** | **0.5963** | **0.7906** | 0.0000 | 0.0000 | Retained baseline |
| λ=0.25 multi-task symbol head | 0.3999 | 0.6005 | 0.7923 | 0.0000 | 0.1667 | F recall improved; Macro-F1 below baseline |
| λ=0.50 multi-task symbol head | **0.4186** | 0.5825 | 0.7709 | 0.0000 | 0.1667 | Higher F1 but degraded ranking metrics |
| Symbol-aware sampler | 0.2863 | 0.5524 | 0.7784 | 0.0000 | 0.0000 | Failed overall and minority gate |
| Symbol-aware loss | 0.1823 | 0.4699 | 0.7204 | 0.0017 | 0.0000 | Failed overall and minority gate |
| Domain-adversarial CNN/RNN | 0.1272 | 0.2875 | 0.5995 | 0.0025 | 0.0000 | Severe degradation |

The decision is not that the baseline solves the task; it does not solve N or F recall. The decision is that the tested alternatives did not provide sufficient evidence of a reliable Pareto improvement under a locked test set, explicit minority requirements, and seed-aware selection.

## Experimental Evidence and Root Cause Analysis

### Representation Ablation: Waveform, RR, and Handcrafted Morphology

| Representation | Macro-F1 | Macro-AUPRC | Macro-AUROC | Parameters | N recall | F recall |
|---|---:|---:|---:|---:|---:|---:|
| Waveform only | 0.2665 | 0.5532 | 0.7752 | 8,133 | 0.0000 | 0.0000 |
| Waveform + RR | 0.1886 | 0.5334 | 0.7493 | 8,261 | 0.0000 | 0.0000 |
| Waveform + RR + handcrafted morphology | **0.4012** | **0.5963** | **0.7906** | 8,517 | 0.0000 | 0.0000 |

The largest Phase 1 gain came from adding handcrafted morphology to waveform and RR inputs, indicating that compact morphology descriptors supplied information not reliably extracted by the small recurrent encoder. Conversely, adding RR features without the morphology block reduced performance, suggesting that the timing features alone were insufficient and may have increased sensitivity to split-specific temporal patterns. The representation improvement therefore strengthened aggregate discrimination without resolving the cross-record N/F failure mode.

### Architecture Benchmark

| Architecture | Macro-F1 | Macro-AUPRC | Macro-AUROC | Balanced accuracy | Parameters |
|---|---:|---:|---:|---:|---:|
| RNN | **0.2665** | **0.5532** | **0.7752** | 0.3790 | 8,133 |
| LSTM | 0.2363 | 0.4304 | 0.6885 | 0.3901 | 14,469 |
| GRU | 0.2572 | 0.5031 | 0.7397 | 0.3824 | 12,357 |
| Bi-LSTM | 0.2332 | 0.4198 | 0.6837 | 0.3481 | 23,077 |
| Bi-LSTM + temporal attention | 0.2690 | 0.5471 | **0.7953** | 0.3851 | 27,302 |
| CNN + Bi-LSTM + temporal attention | 0.1991 | 0.3189 | 0.7073 | 0.3022 | 24,198 |

The deeper recurrent and convolutional variants did not translate additional capacity into Macro-F1 gains. The attention model produced the highest AUROC in this isolated architecture run, but its Macro-F1 remained close to the waveform-only RNN and did not improve N or F recall. These results argue against treating architectural complexity as a substitute for record-diverse representation learning.

### Temporal Context and Window Alignment

| Context or alignment | Window configuration | Macro-F1 | Macro-AUPRC | Macro-AUROC | N recall | F recall |
|---|---|---:|---:|---:|---:|---:|
| 4-beat context | 90/90 | 0.2353 | 0.5350 | 0.7704 | 0.0000 | 0.0000 |
| 8-beat context | 90/90 | **0.2665** | **0.5532** | **0.7752** | 0.0000 | 0.0000 |
| 16-beat context | 90/90 | 0.2621 | 0.5372 | 0.7652 | 0.0000 | 0.0000 |
| 32-beat context | 90/90 | 0.2596 | 0.5426 | 0.7637 | 0.0000 | 0.0000 |
| 60/120 alignment | 180 samples | 0.4051 | 0.6442 | 0.8233 | 0.0075 | 0.0000 |
| 90/90 alignment | 180 samples | **0.4311** | 0.5970 | **0.8424** | 0.0000 | 0.0833 |
| 120/60 alignment | 180 samples | 0.2044 | 0.5079 | 0.7151 | 0.0008 | 0.0000 |

The temporal-context sweep suggests diminishing returns beyond the retained context length: longer histories did not recover the failed classes and slightly reduced aggregate performance in the waveform-only control. Window alignment mattered more than context length in the controlled Phase 2 run; the 120/60 configuration was substantially worse, while 90/90 provided the best overall balance. The isolated F-recall increase under 90/90 did not establish a replacement because the experiment used a distinct Phase 2 control configuration and did not meet the full gate against the frozen checkpoint.

### Learned Morphology Encoders

| Learned morphology model | Macro-F1 | Macro-AUPRC | Macro-AUROC | N recall | F recall | Validation Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| CNN morphology + RNN | 0.1217 | 0.2913 | 0.6120 | 0.0092 | 0.3333 | 0.3576 |
| CNN morphology + Bi-LSTM | 0.1351 | 0.2645 | 0.5579 | 0.0000 | 0.3333 | 0.3199 |
| CNN morphology + Bi-LSTM + attention | 0.1316 | 0.2787 | 0.6074 | 0.0000 | 0.4167 | 0.3138 |

All learned morphology variants underperformed the handcrafted morphology representation on the locked test records. The largest validation Macro-F1 belonged to the CNN morphology + RNN candidate, yet its held-out Macro-F1 was much lower, demonstrating that validation ranking did not transfer across records. The higher F-recall values are not sufficient evidence of improvement because they occur alongside large aggregate and ranking-metric losses and are based on very small F support.

### Class-Weighting, Sampling, and Augmentation

| Intervention | Macro-F1 | Macro-AUPRC | Macro-AUROC | N recall | F recall |
|---|---:|---:|---:|---:|---:|
| No augmentation | 0.4311 | 0.5970 | 0.8424 | 0.0000 | 0.0833 |
| Generic train-only perturbation | 0.4377 | 0.5986 | 0.8441 | 0.0000 | 0.0833 |
| Morphology-aware train-only perturbation | 0.4402 | 0.6030 | 0.8521 | 0.0000 | 0.0833 |
| Symbol-aware sampler | 0.2863 | 0.5524 | 0.7784 | 0.0000 | 0.0000 |
| Symbol-aware loss | 0.1823 | 0.4699 | 0.7204 | 0.0017 | 0.0000 |

The train-only augmentation comparison produced higher scores within its Phase 2 control run, but it did not establish a validated improvement over the frozen baseline under the project’s final selection protocol, so it was not promoted. The symbol-aware sampler and loss provide the sharper causal evidence: reweighting original-symbol combinations damaged overall performance rather than recovering N or F, consistent with over-emphasizing rare combinations that are not separable by the current representation. This rejects the hypothesis that frequency correction alone is sufficient.

### Hierarchical Classification

| Classifier | Macro-F1 | Macro-AUPRC | Macro-AUROC | Balanced accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|
| Flat five-class RNN control | 0.2665 | 0.5532 | 0.7752 | 0.3790 | 0.3960 |
| Hierarchical N-vs-non-N then S/V/F/Q | 0.3963 | 0.5357 | 0.7745 | 0.4855 | 0.4863 |

The hierarchical decomposition improved balanced accuracy and accuracy relative to the waveform-only flat control but reduced Macro-AUPRC and Macro-AUROC and did not provide evidence of reliable N/F recovery. The result suggests that the first-stage N decision may alter aggregate class balance without resolving the underlying morphology/domain ambiguity, so hierarchy alone is not a sufficient remedy.

### Multi-Task Original-Symbol Supervision

| Symbol-supervision weight λ | Macro-F1 | Macro-AUPRC | Macro-AUROC | Balanced accuracy | N recall | F recall | Validation Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.3937 | 0.5804 | 0.7889 | 0.4819 | 0.0000 | 0.0000 | 0.1699 |
| 0.25 | 0.3999 | **0.6005** | **0.7923** | 0.5076 | 0.0000 | 0.1667 | 0.1694 |
| 0.50 | **0.4186** | 0.5825 | 0.7709 | **0.5190** | 0.0000 | 0.1667 | 0.1657 |

The multi-task head is the strongest evidence that auxiliary original-symbol supervision can alter the minority decision boundary: F recall increased from zero for selected settings. The optimized objective was

$$
\mathcal{L}_{MT} = \mathcal{L}_{CE}(y, \hat{y}) + \lambda \mathcal{L}_{CE}(z, \hat{z}),
$$

where $y$ is the mapped five-class target, $\hat{y}$ is the primary classifier output, $z$ is the original-symbol target, and $\hat{z}$ is the auxiliary symbol prediction. However, λ=0.25 remained below the baseline on Macro-F1, while λ=0.50 increased Macro-F1 but degraded both ranking metrics, demonstrating a precision–recall and ranking trade-off rather than a clean win. N recall remained zero across all settings, indicating that original-symbol supervision did not resolve the dominant N failure.

### Domain-Adversarial Representation Learning

| Representation strategy | Macro-F1 | Macro-AUPRC | Macro-AUROC | N recall | F recall |
|---|---:|---:|---:|---:|---:|
| Handcrafted morphology baseline | **0.4012** | **0.5963** | **0.7906** | 0.0000 | 0.0000 |
| CNN morphology + original-symbol loss | 0.1823 | 0.4699 | 0.7204 | 0.0017 | 0.0000 |
| CNN/RNN domain-adversarial representation | 0.1272 | 0.2875 | 0.5995 | 0.0025 | 0.0000 |

The domain-adversarial model failed to preserve task performance while attempting to suppress record-domain information. With a gradient-reversal layer, the intended objective can be written as

$$
\mathcal{L}_{DANN} = \mathcal{L}_{task}(y, \hat{y}) - \gamma \mathcal{L}_{domain}(d, \hat{d}),
$$

where $d$ denotes the record-domain label, $\hat{d}$ is the domain prediction, and $\gamma$ controls the adversarial pressure. A plausible mechanism is **gradient interference** between the reversal-layer domain objective and the small recurrent task representation under the short training schedule; this is an interpretation, not a directly isolated causal measurement. The result demonstrates that domain invariance must be introduced with capacity, optimization, and domain-support controls rather than assumed to improve cross-record generalization automatically.

### Original-Symbol Shift and N-Failure Diagnosis

| Original symbol | Mapped class | Train | Validation | Test | Test error rate | Test recall |
|---|---|---:|---:|---:|---:|---:|
| `/` | Q | 1,168 | 0 | 758 | 0.1517 | 0.8483 |
| `A` | S | 1,068 | 119 | 513 | 0.0195 | 0.9805 |
| `F` | F | 767 | 23 | 12 | 1.0000 | 0.0000 |
| `J` | S | 4 | 0 | 29 | 0.5862 | 0.4138 |
| `N` | N | 1,200 | 0 | 1,200 | 1.0000 | 0.0000 |
| `Q` | Q | 2 | 4 | 18 | 0.9444 | 0.0556 |
| `V` | V | 1,200 | 1,094 | 1,200 | 0.1783 | 0.8217 |
| `f` | Q | 30 | 0 | 424 | 0.6651 | 0.3349 |

The mapped N count is not absent from training, yet original-symbol N has no validation support and test N beats are completely misclassified, frequently as V in the N-failure gallery. The F estimate is intrinsically unstable because the held-out support is very small; a single additional correct or incorrect prediction materially changes recall. These observations make cross-record/original-symbol composition shift the leading root-cause hypothesis, with representation ambiguity as a coupled limitation rather than simple mapped-class under-sampling.

From a biomedical signal-processing perspective, the N-to-V confusion is plausible under an OOD shift in patient-specific conduction morphology. A normal sinus beat is not guaranteed to produce a narrow, stereotyped QRS complex: when conduction encounters a refractory bundle branch, the impulse can propagate aberrantly, producing a bundle-branch-block-like complex with increased width, slurred depolarization, altered terminal forces, and beat-to-beat morphology that is atypical for that patient’s learned baseline. A ventricular ectopic beat, by contrast, also commonly manifests as a broad and morphologically unusual QRS complex. In the feature space available to a model that has not seen the patient’s baseline or that specific conduction phenotype, the width, energy, slope, and QRS-proxy features of an aberrantly conducted normal beat can therefore lie closer to the V manifold than to the learned N manifold. This is a mechanistic interpretation of the observed error pattern, not a retrospective clinical adjudication of individual beats; the dataset annotations and waveform morphology would require expert review to establish the precise physiological cause for each error.

[Insert Figure 2: Record-level generalization heatmap, linking held-out record identity and original-symbol composition to per-class recall and Macro-F1.]

[Insert Figure 3: N-failure gallery, showing representative held-out N waveforms from records 108 and 104, their handcrafted feature profiles, predicted V probabilities, and corresponding annotation symbols.]

### Baseline Confusion Matrix: Raw Counts

| True \\ Predicted | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| N | 0 | 504 | 635 | 61 | 0 |
| S | 6 | 508 | 4 | 24 | 0 |
| V | 0 | 32 | 1,168 | 0 | 0 |
| F | 0 | 6 | 5 | 1 | 0 |
| Q | 0 | 94 | 172 | 876 | 58 |

The raw matrix shows that N is distributed primarily into S and V, while Q is distributed heavily into S and F despite high Q precision. F is almost entirely absorbed by S and V. This is not the signature of one uniformly weak classifier; it is a structured set of class confusions aligned with record-specific morphology and the original-symbol mixture.

[Insert Figure 4: Baseline raw and row-normalized confusion matrices, displayed side by side with class labels N/S/V/F/Q.]

### Baseline Confusion Matrix: Row-Normalized

| True \\ Predicted | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| N | 0.0000 | 0.4200 | 0.5292 | 0.0508 | 0.0000 |
| S | 0.0111 | 0.9373 | 0.0074 | 0.0443 | 0.0000 |
| V | 0.0000 | 0.0267 | 0.9733 | 0.0000 | 0.0000 |
| F | 0.0000 | 0.5000 | 0.4167 | 0.0833 | 0.0000 |
| Q | 0.0000 | 0.0783 | 0.1433 | 0.7300 | 0.0483 |

The normalized matrix confirms perfect N failure and near-total F failure while showing strong recall for S and V. It also reveals that Q recall is low even though Q precision is perfect, implying a conservative decision rule that avoids false Q predictions by routing many Q beats toward S or F.

## Generalization, Calibration, and Robustness

### Record-Level Generalization

| Held-out record | Macro-F1 | Macro-AUROC | Macro-AUPRC | N recall | F recall | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| 108 | 0.0083 | Not reported in source row | Not reported in source row | 0.0000 | Not reported in source row | Severe N-dominated failure |
| 104 | 0.1600 | Not reported in source row | Not reported in source row | Not reported in source row | Not reported in source row | Strong Q recall but poor aggregate performance |
| 200 | 0.3337 | Not reported in source row | Not reported in source row | Not reported in source row | Not reported in source row | Strongest highlighted non-empty record |
| 122 | No modeling rows | No modeling rows | No modeling rows | No modeling rows | No modeling rows | Explicit empty-record handling |

Performance varies substantially across held-out records, with record 108 near failure and record 200 materially stronger. This heterogeneity is difficult to reconcile with a pure global class-imbalance explanation and is more consistent with domain or symbol-composition shift. Record 122 is retained explicitly as empty after documented filtering rather than silently removed, preserving the integrity of the locked record list.

[Insert Figure 5: Record-level heatmap of Macro-F1, Macro-AUPRC, Macro-AUROC, and per-class recall across the ten locked test records.]

### Five-Seed Stability

| Model family | Macro-F1 mean | Macro-F1 std | Macro-AUPRC mean | Macro-AUPRC std | Macro-AUROC mean | Macro-AUROC std | F recall mean | F recall std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Retained RNN + RR + morphology | **0.4087** | 0.0174 | 0.5909 | 0.0211 | 0.7972 | 0.0235 | 0.0167 | 0.0373 |
| Symbol-aware-loss family | 0.1447 | Not reported in source table | Not reported in source table | Not reported in source table | Not reported in source table | Not reported in source table | Not reported in source table | Not reported in source table |

The retained family is reasonably stable on aggregate metrics relative to the failed symbol-aware family, but its minority behavior remains unacceptable for the stated research objective. N recall remained zero in the reported retained-family rows, while F recall was near zero on average and highly variable relative to its small support. Seed stability therefore supports retaining the baseline as a reproducible control, not claiming that it solved the generalization problem.

### Bootstrap Confidence Intervals

| Metric | Point estimate | Bootstrap resamples | 95% lower bound | 95% upper bound |
|---|---:|---:|---:|---:|
| Macro-F1 | 0.4012 | 500 | 0.3924 | 0.4094 |
| Macro-AUROC | 0.7906 | 500 | 0.7522 | 0.8222 |
| Macro-AUPRC | 0.5963 | 500 | 0.5895 | 0.6033 |

The bootstrap intervals quantify uncertainty in the locked test estimate but do not establish clinical validity or population-level transportability. They also do not convert the λ=0.50 multi-task result into a statistically confirmed replacement because candidate-specific paired uncertainty and gate-complete multi-seed evidence were not established for that candidate.

### Calibration Trade-Off

| Model/calibration state | Temperature | ECE | Brier score | Macro-F1 | Macro-AUPRC | Macro-AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Frozen baseline, uncalibrated | 1.0 | 0.2216 | 0.6703 | 0.4012 | 0.5963 | 0.7906 |
| Phase 3 final, uncalibrated | 1.0 | 0.2216 | 0.6703 | 0.4012 | 0.5963 | 0.7906 |
| Frozen baseline, temperature-scaled | 3.0 | **0.1447** | **0.6519** | 0.4012 | 0.5825 | 0.7607 |

Temperature scaling improved calibration error and Brier score while reducing ranking metrics. This is an operational trade-off: the scaled probabilities may be more reliable as confidence estimates, but they are not preferable when ranking quality is the primary objective. Temperature scaling should therefore remain an explicit downstream calibration option rather than being silently folded into the final classifier.

### Controlled Robustness

| Condition | Macro-F1 | Macro-AUPRC | Macro-AUROC | Macro-F1 delta from clean |
|---|---:|---:|---:|---:|
| Clean | 0.4012 | 0.5963 | 0.7906 | 0.0000 |
| Gaussian noise | 0.4026 | 0.5963 | 0.7904 | +0.0014 |
| Baseline wander | 0.4231 | 0.5933 | 0.7872 | +0.0219 |
| Amplitude scaling | 0.3869 | 0.5970 | 0.7926 | -0.0143 |
| Missing samples | 0.4023 | 0.4289 | 0.7263 | +0.0011 |

The model is comparatively stable under the tested Gaussian-noise and baseline-wander perturbations, while amplitude scaling causes a modest Macro-F1 decrease. Missing samples have a more serious effect on ranking quality: Macro-AUPRC and Macro-AUROC decline sharply even though Macro-F1 remains nearly unchanged, illustrating why robustness must be evaluated with both thresholded and ranking metrics. These controlled perturbations are not evidence of noise immunity or deployment readiness.

## Path Forward

### Hypothesis 1: Split-aware original-symbol and record-domain evaluation will identify recoverable versus irreducible failure modes

The next study should replace aggregate-only diagnostics with a pre-registered evaluation matrix crossing mapped class, original symbol, and held-out record. The hypothesis is that N/F failures will partition into record-specific and symbol-specific subsets, allowing targeted representation learning to be evaluated on the strata where it is intended to help rather than on a single aggregate score.

| Proposed experiment | Required control | Primary endpoint | Decision rule |
|---|---|---|---|
| Leave-record-family-out evaluation with original-symbol stratification | Same preprocessing and label mapping | Stratum-level Macro-F1 and N/F recall | Retain only if gains persist across multiple record families |
| Per-symbol calibration and error decomposition | Locked test predictions | Symbol-level AUPRC and confidence error | Reject methods that improve recall only by destroying precision |
| Record-balanced validation | No test-set changes | Across-record variance of Macro-F1 | Prefer lower variance at matched mean performance |

This would directly test the leading root-cause hypothesis instead of treating the current test set as a homogeneous sample. It would also quantify whether a candidate improves generalization broadly or only exploits a favorable record composition.

### Hypothesis 2: Capacity-controlled domain generalization with stable auxiliary objectives will outperform adversarial reversal

The domain-adversarial collapse suggests that the current reversal objective was introduced too aggressively relative to model capacity and training duration. A controlled follow-up should compare gradient reversal against alternatives such as domain-specific batch normalization, invariant risk minimization proxies, supervised contrastive alignment, and gradual auxiliary-loss scheduling while matching parameter counts and compute budgets.

| Proposed intervention | Key control | Main failure risk to monitor |
|---|---|---|
| Gradual domain-loss ramp | Same encoder and task loss; sweep ramp schedules | Task-gradient suppression |
| Domain-specific normalization | Same classifier head and parameter budget | Memorization of record identity |
| Supervised contrastive alignment by mapped class | Same record-balanced batches | Collapse of clinically meaningful morphology proxies |
| Larger-capacity encoder control | Matched training duration and seed count | Mistaking capacity for invariance |

The hypothesis is not that invariance is universally beneficial; it is that the current failure may reflect optimization conflict rather than evidence against all domain-generalized representations. Every candidate should be required to preserve task ranking metrics before minority-recall gains are accepted.

### Hypothesis 3: Learn morphology from beat-centered self-supervision before supervised fine-tuning

The handcrafted morphology block is currently stronger than the learned CNN encoder, indicating that supervised learning from the available labels does not reliably discover the relevant local structure. A next phase should pretrain a beat-centered encoder using reconstruction, masked-sample prediction, or contrastive augmentations across records, then fine-tune under the same locked grouped split.

| Pretraining objective | Intended benefit | Required ablation |
|---|---|---|
| Masked waveform reconstruction | Preserve local morphology and temporal continuity | Random initialization with identical supervised budget |
| Cross-record contrastive learning | Reduce record-specific style dependence | Within-record versus cross-record positives |
| Rhythm-context prediction | Encode temporal context without label leakage | 4-, 8-, and 16-beat context controls |

This approach would test whether the representation bottleneck can be addressed without increasing label reweighting pressure. It should be evaluated with the existing handcrafted model as a strong control, not assumed to be superior because it uses self-supervision.

## Limitations and Scientific Scope

The study uses a single locked grouped split and a finite set of held-out records. The small F test support makes F recall unstable, while validation support for some mapped classes is also sparse. The bootstrap intervals quantify uncertainty conditional on this test composition; they do not estimate performance on future institutions, devices, populations, or clinical workflows.

The dataset is an annotated research benchmark rather than evidence of clinical utility [1]. The work reports algorithmic classification behavior only and makes no claim of diagnostic capability, clinical superiority, or deployment safety. The final model should be treated as an interpretable research prototype whose principal value is the transparent characterization of its strengths and failure modes.

## Conclusion

Across Phases 1–3, the strongest measured model remained the compact RNN combining waveform, RR, and handcrafted morphology features. The experiments establish that the unresolved N/F failures are not solved by architecture scaling, longer temporal context, symbol-aware reweighting, hierarchical decomposition, or direct adversarial domain suppression under the tested conditions. Multi-task original-symbol supervision is the most promising exploratory direction because it produced an F-recall increase, but its aggregate and ranking trade-offs prevent replacement at present.

The scientifically correct outcome is therefore **baseline retention with explicit failure documentation**, not forced model replacement. The next phase should focus on record- and symbol-stratified generalization, capacity-controlled domain alignment, and self-supervised morphology representation learning, with the replacement gate preserved unchanged.

## References

[1]: https://physionet.org/content/mitdb/1.0.0/ "MIT-BIH Arrhythmia Database, PhysioNet"

[2]: https://www.kaggle.com/datasets/abdallahwagih/mit-bih-arrhythmia-database "MIT-BIH Arrhythmia Database Kaggle mirror"

[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8181174/ "Interpatient ECG Heartbeat Classification with an Adversarial Network"

## Appendix: Computational Budget and Reproducibility

The following appendix separates **recorded implementation settings** from an **assumed reference execution profile**. This distinction is necessary because publication reproducibility requires that reported metrics correspond to the committed code and artifacts, while a paper may also specify a standardized hardware profile for independent reruns.

### Recorded implementation contract

| Component | Recorded setting |
|---|---|
| Framework | PyTorch |
| Data loader batch size | 128 by default in the experiment harness |
| Training schedule | 3 epochs by default for the benchmark harness |
| Optimizer in committed harness | `torch.optim.Adam` |
| Primary selection metric | Validation Macro-F1, with Macro-AUPRC as tie-break context |
| Random seeds | 42, 43, 44, 45, 46 for the reported seed study |
| Bootstrap resamples | 500 |
| Test examples | 4,154 |
| Test-set hash | `6866fbcc631936f84892df70dee14f2193ebb731a9ea30d48eaa7268ef282be8` |
| Final checkpoint | `checkpoints/RNN_rr_morphology.pt` |
| Split manifest | `research_results/configs/split_lock.json` |

The recorded optimizer is Adam rather than AdamW; this is reported explicitly to prevent an apparently minor optimizer substitution from being mistaken for an exact reproduction. All quantitative results in the manuscript refer to the committed experiment artifacts and the locked test split, not to an unrecorded reimplementation.

### Assumed reference execution profile

| Component | Reference assumption |
|---|---|
| GPU | 1 × NVIDIA RTX 3080, 10 GB class device |
| CPU | Contemporary multi-core x86-64 CPU |
| Host memory | At least 16 GB RAM |
| Python | Python 3.12-compatible environment |
| Framework | PyTorch |
| Proposed optimizer for a future standardized rerun | AdamW |
| Proposed batch size for a future standardized rerun | 128 |
| Precision | FP32 unless a controlled mixed-precision ablation is explicitly added |
| Determinism | Fixed split seed, fixed experiment seed, recorded package versions, and persisted predictions |

The RTX 3080 and AdamW entries are reference assumptions for a future standardized rerun, not claims about the hardware or optimizer used to generate the present metrics. A future paper revision should either rerun every result under that standardized profile or remove the assumed profile and report only the exact execution environment captured by the repository.

### Artifact-level reproducibility

| Artifact | Function |
|---|---|
| `research_results/configs/split_lock.json` | Locks record and subject partitions and leakage checks |
| `research_results/configs/experiment_manifest.json` | Records seeds, contexts, architectures, perturbations, and bootstrap size |
| `research_results/final_config.json` | Records final-model selection and checkpoint contract |
| `research_results/phase3_metrics.json` | Records final Phase 3 gate outcome |
| `research_results/phase3_bootstrap_results.csv` | Records bootstrap point estimates and confidence intervals |
| `research_results/phase3_calibration.csv` | Records uncalibrated and temperature-scaled calibration results |
| `research_results/phase3_robustness.csv` | Records controlled corruption outcomes |
| `research_results/vigil_research_report.md` | This manuscript and its quantitative synthesis |

Exact reproduction requires preserving the locked test set, preprocessing metadata, original-symbol mapping, model checkpoint, random seeds, and prediction files. Reproducing only the aggregate scores without these artifacts would not reproduce the OOD evaluation itself.
