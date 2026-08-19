# VIGIL Phase 3 — Domain-Generalized ECG Representation Learning

## Final decision

Phase 3 tested symbol-aware sampling, original-symbol-aware loss weighting, multi-task symbol supervision, and a small domain-adversarial CNN/RNN representation on the **same locked 31/7/10 record split** and the same five-class target N/S/V/F/Q. The final result is a measured negative result: **no Phase 3 candidate passed the replacement gate**, so the frozen RNN + RR + handcrafted morphology model remains the final model.

| Model | Macro-F1 | Macro-AUPRC | Macro-AUROC | N recall | F recall |
|---|---:|---:|---:|---:|---:|
| Frozen RNN + RR + morphology | 0.4012 | 0.5963 | 0.7906 | 0.0000 | 0.0000 |
| Final Phase 3 model | 0.4012 | 0.5963 | 0.7906 | 0.0000 | 0.0000 |

The candidate selected for Phase 3 comparison by validation Macro-F1 was **domain_adversarial**, but its held-out Macro-F1 was 0.1272 and its replacement gate was false. It is not deployed as the final model.

## Experimental evidence

### Symbol-aware training

| Experiment | Macro-F1 | Macro-AUPRC | Macro-AUROC | N recall | F recall |
|---|---:|---:|---:|---:|---:|
| experiment                       |   MacroF1 |   MacroAUPRC |   MacroAUROC |   N_recall |   F_recall |
|:---------------------------------|----------:|-------------:|-------------:|-----------:|-----------:|
| current_class_balanced           |    0.4012 |       0.5963 |       0.7906 |     0      |          0 |
| original_symbol_balanced_sampler |    0.2863 |       0.5524 |       0.7784 |     0      |          0 |
| original_symbol_aware_loss       |    0.1823 |       0.4699 |       0.7204 |     0.0017 |          0 |

Balancing original symbols within their mapped classes reduced held-out Macro-F1 from 0.4012 to 0.2863, while the symbol-aware loss weighting reduced it to 0.1823. Neither recovered N or F recall. The result does not support replacing mapped-class weighting with symbol balancing under this split and training budget.

### Multi-task symbol supervision

The shared encoder with an auxiliary original-symbol head was trained with λ values 0.1, 0.25, and 0.5:

|   lambda |   MacroF1 |   MacroAUPRC |   MacroAUROC |   N_recall |   F_recall |   validation_MacroF1 |
|---------:|----------:|-------------:|-------------:|-----------:|-----------:|---------------------:|
|     0.1  |    0.3937 |       0.5804 |       0.7889 |          0 |     0      |               0.1699 |
|     0.25 |    0.3999 |       0.6005 |       0.7923 |          0 |     0.1667 |               0.1694 |
|     0.5  |    0.4186 |       0.5825 |       0.7709 |          0 |     0.1667 |               0.1657 |

λ = 0.5 produced F recall 0.1667, but Macro-AUPRC fell to 0.5825 and Macro-AUROC fell to 0.7709. λ = 0.25 also produced F recall 0.1667 with Macro-F1 0.3999, still below the frozen baseline. Because the F support is only 12 held-out sequences, the isolated recall increase is not treated as a stable deployment improvement.

### Domain-adversarial representation learning

The small CNN/RNN gradient-reversal model used record/domain labels only during training and did not expose record IDs at inference. It reached held-out Macro-F1 0.1272, Macro-AUPRC 0.2875, Macro-AUROC 0.5995, N recall 0.0025, and F recall 0.0000. It therefore did not reduce the cross-record failure in this experiment.

### Representation comparison

| representation                  |   MacroF1 |   MacroAUPRC |   MacroAUROC |   N_recall |   F_recall |
|:--------------------------------|----------:|-------------:|-------------:|-----------:|-----------:|
| waveform_only                   |    0.2665 |       0.5532 |       0.7752 |   nan      |   nan      |
| handcrafted_morphology          |    0.4012 |       0.5963 |       0.7906 |   nan      |   nan      |
| cnn_morphology_embedding        |    0.1217 |       0.2913 |       0.612  |   nan      |   nan      |
| cnn_rr_morph_symbol_loss        |    0.1823 |       0.4699 |       0.7204 |     0.0017 |     0      |
| cnn_rr_morph_multitask          |    0.3999 |       0.6005 |       0.7923 |     0      |     0.1667 |
| cnn_rr_morph_domain_adversarial |    0.1272 |       0.2875 |       0.5995 |     0.0025 |     0      |

The handcrafted morphology representation remains the strongest measured representation. The Phase 3 learned/domain-invariant encoders did not demonstrate better held-out generalization.

## Generalization and N/F analysis

The Phase 3 per-record export is `phase3_record_generalization.csv` and includes Macro-F1, Macro-AUROC, Macro-AUPRC, N/S/V/F/Q recall, and original-symbol composition for each non-empty held-out record. The visualization is `phase3/plots/record_generalization_phase3.png`. The record-level gap remains visible rather than being hidden by an aggregate metric.

The Phase 3 N-failure export is `phase3_n_failure_analysis.csv`. It repeats the existing 100 representative baseline N failures and adds the Phase 3 top class and confidence. The report does not claim success because the retained final N recall is still 0.000. F remains a rare-class stress test with only 12 held-out examples; the final retained model has F precision/recall/F1 at the values recorded in the Phase 2 per-class audit and no Phase 3 intervention produced stable improvement.

## Calibration

| Model | Temperature | ECE | Brier | Macro-F1 | Macro-AUPRC | Macro-AUROC |
|---|---:|---:|---:|---:|---:|---:|
| model                              |   temperature |    ECE |   Brier |   MacroF1 |   MacroAUPRC |   MacroAUROC |
|:-----------------------------------|--------------:|-------:|--------:|----------:|-------------:|-------------:|
| frozen_baseline_test               |             1 | 0.2216 |  0.6703 |    0.4012 |       0.5963 |       0.7906 |
| phase3_final_test                  |             1 | 0.2216 |  0.6703 |    0.4012 |       0.5963 |       0.7906 |
| frozen_baseline_temperature_scaled |             3 | 0.1447 |  0.6519 |    0.4012 |       0.5825 |       0.7607 |

Temperature scaling was fit on validation data only. It reduced ECE from 0.2216 to 0.1447 and Brier from 0.6703 to 0.6519, but reduced Macro-AUPRC from 0.5963 to 0.5825 and Macro-AUROC from 0.7906 to 0.7607. Calibration is therefore an optional probability-quality trade-off, not a classification-performance improvement.

## Five-seed confirmation

The frozen baseline five-seed family produced:

|      |   MacroF1 |   MacroAUPRC |   MacroAUROC |   N_recall |   F_recall |
|:-----|----------:|-------------:|-------------:|-----------:|-----------:|
| mean |    0.4087 |       0.5909 |       0.7972 |          0 |     0.0167 |
| std  |    0.0174 |       0.0211 |       0.0235 |          0 |     0.0373 |

This is approximately Macro-F1 **0.4087 ± 0.0174**, Macro-AUPRC **0.5909 ± 0.0211**, and Macro-AUROC **0.7972 ± 0.0235**. N recall was 0.000 across all five seeds. F recall averaged **0.0167 ± 0.0373**, driven by one seed, and was not consistently recovered.

The Phase 3 symbol-loss candidate was substantially less stable and lower-performing across seeds. Its mean Macro-F1 was 0.1448 ± 0.0333, with mean N recall 0.0185 ± 0.0400 and F recall 0.0000 ± 0.0000.

## Bootstrap and robustness

The final retained model bootstrap intervals use 500 resamples:

| metric     |   estimate |   ci_lower_95 |   ci_upper_95 |   bootstrap_samples |
|:-----------|-----------:|--------------:|--------------:|--------------------:|
| MacroAUPRC |     0.5963 |        0.5895 |        0.6033 |                 500 |
| MacroAUROC |     0.7906 |        0.7522 |        0.8222 |                 500 |
| MacroF1    |     0.4012 |        0.3924 |        0.4094 |                 500 |

The Phase 3 robustness table retains clean, Gaussian noise, baseline wander, amplitude scaling, and missing-sample tests for both baseline and final output. Since the baseline was retained, the two blocks are identical. The outputs are in `phase3_robustness.csv`; no noise immunity is claimed.

## Answer to the final research question

> Can symbol-aware and domain-invariant representation learning reduce the cross-record generalization failure observed in VIGIL?

**Not in the measured Phase 3 experiments.** Symbol-aware sampling and loss weighting reduced aggregate performance. Multi-task symbol supervision produced an isolated F-recall increase at λ = 0.25/0.5 but did not improve Macro-F1 and was not stable enough to replace the baseline. The small domain-adversarial CNN/RNN representation underperformed substantially. The evidence continues to support representation and original-symbol/cross-record composition shift as the core limitation, but these Phase 3 interventions did not solve it.

The recommended model therefore remains **RNN + RR + handcrafted morphology**, with the exact locked split, final configuration, baseline checkpoint, Phase 3 candidate checkpoint, metrics, calibration, bootstrap, robustness, and dashboard/API artifacts preserved. This is a retrospective research conclusion and not a clinical claim.

## References

[1]: https://physionet.org/content/mitdb/1.0.0/ "MIT-BIH Arrhythmia Database on PhysioNet"
[2]: https://www.kaggle.com/datasets/abdallahwagih/mit-bih-arrhythmia-database "Kaggle MIT-BIH Arrhythmia Database mirror"
