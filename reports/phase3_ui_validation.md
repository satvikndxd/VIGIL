Phase 3 UI validation:

The live SQL-workbench dashboard renders the existing ECG analysis shell plus distinct Phase 2 and Phase 3 sections. Phase 3 shows the actual decision `Frozen baseline retained`, candidate explored `domain_adversarial`, Macro-F1 0.401, Macro-AUPRC 0.596, Macro-AUROC 0.791, and N/F recall 0.0%/0.0%.

The Phase 3 representation table shows handcrafted morphology at 0.401 Macro-F1, multi-task at 0.400, symbol-aware loss at 0.182, learned CNN morphology at 0.122, and domain-adversarial at 0.127. Calibration displays the measured ECE/Brier trade-off and robustness displays baseline versus final rows. The dashboard remains non-fabricated and distinguishes the retained final model from explored candidates.
