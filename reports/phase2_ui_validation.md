Phase 2 UI validation:

The live VIGIL dashboard still renders the SQL-workbench ECG experience and now includes a real Phase 2 Research Results panel. It shows the retained final model `RNN + RR + morphology`, Macro-F1 0.401, Macro-AUROC 0.791, Macro-AUPRC 0.596, and N/F recall at 0.0% without fabricated improvement.

The panel renders actual per-record Macro-F1 bars for the ten locked test records, including an empty/unknown row for record 122, the original-symbol shift table with train/test counts and symbol error/recall, the minority-class panel, and clean/corruption robustness results. The panel is compact and consistent with the query-result workbench visual language.
