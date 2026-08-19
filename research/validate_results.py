from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_results'
required = [
    'class_counts.csv', 'per_class_metrics.csv', 'confusion_matrix.png',
    'ablation_results.csv', 'context_results.csv', 'augmentation_results.csv',
    'architecture_results.csv', 'robustness_results.csv', 'seed_results.csv',
    'bootstrap_confidence_intervals.csv', 'final_report.md',
    'diagnosis/error_by_original_symbol.csv', 'diagnosis/confusion_matrix_raw.csv',
    'diagnosis/confusion_matrix_normalized.csv', 'diagnosis/split_integrity.json',
    'diagnosis/mapping_verification.json', 'configs/experiment_manifest.json',
    'checkpoints/RNN.pt', 'checkpoints/LSTM.pt', 'checkpoints/GRU.pt',
    'checkpoints/BiLSTM.pt',     'checkpoints/BiLSTM_Attention.pt', 'checkpoints/CNN_BiLSTM_Attention.pt', 'checkpoints/hierarchical.pt',
    'morphology_feature_ablation.csv', 'learned_morphology_results.csv', 'symbol_shift_analysis.csv', 'record_generalization.csv',
    'morphology_augmentation_results.csv', 'window_alignment_results.csv', 'n_failure_analysis.csv', 'final_model_comparison.csv',
    'final_seed_results.csv', 'final_robustness_results.csv', 'final_model.pt', 'final_config.json', 'feature_schema.json',
    'preprocessing_config.json', 'final_metrics.json', 'phase2_report.md', 'phase2/plots/symbol_shift_composition.png',
    'phase2/plots/record_generalization.png', 'phase2/plots/n_failure_gallery.png', 'phase2/checkpoints/baseline_rr_morphology.pt',
    'phase3_symbol_balance.csv', 'phase3_multitask.csv', 'phase3_domain_adversarial.csv', 'phase3_representation_ablation.csv',
    'phase3_record_generalization.csv', 'phase3_n_failure_analysis.csv', 'phase3_seed_results.csv', 'phase3_bootstrap_results.csv',
    'phase3_calibration.csv', 'phase3_robustness.csv', 'phase3_best_model.pt', 'phase3_best_config.json', 'phase3_metrics.json',
    'phase3_report.md', 'phase3/plots/record_generalization_phase3.png', 'phase3/baseline_metrics.json',
    'phase3/baseline_predictions.csv', 'phase3/baseline_checkpoint.pt'
]
missing = [p for p in required if not (OUT / p).exists()]
if missing: raise AssertionError(f'Missing artifacts: {missing}')

with open(OUT/'diagnosis/split_integrity.json') as f: split = json.load(f)
with open(OUT/'diagnosis/mapping_verification.json') as f: mapping = json.load(f)
with open(OUT/'configs/experiment_manifest.json') as f: manifest = json.load(f)
assert split['no_record_leakage'] and split['no_subject_leakage'], split
assert mapping['exact'], mapping
assert len(split['test_records']) == 10 and len(split['validation_records']) == 7 and len(split['train_records']) == 31
assert manifest['test_examples'] == 4154
assert manifest['test_hash']

class_counts = pd.read_csv(OUT/'class_counts.csv')
assert set(class_counts[class_counts.level=='five_class'].label) == {'N','S','V','F','Q'}
for name, cols in {
    'ablation_results.csv': ['MacroF1','MacroAUPRC','MacroAUROC'],
    'context_results.csv': ['context','MacroF1'],
    'augmentation_results.csv': ['augmentation','MacroF1'],
    'architecture_results.csv': ['architecture','MacroF1'],
    'robustness_results.csv': ['corruption','MacroF1'],
    'seed_results.csv': ['seed','MacroF1'],
    'bootstrap_confidence_intervals.csv': ['metric','estimate','ci_lower_95','ci_upper_95'],
}.items():
    df = pd.read_csv(OUT/name)
    assert set(cols).issubset(df.columns), (name, df.columns.tolist())
    assert len(df) > 0

seed = pd.read_csv(OUT/'seed_results.csv')
assert seed.groupby(seed['experiment'].str.rsplit('_seed_', n=1).str[0]).size().min() >= 5
boot = pd.read_csv(OUT/'bootstrap_confidence_intervals.csv')
assert set(boot.metric) == {'MacroF1','MacroAUROC'}
assert (boot.ci_lower_95 <= boot.estimate).all() and (boot.estimate <= boot.ci_upper_95).all()
phase2_final = json.loads((OUT/'final_metrics.json').read_text())
assert phase2_final['model'] == 'RNN + RR + morphology'
assert abs(float(phase2_final['MacroF1']) - 0.40123549379362033) < 1e-9
assert len(pd.read_csv(OUT/'record_generalization.csv')) == 10
assert len(pd.read_csv(OUT/'n_failure_analysis.csv')) == 100
assert len(pd.read_csv(OUT/'final_seed_results.csv')) == 5
phase3_final = json.loads((OUT/'phase3_metrics.json').read_text())
assert phase3_final['replacement_gate_passed'] is False
assert phase3_final['model'] == 'RNN + RR + morphology'
assert abs(float(phase3_final['MacroF1']) - 0.40123549379362033) < 1e-9
p3_boot = pd.read_csv(OUT/'phase3_bootstrap_results.csv')
assert set(p3_boot.metric) == {'MacroF1','MacroAUROC','MacroAUPRC'} and len(p3_boot) == 3
assert (p3_boot.ci_lower_95 <= p3_boot.estimate).all() and (p3_boot.estimate <= p3_boot.ci_upper_95).all()
p3_seed = pd.read_csv(OUT/'phase3_seed_results.csv')
assert len(p3_seed) == 10 and set(p3_seed.model) == {'frozen_baseline','phase3_symbol_loss'}
assert len(pd.read_csv(OUT/'phase3_record_generalization.csv')) >= 9
assert len(pd.read_csv(OUT/'phase3_n_failure_analysis.csv')) == 100
cm_raw = pd.read_csv(OUT/'diagnosis/confusion_matrix_raw.csv', index_col=0)
cm_norm = pd.read_csv(OUT/'diagnosis/confusion_matrix_normalized.csv', index_col=0)
assert cm_raw.shape == (5,5) and cm_norm.shape == (5,5)
print(json.dumps({'status':'ok','required_artifacts':len(required),'test_examples':manifest['test_examples'],'mapping_exact':mapping['exact'],'no_record_leakage':split['no_record_leakage'],'no_subject_leakage':split['no_subject_leakage'],'seed_families':seed['experiment'].str.rsplit('_seed_',n=1).str[0].value_counts().to_dict(),'bootstrap_metrics':sorted(boot.metric.tolist())}, indent=2))
