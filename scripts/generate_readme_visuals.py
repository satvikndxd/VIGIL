from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
DOCS.mkdir(exist_ok=True)
METRICS = ROOT / 'ml/metrics'
PLOTS = ROOT / 'ml/plots'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 8})
blue, orange, ink, grid = '#315a91', '#e28b42', '#243242', '#d8dde1'

comparison = pd.read_csv(METRICS / 'model_comparison.csv')
comparison = comparison.rename(columns={'Unnamed: 0': 'Model'})
pred = pd.read_csv(ROOT / 'ml/predictions/test_predictions.csv')
serving = json.loads((METRICS / 'serving_benchmark.json').read_text())

fig = plt.figure(figsize=(13, 8), facecolor='#f5f6f7')
gs = GridSpec(2, 3, figure=fig, hspace=.42, wspace=.28)

ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor('white')
leader = comparison.sort_values('MacroF1')
ax.barh(leader['Model'], leader['MacroF1'], color=[orange if x == 'Logistic Regression' else blue for x in leader['Model']])
ax.set_title('Held-out Macro-F1 leaderboard', loc='left', color=ink, weight='bold'); ax.set_xlim(0, max(.5, leader.MacroF1.max() * 1.22)); ax.grid(axis='x', color=grid, linewidth=.6); ax.set_axisbelow(True)
for i, v in enumerate(leader['MacroF1']): ax.text(v + .01, i, f'{v:.3f}', va='center', color=ink, fontsize=7)
ax.spines[['top','right','left']].set_visible(False); ax.tick_params(axis='y', length=0, labelsize=7)

ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor('white')
metric_cols = ['MacroF1', 'AUROC_OVR', 'AUPRC_macro']
for col, color in zip(metric_cols, [blue, orange, '#6b7ea2']): ax.plot(comparison['Model'], comparison[col], marker='o', linewidth=1.5, label=col, color=color)
ax.set_title('Model comparison by metric', loc='left', color=ink, weight='bold'); ax.set_ylim(0, 1); ax.grid(axis='y', color=grid, linewidth=.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7, loc='lower left'); ax.tick_params(axis='x', rotation=55, labelsize=7); ax.spines[['top','right']].set_visible(False)

ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor('white')
labels = ['N','S','V','F','Q']; counts = pred['true_class'].value_counts().reindex(labels).fillna(0)
ax.bar(labels, counts.values, color=[blue, orange, '#c85c43', '#806b9b', '#7e8791'], width=.65)
ax.set_title('Held-out class distribution', loc='left', color=ink, weight='bold'); ax.set_ylabel('Beats'); ax.grid(axis='y', color=grid, linewidth=.6); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
for i, v in enumerate(counts.values): ax.text(i, v + max(counts.values) * .02, f'{int(v):,}', ha='center', fontsize=7, color=ink)

ax = fig.add_subplot(gs[1, 0]); ax.set_facecolor('white')
cm = pd.crosstab(pred['true_class'], pred['pred_class']).reindex(index=labels, columns=labels, fill_value=0).values
im = ax.imshow(cm, cmap='Blues'); ax.set_title('Confusion matrix · held-out predictions', loc='left', color=ink, weight='bold'); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_xticks(range(5), labels); ax.set_yticks(range(5), labels)
for i in range(5):
    for j in range(5): ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=7, color=ink)
ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, 5, 1), minor=True); ax.grid(which='minor', color='white', linewidth=1); ax.tick_params(which='minor', bottom=False, left=False)

ax = fig.add_subplot(gs[1, 1]); ax.set_facecolor('white')
if 'confidence' in pred.columns:
    bins = np.linspace(0, 1, 11); pred['bin'] = pd.cut(pred['confidence'], bins=bins, include_lowest=True); grouped = pred.groupby('bin', observed=False).agg(conf=('confidence','mean'), correct=('correct','mean')).dropna(); ax.plot(grouped.conf, grouped.correct, marker='o', color=blue, label='Observed'); ax.plot([0,1],[0,1], '--', color='#9aa4ad', label='Perfectly calibrated'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel('Mean confidence'); ax.set_ylabel('Accuracy'); ax.legend(frameon=False, fontsize=7)
ax.set_title('Reliability view', loc='left', color=ink, weight='bold'); ax.grid(color=grid, linewidth=.6); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)

ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor('white')
summary = [serving.get('p50_ms', np.nan), serving.get('p95_ms', np.nan), serving.get('mean_ms', np.nan)]
ax.bar(['p50', 'p95', 'mean'], summary, color=[blue, orange, '#6b7ea2'], width=.55); ax.set_title('Measured API serving latency', loc='left', color=ink, weight='bold'); ax.set_ylabel('Milliseconds'); ax.grid(axis='y', color=grid, linewidth=.6); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
for i, v in enumerate(summary): ax.text(i, v + max(summary) * .03, f'{v:.2f}', ha='center', fontsize=8, color=ink)

fig.suptitle('VIGIL · MIT-BIH ECG research snapshot', x=.03, ha='left', fontsize=15, color='#1e344c', weight='bold')
fig.text(.03, .945, 'Real repository artifacts only · grouped record-level evaluation · research prototype', color='#78838d', fontsize=8)
fig.savefig(DOCS / 'vigil_readme_overview.png', dpi=180, bbox_inches='tight')
plt.close(fig)

# Standalone waveform image with the real bundled sample.
sample = json.loads((ROOT / 'ml/sample_waveform.json').read_text())
fig, ax = plt.subplots(figsize=(12, 2.8), facecolor='white'); ax.plot(sample['signal'], color='#c85c43', linewidth=.7); ax.set_title('Real MIT-BIH record 100 waveform segment', loc='left', color=ink, weight='bold'); ax.set_xlabel('Samples at 360 Hz'); ax.set_ylabel('Amplitude'); ax.grid(color=grid, linewidth=.5); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); fig.savefig(DOCS / 'vigil_ecg_waveform.png', dpi=180, bbox_inches='tight'); plt.close(fig)
print(DOCS / 'vigil_readme_overview.png')
print(DOCS / 'vigil_ecg_waveform.png')
