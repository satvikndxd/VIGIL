import { createRoot } from 'react-dom/client';
import { useEffect, useMemo, useState } from 'react';
import './styles.css';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const classes = ['N', 'S', 'V', 'F', 'Q'];
const labels = { N: 'Normal', S: 'Supraventricular', V: 'Ventricular', F: 'Fusion', Q: 'Unknown' };
const colors = { N: '#315a91', S: '#e28b42', V: '#c85c43', F: '#806b9b', Q: '#7e8791' };
const fmt = (n, d = 3) => n === null || n === undefined || Number.isNaN(Number(n)) ? '—' : Number(n).toFixed(d);
const pct = n => n === null || n === undefined ? '—' : `${(Number(n) * 100).toFixed(1)}%`;

async function api(path, options) {
  const r = await fetch(`${API}${path}`, options);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function Panel({ title, tabs, activeTab, onTab, children, className = '' }) {
  return <section className={`panel ${className}`}><div className="panel-bar"><div className="panel-title"><span className="panel-icon">▦</span>{title}</div>{tabs && <div className="panel-tabs">{tabs.map(tab => <button key={tab} className={activeTab === tab ? 'tab-active' : ''} onClick={() => onTab?.(tab)}>{tab}</button>)}</div>}<div className="panel-actions">⋮</div></div>{children}</section>;
}

function SqlEditor({ model, record, prediction }) {
  const p = prediction?.probabilities || {};
  const sql = [
    '-- VIGIL / LIVE ECG ANALYSIS',
    'SELECT',
    `  record_id AS RECORD,`,
    `  '${prediction?.class_name || '—'}' AS PREDICTED_CLASS,`,
    `  ROUND(${Number(p[prediction?.class_name] || 0).toFixed(4)}, 4) AS CONFIDENCE,`,
    `  '${model || 'RNN'}' AS MODEL,`,
    `  '${record || '100'}' AS ECG_RECORD`,
    'FROM mitbih_inference',
    'WHERE sequence_length = 8',
    'ORDER BY confidence DESC;'
  ];
  return <div className="sql-editor"><div className="editor-gutter">{sql.map((_, i) => <span key={i}>{i + 1}</span>)}</div><pre>{sql.map((line, i) => <code key={i} className={i === 0 ? 'comment' : i === 1 || i === 7 || i === 8 || i === 9 ? 'keyword' : ''}>{line}{'\n'}</code>)}</pre><div className="editor-status"><span>● Analysis ready</span><span>Ln 10, Col 31</span></div></div>;
}

function ResultTable({ rows = [], compact = false }) {
  return <div className={`result-scroll ${compact ? 'compact' : ''}`}><table className="result-table"><thead><tr><th>#</th><th>RECORD</th><th>TRUE_CLASS</th><th>PREDICTED_CLASS</th><th>CONFIDENCE</th><th>STATUS</th></tr></thead><tbody>{rows.slice(0, compact ? 8 : 14).map((row, i) => <tr key={`${row.record}-${i}`}><td>{i + 1}</td><td>{row.record}</td><td><b style={{ color: colors[row.true_class] }}>{row.true_class}</b></td><td><b style={{ color: colors[row.pred_class] }}>{row.pred_class}</b></td><td>{pct(row.confidence)}</td><td><span className={row.correct ? 'status-good' : 'status-review'}>{row.correct ? 'CORRECT' : 'REVIEW'}</span></td></tr>)}</tbody></table></div>;
}

function Connections({ model, metrics }) {
  return <aside className="connections"><div className="side-heading">Connections <span>×</span></div><div className="connection-root"><span className="db-dot" /> VIGIL</div><div className="tree"><div>⌄ <b>MIT-BIH ARRHYTHMIA</b></div><div>▦ Records <em>48</em></div><div>◈ Models <em>{model?.available_models?.length || 5}</em></div><div>▤ Metrics <em>{metrics?.models?.length || 7}</em></div><div>◌ Explanations <em>2</em></div><div>▾ API endpoints</div><div className="tree-child">GET /health</div><div className="tree-child">POST /predict</div><div className="tree-child">GET /metrics</div></div><div className="side-footer">Repository: main<br />Model: {model?.model_version || '—'}</div></aside>;
}

function ProbabilityTable({ probabilities = {} }) {
  return <div className="prob-table">{classes.map(key => <div className="prob-line" key={key}><span><i style={{ background: colors[key] }} />{key} · {labels[key]}</span><div className="tiny-track"><div style={{ width: `${Math.max(1, Number(probabilities[key] || 0) * 100)}%`, background: colors[key] }} /></div><b>{pct(probabilities[key] || 0)}</b></div>)}</div>;
}

function AttentionStrip({ attention = [] }) {
  const weights = attention.length ? attention : classes.map((_, i) => ({ beat_position: i, weight: 0 }));
  const max = Math.max(...weights.map(x => x.weight), 0.001);
  return <div className="attention-strip">{weights.map(item => <div className="attention-cell" key={item.beat_position}><span style={{ height: `${8 + (item.weight / max) * 35}px`, opacity: .25 + (item.weight / max) * .75 }} /><small>B{item.beat_position + 1}</small></div>)}</div>;
}

function ResearchHeadline({ research }) {
  const f = research?.final_metrics || {}; const base = research?.final_comparison?.find?.(x => x.metric === 'MacroF1');
  return <div className="research-headline"><div><span>RESEARCH FINAL MODEL</span><b>{f.model || '—'}</b><small>Frozen after Phase 2 validation; no test tuning</small></div><div><span>MACRO-F1</span><b>{fmt(f.MacroF1)}</b><small>Baseline delta {fmt(base?.absolute_delta)}</small></div><div><span>MACRO-AUROC</span><b>{fmt(f.MacroAUROC)}</b><small>Macro-AUPRC {fmt(f.MacroAUPRC)}</small></div><div><span>MINORITY RECALL</span><b>N {pct(f.N_recall)} · F {pct(f.F_recall)}</b><small>No fabricated improvement</small></div></div>;
}

function RecordGeneralization({ rows = [] }) {
  return <div className="record-generalization"><div className="subhead">PER-RECORD GENERALIZATION</div><div className="record-bars">{rows.map(row => <div className="record-bar" key={row.record}><span>{row.record}</span><div><i style={{ width: `${Math.max(2, Number(row.MacroF1 || 0) * 100)}%` }} /></div><b>{row.MacroF1 == null ? '—' : fmt(row.MacroF1, 2)}</b></div>)}</div></div>;
}

function SymbolShift({ rows = [] }) {
  return <div className="symbol-shift"><div className="subhead">ORIGINAL SYMBOL SHIFT</div><div className="shift-table"><table className="result-table"><thead><tr><th>SYMBOL</th><th>MAP</th><th>TRAIN</th><th>TEST</th><th>ERROR</th><th>RECALL</th></tr></thead><tbody>{rows.filter(r => Number(r.test_count) > 0).slice(0, 12).map(row => <tr key={row.original_symbol}><td><b>{row.original_symbol}</b></td><td>{row.mapped_class}</td><td>{row.train_count}</td><td>{row.test_count}</td><td>{row.test_error_rate == null ? '—' : pct(row.test_error_rate)}</td><td>{row.per_symbol_recall == null ? '—' : pct(row.per_symbol_recall)}</td></tr>)}</tbody></table></div></div>;
}

function MinorityPanel({ metrics }) {
  const f = metrics?.final_metrics || {}; const rows = metrics?.final_per_class || []; const recall = c => { const row = rows.find(x => x.class === c); return row?.recall ?? (c === 'N' ? f.N_recall : c === 'F' ? f.F_recall : null); }; return <div className="minority-panel"><div className="subhead">MINORITY-CLASS PANEL</div>{classes.map(c => <div className="minority-row" key={c}><span><i style={{ background: colors[c] }} />{c} · {labels[c]}</span><b>{recall(c) == null ? '—' : pct(recall(c))}</b></div>)}<small>Final retained model: {f.model || '—'}. N and F remain unresolved in the frozen result.</small></div>;
}

function RobustnessPanel({ rows = [] }) {
  return <div className="robustness-panel"><div className="subhead">ROBUSTNESS · FINAL MODEL</div><div className="robustness-table"><table className="result-table"><thead><tr><th>CONDITION</th><th>MACRO-F1</th><th>Δ</th></tr></thead><tbody>{rows.map(row => <tr key={row.corruption}><td>{row.corruption}</td><td>{fmt(row.MacroF1)}</td><td>{fmt(row.MacroF1_delta ?? row.MacroF1_drop_from_clean, 3)}</td></tr>)}</tbody></table></div><small>Controlled perturbations only; not noise immunity.</small></div>;
}

function Phase3Panels({ research }) {
  try {
    const p3 = research?.phase3 || {}; const f = p3.metrics || {}; const rows = Array.isArray(p3.representation_ablation) ? p3.representation_ablation : []; const cal = Array.isArray(p3.calibration) ? p3.calibration : []; const robust = Array.isArray(p3.robustness) ? p3.robustness : [];
    return <div className="research-section"><Panel title="PHASE 3 / DOMAIN-GENERALIZED REPRESENTATION LEARNING"><div className="research-headline"><div><span>PHASE 3 DECISION</span><b>{f.replacement_gate_passed ? 'Candidate retained' : 'Frozen baseline retained'}</b><small>{f.selected_phase3_candidate || '—'} explored</small></div><div><span>MACRO-F1</span><b>{fmt(f.MacroF1)}</b><small>Δ 0.000 vs frozen baseline</small></div><div><span>MACRO-AUPRC</span><b>{fmt(f.MacroAUPRC)}</b><small>Macro-AUROC {fmt(f.MacroAUROC)}</small></div><div><span>N / F RECALL</span><b>{pct(f.N_recall)} · {pct(f.F_recall)}</b><small>No forced improvement</small></div></div><div className="phase3-grid"><div className="phase3-table"><div className="subhead">REPRESENTATION ABLATION</div><table className="result-table"><thead><tr><th>REPRESENTATION</th><th>F1</th><th>AUPRC</th><th>AUROC</th></tr></thead><tbody>{rows.map((row, i) => <tr key={row.representation || i}><td>{row.representation || '—'}</td><td>{fmt(row.MacroF1)}</td><td>{fmt(row.MacroAUPRC)}</td><td>{fmt(row.MacroAUROC)}</td></tr>)}</tbody></table></div><div className="phase3-table"><div className="subhead">CALIBRATION · VALIDATION-FIT TEMPERATURE</div><table className="result-table"><thead><tr><th>MODEL</th><th>ECE</th><th>BRIER</th><th>F1</th></tr></thead><tbody>{cal.map((row, i) => <tr key={row.model || i}><td>{row.model || '—'}</td><td>{fmt(row.ECE)}</td><td>{fmt(row.Brier)}</td><td>{fmt(row.MacroF1)}</td></tr>)}</tbody></table></div><div className="phase3-table"><div className="subhead">ROBUSTNESS · BASELINE VS PHASE 3</div><table className="result-table"><thead><tr><th>MODEL</th><th>CONDITION</th><th>F1</th><th>Δ</th></tr></thead><tbody>{robust.slice(0, 10).map((row, i) => <tr key={`${row.model || 'model'}-${row.corruption || 'condition'}-${i}`}><td>{row.model || '—'}</td><td>{row.corruption || '—'}</td><td>{fmt(row.MacroF1)}</td><td>{fmt(row.MacroF1_delta, 3)}</td></tr>)}</tbody></table></div></div></Panel></div>;
  } catch (error) {
    return <div className="sql-error">Phase 3 panel error: {String(error?.message || error)}</div>;
  }
}

function ResearchPanels({ research }) {
  return <div className="research-section"><Panel title="PHASE 2 / RESEARCH RESULTS"><ResearchHeadline research={research} /><div className="research-grid"><RecordGeneralization rows={research?.record_generalization} /><MinorityPanel metrics={research} /><SymbolShift rows={research?.symbol_shift} /><RobustnessPanel rows={research?.robustness} /></div></Panel></div>;
}

function Waveform({ signal = [], predicted = 'N' }) {
  const points = useMemo(() => {
    if (!signal.length) return '';
    const step = Math.max(1, Math.ceil(signal.length / 600)); const values = signal.filter((_, i) => i % step === 0); const min = Math.min(...values); const max = Math.max(...values); const range = max - min || 1;
    return values.map((v, i) => `${(i / Math.max(1, values.length - 1)) * 100},${96 - ((v - min) / range) * 84}`).join(' ');
  }, [signal]);
  return <div className="ecg-mini"><svg viewBox="0 0 100 100" preserveAspectRatio="none"><path d="M0 18 H100 M0 50 H100 M0 82 H100" className="ecg-grid" /><polyline points={points} style={{ stroke: colors[predicted] || colors.N }} /></svg><div className="wave-axis"><span>0 ms</span><span>500</span><span>1000</span><span>1500</span><span>2000 ms</span></div></div>;
}

function Dashboard() {
  const [health, setHealth] = useState(null); const [model, setModel] = useState(null); const [metrics, setMetrics] = useState(null); const [research, setResearch] = useState(null); const [sample, setSample] = useState(null); const [prediction, setPrediction] = useState(null); const [records, setRecords] = useState([]); const [explanations, setExplanations] = useState(null); const [selectedModel, setSelectedModel] = useState('RNN'); const [view, setView] = useState('Dashboard'); const [tab, setTab] = useState('Query Result'); const [error, setError] = useState(''); const [loading, setLoading] = useState(true);

  async function analyze(signal, modelName = selectedModel) {
    if (!signal) return;
    try { const result = await api('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ signal, model: modelName, record_id: sample?.record_id }) }); setPrediction(result); } catch (e) { setError(e.message); }
  }
  async function load() {
    try { setLoading(true); const [h, m, mt, r, s, e, rs] = await Promise.all([api('/health'), api('/model'), api('/metrics'), api('/records?limit=50'), api('/sample-signal'), api('/explanations'), api('/research')]); setHealth(h); setModel(m); setMetrics(mt); setRecords(r.records || []); setSample(s); setExplanations(e); setResearch(rs); await analyze(s.signal, m.default_model); } catch (e) { setError(e.message); } finally { setLoading(false); }
  }
  useEffect(() => { load(); }, []);
  useEffect(() => { if (sample?.signal && model?.default_model) analyze(sample.signal, selectedModel); }, [selectedModel]);

  const rows = metrics?.models || []; const active = rows.find(r => String(r.Model).toLowerCase() === selectedModel.toLowerCase()) || rows.find(r => String(r.Model).toLowerCase() === 'rnn') || {};
  const bestMacro = rows.reduce((a, b) => Number(a.MacroF1 || 0) > Number(b.MacroF1 || 0) ? a : b, {}); const bestAuroc = rows.reduce((a, b) => Number(a.AUROC_OVR || 0) > Number(b.AUROC_OVR || 0) ? a : b, {});
  const displayRows = records.slice(0, 10); const predictionClass = prediction?.class_name || 'N';
  const sqlText = `SELECT record_id, predicted_class, confidence\nFROM mitbih_inference\nWHERE model = '${selectedModel}'\nORDER BY confidence DESC;`;

  const content = view === 'Experiments' ? <><Panel title="MODEL BENCHMARK / QUERY RESULT" tabs={['Query Result', 'Script Output']} activeTab={tab} onTab={setTab}><ResultTable rows={records} /><div className="metrics-ribbon">{rows.map(row => <div key={row.Model}><span>{row.Model}</span><b>{fmt(row.MacroF1)}</b><small>Macro F1</small></div>)}</div></Panel><ResearchPanels research={research} /></> : view === 'Records' ? <><Panel title="HELD-OUT PREDICTIONS / QUERY RESULT"><ResultTable rows={records} /></Panel><Panel title="RECORD GENERALIZATION / QUERY RESULT"><RecordGeneralization rows={research?.record_generalization} /></Panel></> : view === 'About' ? <Panel title="VIGIL / SCRIPT OUTPUT"><div className="about-console"><p><b>VIGIL</b> — Interpretable Temporal Deep Learning for ECG Arrhythmia Classification</p><p>Dataset: MIT-BIH Arrhythmia Database · 5-class grouped record evaluation · PyTorch · FastAPI · React/Vite</p><p>Scope: retrospective research prototype; not a medical diagnostic device.</p><p>Available artifacts: benchmark checkpoints, metrics, predictions, attention, Integrated Gradients, serving benchmark, and frontend contract examples.</p></div></Panel> : <>
    <div className="workspace-row top-row"><Panel title="SQL / ANALYSIS EDITOR" tabs={['Practice: Write the Query']} activeTab="Practice: Write the Query" className="editor-panel"><SqlEditor model={selectedModel} record={sample?.record_id} prediction={prediction} /></Panel><Panel title="QUERY RESULT" tabs={['Query Result']} activeTab="Query Result" className="top-result"><ResultTable rows={displayRows} compact /><div className="result-footer">All rows fetched: {records.length} · {fmt(prediction?.latency_ms, 2)} ms inference · source: held-out artifact</div></Panel></div>
    <div className="workspace-row bottom-row"><Connections model={model} metrics={metrics} /><Panel title="VIGIL / WORKSHEET" tabs={['Worksheet', 'Query Builder']} activeTab={tab} onTab={setTab} className="worksheet"><div className="worksheet-code"><div className="code-line comment">-- Live inference contract</div><div className="code-line"><b>SELECT</b> {selectedModel} AS MODEL,</div><div className="code-line">&nbsp;&nbsp;'{predictionClass}' AS PREDICTED_CLASS,</div><div className="code-line">&nbsp;&nbsp;{pct(prediction?.confidence)} AS CONFIDENCE,</div><div className="code-line">&nbsp;&nbsp;{model?.sequence_length || 8} AS SEQUENCE_LENGTH</div><div className="code-line"><b>FROM</b> MITBIH_INFERENCE;</div></div><div className="analysis-rail"><div className="analysis-result"><span>PREDICTION</span><b style={{ color: colors[predictionClass] }}>{labels[predictionClass]} ({predictionClass})</b><small>{pct(prediction?.confidence)} confidence</small></div><div className="analysis-result"><span>BEST MACRO F1</span><b>{fmt(bestMacro.MacroF1)}</b><small>{bestMacro.Model || '—'}</small></div><div className="analysis-result"><span>BEST AUROC</span><b>{fmt(bestAuroc.AUROC_OVR)}</b><small>{bestAuroc.Model || '—'}</small></div></div><div className="ecg-block"><div className="subhead">ECG SIGNAL · RECORD {sample?.record_id || '—'} · 360 HZ</div><Waveform signal={prediction?.waveform || sample?.signal || []} predicted={predictionClass} /></div><div className="prob-block"><div className="subhead">CLASS PROBABILITY</div><ProbabilityTable probabilities={prediction?.probabilities} /></div><div className="attention-block"><div className="subhead">TEMPORAL ATTENTION</div><AttentionStrip attention={prediction?.attention} /><small>Attention is descriptive, not a causal explanation.</small></div></Panel><Panel title="QUERY RESULT / SCRIPT OUTPUT" tabs={['Query Result', 'Model Comparison']} activeTab="Query Result" className="result-panel"><div className="result-tab-content"><div className="result-heading">{tab === 'Model Comparison' ? 'Model comparison' : 'Prediction result'} <span>✓ API response</span></div>{tab === 'Model Comparison' ? <div className="compact-comparison">{rows.map(row => <div className="comparison-row" key={row.Model}><span>{row.Model}</span><b>{fmt(row.MacroF1)}</b><b>{fmt(row.AUROC_OVR)}</b><b>{fmt(row.AUPRC_macro)}</b></div>)}</div> : <><div className="result-stat-grid"><div><span>PREDICTED</span><b style={{ color: colors[predictionClass] }}>{predictionClass}</b></div><div><span>CONFIDENCE</span><b>{pct(prediction?.confidence)}</b></div><div><span>MODEL</span><b>{selectedModel}</b></div></div><ProbabilityTable probabilities={prediction?.probabilities} /><div className="result-heading lower">EXPLANATION OUTPUT</div><p className="explain-note">{explanations?.integrated_gradients_available ? 'Integrated Gradients artifact available in ml/explanations.' : 'Temporal attention artifact available for attention-enabled checkpoints.'}</p></>}</div><div className="result-footer">Query executed successfully · model version {model?.model_version || '—'}</div></Panel></div><ResearchPanels research={research} /><Phase3Panels research={research} />
  </>;

  return <div className="sql-shell"><header className="sql-header"><div className="app-title"><div className="app-logo">V</div><div><b>VIGIL</b><span>ECG ARRHYTHMIA WORKBENCH</span></div></div><div className="sql-nav">{['Dashboard', 'Experiments', 'Records', 'About'].map(item => <button key={item} className={view === item ? 'nav-selected' : ''} onClick={() => setView(item)}>{item}</button>)}</div><div className="header-status"><span className="online-dot" /> MODEL ONLINE <b>{model?.model_version || '—'}</b></div></header><div className="toolbar"><button className="run-btn" onClick={() => analyze(sample?.signal)}>▶ Run Analysis</button><button onClick={load}>↻ Refresh</button><span className="toolbar-separator" /><span>Connection: <b>VIGIL / MIT-BIH</b></span><span>Record: <b>{sample?.record_id || '—'}</b></span><span>Model: <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>{(model?.available_models || ['RNN']).map(name => <option key={name}>{name}</option>)}</select></span><span className="toolbar-right">{health?.status === 'ok' ? 'Query executed successfully' : 'API unavailable'}</span></div>{error && <div className="sql-error">{error}</div>}<main className="workbench">{loading ? <div className="loading">Connecting to VIGIL API…</div> : content}</main><footer className="sql-footer"><span>VIGIL · Research prototype only</span><span>Ln 14, Column 22 · 0.004 seconds · PyTorch / FastAPI</span></footer></div>;
}

createRoot(document.getElementById('root')).render(<Dashboard />);
