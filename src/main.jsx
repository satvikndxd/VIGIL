import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const colors = { N: '#315a91', S: '#e28b42', V: '#c85c43', F: '#806b9b', Q: '#7e8791' };
const labels = { N: 'Normal', S: 'Supraventricular', V: 'Ventricular', F: 'Fusion', Q: 'Unknown' };

function fmt(n, digits = 3) { return n === null || n === undefined || Number.isNaN(Number(n)) ? '—' : Number(n).toFixed(digits); }
function pct(n) { return n === null || n === undefined ? '—' : `${(Number(n) * 100).toFixed(1)}%`; }

async function getJson(path, options) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function Card({ title, kicker, children, className = '' }) {
  return <section className={`card ${className}`}><div className="card-head"><span className="card-title">{title}</span>{kicker && <span className="card-kicker">{kicker}</span>}</div>{children}</section>;
}

function Kpi({ label, value, note, accent = 'blue' }) {
  return <div className="kpi" style={{ '--accent': accent }}><div className="kpi-label">{label}</div><div className="kpi-value">{value}</div><div className="kpi-note">{note}</div></div>;
}

function Waveform({ signal = [], probability = {}, predicted = 'N' }) {
  const width = 920, height = 230, pad = 18;
  const points = useMemo(() => {
    if (!signal.length) return '';
    const slice = signal.length > 720 ? signal.filter((_, i) => i % Math.ceil(signal.length / 720) === 0) : signal;
    const min = Math.min(...slice), max = Math.max(...slice), range = max - min || 1;
    return slice.map((v, i) => `${pad + (i / Math.max(slice.length - 1, 1)) * (width - 2 * pad)},${height - pad - ((v - min) / range) * (height - 2 * pad)}`).join(' ');
  }, [signal]);
  return <div className="wave-wrap">
    <div className="wave-meta"><div><span className="micro-label">PREDICTED RHYTHM</span><strong className="prediction" style={{ color: colors[predicted] }}>{labels[predicted] || predicted} ({predicted})</strong></div><div className="wave-confidence"><span className="micro-label">CONFIDENCE</span><strong>{pct(Math.max(...Object.values(probability || { N: 0 })))}</strong></div></div>
    <svg className="waveform" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="ECG waveform">
      {[40, 85, 130, 175, 220].map(y => <line key={y} x1="0" x2={width} y1={y} y2={y} className="grid-line" />)}
      {[80, 240, 400, 560, 720, 880].map(x => <line key={x} x1={x} x2={x} y1="0" y2={height} className="grid-line" />)}
      <polyline points={points} fill="none" stroke={colors[predicted] || colors.N} strokeWidth="1.7" vectorEffect="non-scaling-stroke" />
    </svg>
    <div className="axis"><span>0 ms</span><span>500 ms</span><span>1000 ms</span><span>1500 ms</span><span>2000 ms</span></div>
  </div>;
}

function ProbabilityBars({ probabilities = {} }) {
  return <div className="prob-list">{Object.entries(probabilities).map(([key, value]) => <div className="prob-row" key={key}><div className="prob-label"><span className="dot" style={{ background: colors[key] }} />{key}<span className="prob-name">{labels[key]}</span></div><div className="prob-track"><div className="prob-fill" style={{ width: `${Math.max(1, value * 100)}%`, background: colors[key] }} /></div><strong>{pct(value)}</strong></div>)}</div>;
}

function Attention({ attention = [] }) {
  const items = attention.length ? attention : Array.from({ length: 8 }, (_, i) => ({ beat_position: i, weight: 0 }));
  const max = Math.max(...items.map(x => x.weight), 0.001);
  return <div className="attention"><div className="attention-bars">{items.map(item => <div className="att-cell" key={item.beat_position}><div className="att-bar" style={{ height: `${12 + (item.weight / max) * 54}px`, opacity: 0.28 + (item.weight / max) * 0.72 }} /><span>Beat {item.beat_position + 1}</span><small>{fmt(item.weight, 3)}</small></div>)}</div><div className="attention-note">Temporal attention is descriptive and is not a causal explanation.</div></div>;
}

function ConfusionMatrix({ matrix = [] }) {
  const values = matrix.length ? matrix : Array.from({ length: 5 }, () => Array(5).fill(0));
  return <div className="matrix-wrap"><div className="matrix-label top">PREDICTED</div><div className="matrix-grid"><div className="matrix-y">TRUE</div><div className="matrix-table"><div className="matrix-row header"><span />{Object.keys(labels).map(k => <span key={k}>{k}</span>)}</div>{values.map((row, i) => <div className="matrix-row" key={i}><b>{Object.keys(labels)[i]}</b>{row.map((value, j) => <span key={j} style={{ background: `rgba(49,90,145,${Math.min(0.76, 0.08 + value / Math.max(1, Math.max(...values.flat())) * .68)})` }}>{value}</span>)}</div>)}</div></div></div>;
}

function Dashboard() {
  const [health, setHealth] = useState(null); const [model, setModel] = useState(null); const [metrics, setMetrics] = useState(null); const [sample, setSample] = useState(null); const [prediction, setPrediction] = useState(null); const [records, setRecords] = useState([]); const [explanations, setExplanations] = useState(null); const [selectedModel, setSelectedModel] = useState('RNN'); const [classFilter, setClassFilter] = useState('All'); const [loading, setLoading] = useState(true); const [error, setError] = useState(''); const [view, setView] = useState('Dashboard');

  async function load() {
    try { setLoading(true); setError(''); const [h, m, r, s, e] = await Promise.all([getJson('/health'), getJson('/model'), getJson('/metrics'), getJson('/records?limit=25'), getJson('/explanations')]); setHealth(h); setModel(m); setMetrics(r); setRecords(s.records || []); setSample(await getJson('/sample-signal')); setExplanations(e); } catch (err) { setError(err.message); } finally { setLoading(false); }
  }
  useEffect(() => { load(); }, []);
  useEffect(() => { if (sample?.signal?.length) analyze(sample.signal); }, [sample, selectedModel]);
  async function analyze(signal = sample?.signal) { if (!signal) return; try { const p = await getJson('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ signal, model: selectedModel, record_id: sample?.record_id }) }); setPrediction(p); } catch (err) { setError(err.message); } }

  const modelRows = metrics?.models || [];
  const rowFor = (name) => modelRows.find(row => String(row.Model || row.model || '').toLowerCase().includes(name.toLowerCase())) || {};
  const activeRow = rowFor(selectedModel === 'BiLSTM_Attention' ? 'BiLSTM_Attention' : selectedModel);
  const best = metrics?.best_neural_model || model?.default_model || 'RNN';
  const classCounts = Object.keys(labels).map(k => ({ key: k, value: Number(metrics?.class_distribution?.[k] || 0) }));
  const matrix = metrics?.confusion_matrix || Object.keys(labels).map(a => Object.keys(labels).map(b => records.filter(r => r.true_class === a && r.pred_class === b).length));
  const shownRecords = classFilter === 'All' ? records : records.filter(r => r.true_class === classFilter || r.pred_class === classFilter);

  return <div className="shell">
    <header className="topbar"><div className="brand"><div className="brand-mark">V</div><div><div className="brand-name">VIGIL</div><div className="brand-sub">Interpretable Temporal Deep Learning for ECG Arrhythmia Classification</div></div></div><nav>{['Dashboard', 'Experiments', 'Records', 'About'].map(item => <button className={view === item ? 'nav-active' : ''} onClick={() => setView(item)} key={item}>{item}</button>)}</nav><div className="status"><span className={`status-dot ${health?.status === 'ok' ? 'online' : ''}`} /> <span>{health?.status === 'ok' ? 'MODEL ONLINE' : 'API OFFLINE'}</span><b>{model?.model_version || '—'}</b></div></header>
    <main>
      <div className="controlbar"><div className="control"><label>RECORD / PATIENT</label><select value={sample?.record_id || ''} onChange={() => {}}><option>{sample?.record_id || 'Loading'}</option></select></div><div className="control"><label>MODEL</label><select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>{(model?.available_models || ['RNN']).map(name => <option key={name}>{name}</option>)}</select></div><div className="control"><label>SEQUENCE</label><select><option>{model?.sequence_length || 8} beats</option></select></div><div className="control"><label>CLASS FILTER</label><select value={classFilter} onChange={e => setClassFilter(e.target.value)}><option>All</option>{Object.keys(labels).map(k => <option key={k}>{k}</option>)}</select></div><button className="analyze" onClick={() => analyze()}>ANALYZE ECG</button><span className="control-note">LIVE API OUTPUTS</span></div>
      {error && <div className="notice error">API unavailable: {error}. Start the FastAPI service to load measured outputs.</div>}
      {loading ? <div className="loading">Loading VIGIL analytics workspace…</div> : view === 'Experiments' ? <div className="page-view"><div className="section-line"><div><span className="eyebrow">RESEARCH BENCHMARK</span><h1>Experiments</h1></div><div className="section-meta">Actual held-out metrics from repository artifacts</div></div><Card title="EXPERIMENTS" kicker="Grouped record-level split"><div className="mini-table experiment-table"><div className="mini-row header"><span>MODEL</span><span>MACRO F1</span><span>AUROC</span><span>AUPRC</span></div>{modelRows.map((r, i) => <div className={`mini-row ${String(r.Model || r.model) === best ? 'highlight' : ''}`} key={i}><span>{r.Model || r.model}</span><span>{fmt(r.MacroF1)}</span><span>{fmt(r.AUROC_OVR)}</span><span>{fmt(r.AUPRC_macro)}</span></div>)}</div></Card><div className="lower-grid"><Card title="EXPERIMENT NOTES" kicker="No fabricated benchmark claims"><p className="body-copy">The benchmark preserves the measured result: the RNN is the best neural model by Macro-F1 in the current run, while Logistic Regression and Random Forest remain important comparison baselines. The dashboard does not visually imply that Bi-LSTM + Attention wins.</p></Card><Card title="MODEL INFORMATION" kicker="Current artifact contract"><dl className="info-grid"><dt>Best neural model</dt><dd>{best}</dd><dt>Framework</dt><dd>{model?.framework || 'PyTorch'}</dd><dt>Sequence</dt><dd>{model?.sequence_length || 8} beats</dd><dt>Input</dt><dd>{model?.samples_per_beat || 180} samples / beat</dd><dt>Parameters</dt><dd>{activeRow.Params ? Number(activeRow.Params).toLocaleString() : '—'}</dd><dt>Seed</dt><dd>42</dd></dl></Card></div></div> : view === 'Records' ? <div className="page-view"><div className="section-line"><div><span className="eyebrow">HELD-OUT PREDICTIONS</span><h1>Records</h1></div><div className="section-meta">{shownRecords.length} rows loaded from the actual test prediction artifact</div></div><Card title="PREDICTION HISTORY" kicker="Use the class filter above"><div className="table-scroll tall"><table><thead><tr><th>RECORD</th><th>TRUE CLASS</th><th>PREDICTED</th><th>CONFIDENCE</th><th>STATUS</th></tr></thead><tbody>{shownRecords.map((r, i) => <tr key={i}><td>{r.record}</td><td><span className="class-tag" style={{ color: colors[r.true_class] }}>{r.true_class}</span></td><td><span className="class-tag" style={{ color: colors[r.pred_class] }}>{r.pred_class}</span></td><td>{pct(r.confidence)}</td><td><span className={r.correct ? 'ok-tag' : 'bad-tag'}>{r.correct ? 'CORRECT' : 'REVIEW'}</span></td></tr>)}</tbody></table></div></Card></div> : view === 'About' ? <div className="page-view"><div className="section-line"><div><span className="eyebrow">PROJECT INFORMATION</span><h1>About VIGIL</h1></div><div className="section-meta">Research engineering system</div></div><div className="lower-grid"><Card title="VIGIL" kicker="Interpretable temporal deep learning"><p className="body-copy">VIGIL is an enterprise-style ECG analytics workstation for five-class MIT-BIH arrhythmia research. It combines the existing PyTorch checkpoints, grouped record-level evaluation, FastAPI inference, and a compact BI dashboard.</p><p className="body-copy">This is a retrospective research prototype and is not a medical diagnostic device. Temporal attention is descriptive, not causal, and the measured minority-class limitations remain visible in the experiment artifacts.</p></Card><Card title="DATA CONTRACT" kicker="Real outputs only"><dl className="info-grid"><dt>Dataset</dt><dd>MIT-BIH mirror</dd><dt>Classes</dt><dd>N · S · V · F · Q</dd><dt>Sample rate</dt><dd>{model?.sample_rate_hz || 360} Hz</dd><dt>Model version</dt><dd>{model?.model_version || '—'}</dd><dt>API</dt><dd>FastAPI</dd><dt>Frontend</dt><dd>React + Vite</dd></dl></Card></div></div> : <>
        <div className="section-line"><div><span className="eyebrow">ECG ANALYTICS WORKSPACE</span><h1>{view === 'Dashboard' ? 'Performance overview' : view}</h1></div><div className="section-meta">Dataset: MIT-BIH · 5 classes · Grouped record split · Seed 42</div></div>
        <div className="kpi-grid"><Kpi label="ACCURACY" value={fmt(activeRow.Accuracy)} note={`${selectedModel} held-out test`} /><Kpi label="MACRO F1" value={fmt(activeRow.MacroF1)} note="Class-balanced summary" accent="orange" /><Kpi label="AUROC OVR" value={fmt(activeRow.AUROC_OVR)} note="Macro one-vs-rest" /><Kpi label="AUPRC MACRO" value={fmt(activeRow.AUPRC_macro)} note="Imbalance-aware" accent="orange" /><Kpi label="INFERENCE" value={prediction ? `${fmt(prediction.latency_ms, 2)} ms` : '—'} note="Current request" /><Kpi label="TEST SAMPLES" value={metrics?.test_samples || records.length || '—'} note="Held-out predictions" /><Kpi label="BEST NEURAL" value={best} note="Selected by Macro F1" accent="orange" /><Kpi label="CONFIDENCE" value={prediction ? pct(prediction.confidence) : '—'} note="Current prediction" /></div>
        <div className="dashboard-grid"><Card title="ECG SIGNAL" kicker={`Record ${sample?.record_id || '—'} · ${model?.sequence_length || 8} beats`} className="wave-card"><Waveform signal={prediction?.waveform || sample?.signal || []} probability={prediction?.probabilities} predicted={prediction?.class_name || 'N'} /><div className="wave-controls"><button onClick={() => analyze()}>↻ Re-analyze</button><span>Channel 0 · 360 Hz · real MIT-BIH sample</span></div></Card><Card title="CLASS PROBABILITY" kicker={prediction ? `${prediction.class_name} selected` : 'Awaiting inference'}><ProbabilityBars probabilities={prediction?.probabilities || { N: 0, S: 0, V: 0, F: 0, Q: 0 }} /></Card><Card title="TEMPORAL ATTENTION" kicker="Sequence focus"><Attention attention={prediction?.attention || []} /></Card><Card title="CONFUSION MATRIX" kicker="Loaded test predictions"><ConfusionMatrix matrix={matrix} /></Card><Card title="MODEL COMPARISON" kicker="Held-out test metrics" className="comparison-card"><div className="mini-table"><div className="mini-row header"><span>MODEL</span><span>F1</span><span>AUROC</span><span>AUPRC</span></div>{modelRows.map((r, i) => <div className={`mini-row ${String(r.Model || r.model) === best ? 'highlight' : ''}`} key={i}><span>{r.Model || r.model}</span><span>{fmt(r.MacroF1)}</span><span>{fmt(r.AUROC_OVR)}</span><span>{fmt(r.AUPRC_macro)}</span></div>)}</div></Card><Card title="CLASS DISTRIBUTION" kicker="Held-out test rows"><div className="dist-list">{classCounts.map(item => <div className="dist-row" key={item.key}><span className="dist-key"><i className="dot" style={{ background: colors[item.key] }} />{item.key}</span><div className="dist-track"><div style={{ width: `${Math.min(100, item.value / Math.max(...classCounts.map(x => x.value), 1) * 100)}%`, background: colors[item.key] }} /></div><strong>{item.value}</strong></div>)}</div></Card></div>
        <div className="lower-grid"><Card title="ERROR ANALYSIS" kicker="Click a row for record context"><div className="error-summary"><span>False positives <b>{records.filter(r => r.correct === false && r.pred_class !== r.true_class).length}</b></span><span>Low confidence <b>{records.filter(r => Number(r.confidence) < .5).length}</b></span><span>Filtered {shownRecords.length}</span></div><div className="table-scroll"><table><thead><tr><th>RECORD</th><th>TRUE</th><th>PREDICTED</th><th>CONFIDENCE</th><th>STATUS</th></tr></thead><tbody>{shownRecords.slice(0, 12).map((r, i) => <tr key={i}><td>{r.record}</td><td><span className="class-tag" style={{ color: colors[r.true_class] }}>{r.true_class}</span></td><td><span className="class-tag" style={{ color: colors[r.pred_class] }}>{r.pred_class}</span></td><td>{pct(r.confidence)}</td><td><span className={r.correct ? 'ok-tag' : 'bad-tag'}>{r.correct ? 'CORRECT' : 'REVIEW'}</span></td></tr>)}</tbody></table></div></Card><Card title="MODEL INFORMATION" kicker="Current artifact contract"><dl className="info-grid"><dt>Framework</dt><dd>{model?.framework || 'PyTorch'}</dd><dt>Model version</dt><dd>{model?.model_version || '—'}</dd><dt>Input</dt><dd>{model?.samples_per_beat || 180} samples / beat</dd><dt>Sequence</dt><dd>{model?.sequence_length || 8} beats</dd><dt>Classes</dt><dd>{model?.classes?.join(' · ') || 'N · S · V · F · Q'}</dd><dt>Explainability</dt><dd>{explanations?.integrated_gradients_available ? 'Integrated Gradients' : 'Temporal attention'}</dd><dt>Scope</dt><dd>Retrospective research prototype</dd></dl></Card></div>
        <footer>VIGIL · ECG ARRHYTHMIA ANALYTICS · Research prototype only · Metrics are sourced from repository artifacts and API responses.</footer>
      </>}
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<Dashboard />);
