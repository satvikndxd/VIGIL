import { useEffect, useMemo, useState } from 'react';
import { api, CLASSES, CLASS_LABELS, classColor, fmt, fmtInt, pct } from './lib.js';
import { demoSnapshot } from './demo.js';
import {
  AttentionStrip,
  BarRow,
  Card,
  ClassBadge,
  EmptyState,
  ProbabilityBars,
  StatTile,
  StatusBadge,
  Waveform,
} from './components.jsx';
import { Phase2Section, Phase3Section } from './research.jsx';

const VIEWS = ['Overview', 'Benchmark', 'Records', 'Research', 'About'];

function Logo() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
      <path
        d="M2 12h4l2.5-6 4 12 2.5-6h7"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ThemeToggle({ theme, onToggle }) {
  return (
    <button className="icon-btn" onClick={onToggle} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
      {theme === 'dark' ? (
        <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
          <circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" strokeWidth="2" />
          <g stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M12 2.5v2.4M12 19.1v2.4M2.5 12h2.4M19.1 12h2.4M5 5l1.7 1.7M17.3 17.3 19 19M19 5l-1.7 1.7M6.7 17.3 5 19" />
          </g>
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
          <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4 7 7 0 0 0 20 14.5Z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
        </svg>
      )}
    </button>
  );
}

function PredictionsTable({ rows = [], limit = 12 }) {
  if (!rows.length) return <EmptyState title="No predictions available" />;
  return (
    <div className="table-wrap">
      <table className="data-table predictions-table">
        <thead>
          <tr>
            <th className="num">#</th>
            <th>Record</th>
            <th>True</th>
            <th>Predicted</th>
            <th className="num">Confidence</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, limit).map((r, i) => (
            <tr key={`${r.record}-${r.sample ?? i}`}>
              <td className="num muted">{i + 1}</td>
              <td>{r.record}</td>
              <td><ClassBadge code={r.true_class} /></td>
              <td><ClassBadge code={r.pred_class} /></td>
              <td className="num">{pct(r.confidence)}</td>
              <td><StatusBadge ok={r.correct === true || r.correct === 'True'} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------ overview */

function Overview({ state, prediction, offline }) {
  const { sample, metrics, records, model, selectedModel } = state;
  const rows = metrics?.models || [];
  const maxF1 = Math.max(...rows.map((r) => Number(r.MacroF1) || 0), 1e-6);
  const predicted = prediction?.class_name;

  return (
    <>
      <div className="tile-row">
        <StatTile
          label="Predicted class"
          value={predicted ? `${CLASS_LABELS[predicted]} (${predicted})` : offline ? 'Offline' : '—'}
          sub={predicted ? `Model ${prediction?.model_name || selectedModel}` : 'Live inference requires the API'}
          dot={predicted ? classColor(predicted) : undefined}
          hero
        />
        <StatTile label="Confidence" value={pct(prediction?.confidence)} sub={predicted ? 'Top-1 posterior probability' : '—'} />
        <StatTile label="Inference latency" value={prediction?.latency_ms != null ? `${fmt(prediction.latency_ms, 2)} ms` : '—'} sub="Single-segment forward pass" />
        <StatTile label="Held-out beats" value={fmtInt(metrics?.test_samples)} sub={`${rows.length} benchmarked models`} />
      </div>

      <div className="grid-main">
        <Card
          title={`ECG signal — record ${sample?.record_id || '—'}`}
          meta={`${sample?.channel || 'MLII'} · ${sample?.sample_rate_hz || 360} Hz · ${sample?.signal ? (sample.signal.length / (sample.sample_rate_hz || 360)).toFixed(1) : '—'} s window`}
        >
          <Waveform
            signal={prediction?.waveform?.length ? prediction.waveform : sample?.signal || []}
            sampleRate={sample?.sample_rate_hz || 360}
            color={classColor(predicted || 'N')}
          />
        </Card>
        <div className="stack">
          <Card title="Class probabilities" meta={predicted ? `argmax → ${predicted}` : 'awaiting live inference'}>
            <ProbabilityBars probabilities={prediction?.probabilities} />
          </Card>
          <Card title="Temporal attention" meta="Descriptive, not a causal explanation">
            <AttentionStrip attention={prediction?.attention || []} />
          </Card>
        </div>
      </div>

      <div className="grid-2">
        <Card title="Held-out predictions" meta={`Showing ${Math.min(12, records.length)} of ${records.length} fetched`}>
          <PredictionsTable rows={records} limit={12} />
        </Card>
        <Card title="Model comparison" meta="Macro-F1 on the held-out split">
          <div className="bar-list">
            {[...rows]
              .sort((a, b) => Number(b.MacroF1 || 0) - Number(a.MacroF1 || 0))
              .map((r) => (
                <BarRow
                  key={r.Model}
                  label={r.Model}
                  value={Number(r.MacroF1) || 0}
                  max={maxF1}
                  highlight={String(r.Model).toLowerCase() === selectedModel.toLowerCase()}
                />
              ))}
          </div>
          <p className="footnote">Selected model highlighted. Values are measured artifacts from {model?.model_version || 'the bundled benchmark'}.</p>
        </Card>
      </div>
    </>
  );
}

/* ----------------------------------------------------------- benchmark */

function Benchmark({ metrics, selectedModel }) {
  const rows = [...(metrics?.models || [])].sort((a, b) => Number(b.MacroF1 || 0) - Number(a.MacroF1 || 0));
  if (!rows.length) return <EmptyState title="No benchmark metrics available" />;
  return (
    <Card title="Model benchmark" meta="All classifiers on the fixed held-out split · sorted by Macro-F1">
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">Accuracy</th>
              <th className="num">Balanced acc.</th>
              <th className="num">Macro-F1</th>
              <th className="num">AUROC (OVR)</th>
              <th className="num">AUPRC (macro)</th>
              <th className="num">ECE</th>
              <th className="num">Params</th>
              <th className="num">Inference (ms)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.Model} className={String(r.Model).toLowerCase() === selectedModel.toLowerCase() ? 'is-selected' : ''}>
                <td><b>{r.Model}</b></td>
                <td className="num">{fmt(r.Accuracy)}</td>
                <td className="num">{fmt(r.BalancedAccuracy)}</td>
                <td className="num">{fmt(r.MacroF1)}</td>
                <td className="num">{fmt(r.AUROC_OVR)}</td>
                <td className="num">{fmt(r.AUPRC_macro)}</td>
                <td className="num">{fmt(r.ECE)}</td>
                <td className="num">{r.Params == null ? '—' : fmtInt(r.Params)}</td>
                <td className="num">{r.InferenceMs == null ? '—' : fmt(r.InferenceMs, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="footnote">
        Low absolute scores are the honest result of grouped-by-record evaluation with severe class imbalance — see the Research tab.
      </p>
    </Card>
  );
}

/* ------------------------------------------------------------- records */

function RecordsView({ records, research }) {
  const [filter, setFilter] = useState('All');
  const filtered = filter === 'All' ? records : records.filter((r) => r.true_class === filter || r.pred_class === filter);
  return (
    <>
      <Card
        title="Held-out predictions"
        meta={`${filtered.length} rows`}
        actions={
          <div className="pill-group">
            {['All', ...CLASSES].map((c) => (
              <button key={c} className={`pill ${filter === c ? 'is-active' : ''}`} onClick={() => setFilter(c)}>
                {c !== 'All' && <i className="class-dot" style={{ background: classColor(c) }} />}
                {c}
              </button>
            ))}
          </div>
        }
      >
        <PredictionsTable rows={filtered} limit={50} />
      </Card>
      <Card title="Per-record generalization" meta="Held-out Macro-F1 by MIT-BIH record">
        <div className="bar-list">
          {(research?.record_generalization || []).map((r) => (
            <BarRow key={r.record} label={String(r.record)} value={Number(r.MacroF1 || 0)} display={r.MacroF1 == null ? '—' : fmt(r.MacroF1, 2)} />
          ))}
        </div>
      </Card>
    </>
  );
}

/* --------------------------------------------------------------- about */

function About({ model }) {
  return (
    <Card title="About VIGIL">
      <div className="about">
        <p>
          <b>VIGIL</b> — interpretable temporal deep learning for ECG arrhythmia classification on the
          MIT-BIH Arrhythmia Database.
        </p>
        <ul>
          <li>5-class AAMI grouping (N · S · V · F · Q), grouped-by-record evaluation, fixed held-out split.</li>
          <li>PyTorch models served through FastAPI; React + Vite frontend.</li>
          <li>Artifacts: benchmark checkpoints, metrics, held-out predictions, temporal attention, Integrated Gradients, Phase 2/3 research reports.</li>
          <li>All displayed numbers are measured; negative results are reported as-is.</li>
        </ul>
        <p className="disclaimer">
          Retrospective research prototype only — <b>not a medical diagnostic device</b>.
        </p>
        <p className="footnote">Model version: {model?.model_version || '—'} · Framework: {model?.framework || 'PyTorch'}</p>
      </div>
    </Card>
  );
}

/* ----------------------------------------------------------------- app */

function initialTheme() {
  const stored = localStorage.getItem('vigil-theme');
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [theme, setTheme] = useState(initialTheme);
  const [view, setView] = useState('Overview');
  const [health, setHealth] = useState(null);
  const [model, setModel] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [research, setResearch] = useState(null);
  const [records, setRecords] = useState([]);
  const [sample, setSample] = useState(null);
  const [explanations, setExplanations] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [selectedModel, setSelectedModel] = useState('RNN');
  const [offline, setOffline] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('vigil-theme', theme);
  }, [theme]);

  async function runInference(signal, modelName, recordId) {
    if (!signal?.length) return;
    try {
      setBusy(true);
      const result = await api('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signal, model: modelName, record_id: recordId }),
      });
      setPrediction(result);
      setError('');
    } catch (e) {
      setError(`Inference failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  const analyze = (signal, modelName = selectedModel) => {
    if (!offline) return runInference(signal, modelName, sample?.record_id);
  };

  async function load() {
    setLoading(true);
    setError('');
    try {
      const [h, m, mt, r, s, e, rs] = await Promise.all([
        api('/health'),
        api('/model'),
        api('/metrics'),
        api('/records?limit=200'),
        api('/sample-signal'),
        api('/explanations'),
        api('/research'),
      ]);
      setHealth(h);
      setModel(m);
      setMetrics(mt);
      setRecords(r.records || []);
      setSample(s);
      setExplanations(e);
      setResearch(rs);
      setOffline(false);
      setSelectedModel(m.default_model || 'RNN');
      setLoading(false);
      await runInference(s.signal, m.default_model || 'RNN', s.record_id);
    } catch {
      const snap = demoSnapshot();
      setHealth(snap.health);
      setModel(snap.model);
      setMetrics(snap.metrics);
      setRecords(snap.records);
      setSample(snap.sample);
      setExplanations(snap.explanations);
      setResearch(snap.research);
      setPrediction(null);
      setOffline(true);
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!loading && !offline && sample?.signal) analyze(sample.signal, selectedModel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel]);

  const state = useMemo(
    () => ({ sample, metrics, records, model, explanations, selectedModel }),
    [sample, metrics, records, model, explanations, selectedModel],
  );

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Logo /></span>
          <span className="brand-name">
            <b>VIGIL</b>
            <small>ECG Arrhythmia Workbench</small>
          </span>
        </div>
        <nav className="nav">
          {VIEWS.map((v) => (
            <button key={v} className={view === v ? 'is-active' : ''} onClick={() => setView(v)}>
              {v}
            </button>
          ))}
        </nav>
        <div className="topbar-right">
          <span className={`status-pill ${offline ? 'is-offline' : 'is-live'}`}>
            <i />
            {offline ? 'Offline snapshot' : `Live · ${model?.model_version || 'API'}`}
          </span>
          <ThemeToggle theme={theme} onToggle={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} />
        </div>
      </header>

      <div className="controlbar">
        <button className="btn btn-primary" onClick={() => analyze(sample?.signal)} disabled={offline || busy || !sample}>
          {busy ? 'Analyzing…' : '▶ Run analysis'}
        </button>
        <button className="btn" onClick={load} disabled={loading}>↻ Refresh</button>
        <span className="control-sep" />
        <label className="control-field">
          Model
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} disabled={offline}>
            {(model?.available_models || ['RNN']).map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
        <span className="control-info">Record <b>{sample?.record_id || '—'}</b></span>
        <span className="control-info">Dataset <b>MIT-BIH</b></span>
        <span className="control-note">Research prototype · not a diagnostic device</span>
      </div>

      {offline && (
        <div className="banner banner-info">
          API unreachable — showing the bundled offline snapshot. Every number is a real measured artifact from the
          repository; live inference (prediction, probabilities, attention) needs the FastAPI server on
          <code> :8000</code>.
        </div>
      )}
      {error && <div className="banner banner-error">{error}</div>}

      <main className="content">
        {loading ? (
          <div className="loading">
            <span className="spinner" />
            Connecting to VIGIL API…
          </div>
        ) : view === 'Benchmark' ? (
          <Benchmark metrics={metrics} selectedModel={selectedModel} />
        ) : view === 'Records' ? (
          <RecordsView records={records} research={research} />
        ) : view === 'Research' ? (
          <>
            <h1 className="section-title">Phase 2 · Generalization study</h1>
            <Phase2Section research={research} />
            <h1 className="section-title">Phase 3 · Domain-generalized representation learning</h1>
            <Phase3Section research={research} />
          </>
        ) : view === 'About' ? (
          <About model={model} />
        ) : (
          <Overview state={state} prediction={prediction} offline={offline} />
        )}
      </main>

      <footer className="footer">
        <span>VIGIL · MIT-BIH Arrhythmia Database · research prototype only</span>
        <span>PyTorch · FastAPI · React + Vite</span>
      </footer>
    </div>
  );
}
