import { useMemo, useRef, useState } from 'react';
import { CLASSES, CLASS_LABELS, classColor, fmt, pct } from './lib.js';

/* ---------------------------------------------------------------- cards */

export function Card({ title, meta, actions, children, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          <div>
            <h2>{title}</h2>
            {meta && <span className="card-meta">{meta}</span>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      <div className="card-body">{children}</div>
    </section>
  );
}

export function StatTile({ label, value, sub, dot, hero = false }) {
  return (
    <div className={`stat-tile ${hero ? 'stat-hero' : ''}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">
        {dot && <i className="class-dot" style={{ background: dot }} />}
        {value}
      </span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

export function ClassBadge({ code }) {
  if (!code || !CLASS_LABELS[code]) return <span className="muted">—</span>;
  return (
    <span className="class-badge">
      <i className="class-dot" style={{ background: classColor(code) }} />
      {code}
    </span>
  );
}

export function StatusBadge({ ok }) {
  return (
    <span className={`status-badge ${ok ? 'is-good' : 'is-review'}`}>
      {ok ? '✓ Correct' : '✕ Review'}
    </span>
  );
}

export function EmptyState({ title, body }) {
  return (
    <div className="empty-state">
      <b>{title}</b>
      {body && <p>{body}</p>}
    </div>
  );
}

/* ------------------------------------------------------------- waveform */

const W = 1000;
const H = 300;
const PAD = { top: 14, right: 14, bottom: 30, left: 52 };

export function Waveform({ signal = [], sampleRate = 360, color = 'var(--c-N)' }) {
  const wrapRef = useRef(null);
  const [hover, setHover] = useState(null);

  const model = useMemo(() => {
    if (!signal.length) return null;
    const stride = Math.max(1, Math.ceil(signal.length / 1400));
    const values = [];
    for (let i = 0; i < signal.length; i += stride) values.push({ i, v: Number(signal[i]) });
    let min = Infinity;
    let max = -Infinity;
    for (const p of values) {
      if (p.v < min) min = p.v;
      if (p.v > max) max = p.v;
    }
    const range = max - min || 1;
    const innerW = W - PAD.left - PAD.right;
    const innerH = H - PAD.top - PAD.bottom;
    const duration = signal.length / sampleRate;
    const x = (i) => PAD.left + (i / Math.max(1, signal.length - 1)) * innerW;
    const y = (v) => PAD.top + (1 - (v - min) / range) * innerH;
    const path = values
      .map((p, k) => `${k === 0 ? 'M' : 'L'}${x(p.i).toFixed(1)},${y(p.v).toFixed(1)}`)
      .join('');
    return { values, min, max, duration, x, y, path, innerW, innerH };
  }, [signal, sampleRate]);

  if (!model) {
    return <EmptyState title="No signal loaded" body="Waiting for a sample waveform." />;
  }

  const seconds = Math.max(1, Math.round(model.duration));
  const majors = Array.from({ length: seconds + 1 }, (_, s) => s);
  const yTicks = [model.min, (model.min + model.max) / 2, model.max];

  function onMove(e) {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const frac = Math.min(1, Math.max(0, (px - PAD.left) / model.innerW));
    const idx = Math.round(frac * (model.values.length - 1));
    const point = model.values[idx];
    if (!point) return;
    setHover({
      x: model.x(point.i),
      y: model.y(point.v),
      t: point.i / sampleRate,
      v: point.v,
      leftPct: (model.x(point.i) / W) * 100,
      topPct: (model.y(point.v) / H) * 100,
    });
  }

  return (
    <div className="waveform" ref={wrapRef} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="ECG waveform">
        {/* minor grid — 200 ms */}
        {Array.from({ length: seconds * 5 + 1 }, (_, k) => {
          const gx = PAD.left + (k / (seconds * 5)) * model.innerW;
          return <line key={`m${k}`} x1={gx} y1={PAD.top} x2={gx} y2={H - PAD.bottom} className="grid-minor" />;
        })}
        {/* major grid — 1 s */}
        {majors.map((s) => {
          const gx = PAD.left + (s / seconds) * model.innerW;
          return <line key={`s${s}`} x1={gx} y1={PAD.top} x2={gx} y2={H - PAD.bottom} className="grid-major" />;
        })}
        {yTicks.map((v, k) => (
          <g key={`y${k}`}>
            <line x1={PAD.left} y1={model.y(v)} x2={W - PAD.right} y2={model.y(v)} className="grid-major" />
            <text x={PAD.left - 8} y={model.y(v) + 3} className="axis-text" textAnchor="end">
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {majors.map((s) => (
          <text
            key={`t${s}`}
            x={PAD.left + (s / seconds) * model.innerW}
            y={H - 10}
            className="axis-text"
            textAnchor="middle"
          >
            {s}s
          </text>
        ))}
        <path d={model.path} className="wave-line" style={{ stroke: color }} />
        {hover && (
          <g>
            <line x1={hover.x} y1={PAD.top} x2={hover.x} y2={H - PAD.bottom} className="crosshair" />
            <circle cx={hover.x} cy={hover.y} r={5} style={{ fill: color }} className="hover-dot" />
          </g>
        )}
      </svg>
      {hover && (
        <div
          className="chart-tooltip"
          style={{
            left: `${Math.min(88, Math.max(6, hover.leftPct))}%`,
            top: `${Math.max(4, hover.topPct - 14)}%`,
          }}
        >
          <b>{(hover.t * 1000).toFixed(0)} ms</b>
          <span>{hover.v.toFixed(3)} mV</span>
        </div>
      )}
    </div>
  );
}

/* --------------------------------------------------- probability + bars */

export function ProbabilityBars({ probabilities }) {
  const has = probabilities && Object.keys(probabilities).length > 0;
  const top = has
    ? CLASSES.reduce((a, b) => (Number(probabilities[a] || 0) >= Number(probabilities[b] || 0) ? a : b))
    : null;
  return (
    <div className="prob-list">
      {CLASSES.map((c) => {
        const v = has ? Number(probabilities[c] || 0) : null;
        return (
          <div className={`prob-row ${c === top ? 'is-top' : ''}`} key={c} title={`${CLASS_LABELS[c]}: ${pct(v)}`}>
            <span className="prob-label">
              <i className="class-dot" style={{ background: classColor(c) }} />
              {c} · {CLASS_LABELS[c]}
            </span>
            <span className="bar-track">
              <i
                className="bar-fill"
                style={{ width: v === null ? 0 : `${Math.max(v > 0 ? 1 : 0, v * 100)}%`, background: classColor(c) }}
              />
            </span>
            <b className="bar-value">{v === null ? '—' : pct(v)}</b>
          </div>
        );
      })}
    </div>
  );
}

export function AttentionStrip({ attention = [] }) {
  if (!attention.length) {
    return <EmptyState title="No attention weights" body="Run live inference with an attention-enabled checkpoint." />;
  }
  const max = Math.max(...attention.map((a) => Number(a.weight) || 0), 1e-6);
  return (
    <div className="attention-strip" role="img" aria-label="Temporal attention per beat">
      {attention.map((a) => {
        const ratio = (Number(a.weight) || 0) / max;
        return (
          <div className="attention-col" key={a.beat_position} title={`Beat ${a.beat_position + 1}: ${fmt(a.weight, 4)}`}>
            <span className="attention-value">{fmt(a.weight, 2)}</span>
            <span
              className="attention-bar"
              style={{ height: `${8 + ratio * 72}%`, opacity: 0.35 + ratio * 0.65 }}
            />
            <small>B{a.beat_position + 1}</small>
          </div>
        );
      })}
    </div>
  );
}

/** Horizontal labeled bar — single-hue magnitude comparison. */
export function BarRow({ label, value, max = 1, display, highlight = false, color = 'var(--accent)' }) {
  const ratio = Math.max(0, Math.min(1, Number(value || 0) / max));
  return (
    <div className={`bar-row ${highlight ? 'is-highlight' : ''}`} title={`${label}: ${display ?? fmt(value)}`}>
      <span className="bar-row-label">{label}</span>
      <span className="bar-track">
        <i className="bar-fill" style={{ width: `${Math.max(ratio * 100, value > 0 ? 1.5 : 0)}%`, background: color }} />
      </span>
      <b className="bar-value">{display ?? fmt(value)}</b>
    </div>
  );
}
