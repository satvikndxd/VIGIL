export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export const CLASSES = ['N', 'S', 'V', 'F', 'Q'];

export const CLASS_LABELS = {
  N: 'Normal',
  S: 'Supraventricular',
  V: 'Ventricular',
  F: 'Fusion',
  Q: 'Unknown',
};

/** Class colors live in CSS custom properties so both themes stay validated. */
export const classColor = (c) => `var(--c-${CLASSES.includes(c) ? c : 'Q'})`;

export const fmt = (n, d = 3) =>
  n === null || n === undefined || Number.isNaN(Number(n)) ? '—' : Number(n).toFixed(d);

export const pct = (n, d = 1) =>
  n === null || n === undefined || Number.isNaN(Number(n))
    ? '—'
    : `${(Number(n) * 100).toFixed(d)}%`;

export const fmtInt = (n) =>
  n === null || n === undefined || Number.isNaN(Number(n))
    ? '—'
    : Number(n).toLocaleString('en-US');

export async function api(path, options) {
  const r = await fetch(`${API_BASE}${path}`, options);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${path}`);
  return r.json();
}
