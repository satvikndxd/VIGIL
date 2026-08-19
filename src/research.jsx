import { CLASSES, CLASS_LABELS, classColor, fmt, pct } from './lib.js';
import { BarRow, Card, ClassBadge, EmptyState, StatTile } from './components.jsx';

export function Phase2Headline({ research }) {
  const f = research?.final_metrics || {};
  const base = research?.final_comparison?.find?.((x) => x.metric === 'MacroF1');
  return (
    <div className="tile-row">
      <StatTile label="Research final model" value={f.model || '—'} sub="Frozen after Phase 2 validation · no test tuning" hero />
      <StatTile label="Macro-F1" value={fmt(f.MacroF1)} sub={`Baseline Δ ${fmt(base?.absolute_delta)}`} />
      <StatTile label="Macro-AUROC" value={fmt(f.MacroAUROC)} sub={`Macro-AUPRC ${fmt(f.MacroAUPRC)}`} />
      <StatTile label="Minority recall" value={`N ${pct(f.N_recall)} · F ${pct(f.F_recall)}`} sub="Reported as measured — no fabricated improvement" />
    </div>
  );
}

export function RecordGeneralization({ rows = [] }) {
  if (!rows.length) return <EmptyState title="No record-level results" />;
  return (
    <div className="bar-list">
      {rows.map((r) => (
        <BarRow
          key={r.record}
          label={String(r.record)}
          value={Number(r.MacroF1 || 0)}
          display={r.MacroF1 == null ? '—' : fmt(r.MacroF1, 2)}
        />
      ))}
    </div>
  );
}

export function MinorityPanel({ research }) {
  const f = research?.final_metrics || {};
  const rows = research?.final_per_class || [];
  const recall = (c) => {
    const row = rows.find((x) => x.class === c);
    return row?.recall ?? (c === 'N' ? f.N_recall : c === 'F' ? f.F_recall : null);
  };
  return (
    <div className="minority-list">
      {CLASSES.map((c) => (
        <div className="minority-row" key={c}>
          <span>
            <i className="class-dot" style={{ background: classColor(c) }} />
            {c} · {CLASS_LABELS[c]}
          </span>
          <b>{recall(c) == null ? '—' : pct(recall(c))}</b>
        </div>
      ))}
      <p className="footnote">
        Final retained model: {f.model || '—'}. N and F recall remain unresolved in the frozen result.
      </p>
    </div>
  );
}

export function SymbolShift({ rows = [] }) {
  const visible = rows.filter((r) => Number(r.test_count) > 0).slice(0, 14);
  if (!visible.length) return <EmptyState title="No symbol shift data" />;
  return (
    <div className="table-wrap">
      <table className="data-table compact">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Map</th>
            <th className="num">Train</th>
            <th className="num">Test</th>
            <th className="num">Error</th>
            <th className="num">Recall</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((r) => (
            <tr key={r.original_symbol}>
              <td><b>{r.original_symbol}</b></td>
              <td><ClassBadge code={r.mapped_class} /></td>
              <td className="num">{r.train_count}</td>
              <td className="num">{r.test_count}</td>
              <td className="num">{r.test_error_rate == null ? '—' : pct(r.test_error_rate)}</td>
              <td className="num">{r.per_symbol_recall == null ? '—' : pct(r.per_symbol_recall)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RobustnessTable({ rows = [] }) {
  if (!rows.length) return <EmptyState title="No robustness results" />;
  return (
    <div className="table-wrap">
      <table className="data-table compact">
        <thead>
          <tr>
            <th>Condition</th>
            <th className="num">Macro-F1</th>
            <th className="num">Δ vs clean</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.corruption}>
              <td>{r.corruption}</td>
              <td className="num">{fmt(r.MacroF1)}</td>
              <td className="num">{fmt(r.MacroF1_delta ?? r.MacroF1_drop_from_clean, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="footnote">Controlled perturbations only; not evidence of noise immunity.</p>
    </div>
  );
}

export function Phase2Section({ research }) {
  return (
    <>
      <Phase2Headline research={research} />
      <div className="grid-2">
        <Card title="Per-record generalization" meta="Held-out Macro-F1 by record">
          <RecordGeneralization rows={research?.record_generalization} />
        </Card>
        <Card title="Minority-class recall" meta="Final frozen model">
          <MinorityPanel research={research} />
        </Card>
      </div>
      <div className="grid-2">
        <Card title="Original symbol shift" meta="Train → test distribution per MIT-BIH symbol">
          <SymbolShift rows={research?.symbol_shift} />
        </Card>
        <Card title="Robustness" meta="Final model under controlled corruption">
          <RobustnessTable rows={research?.robustness} />
        </Card>
      </div>
    </>
  );
}

export function Phase3Section({ research }) {
  const p3 = research?.phase3 || {};
  const f = p3.metrics || {};
  const ablation = Array.isArray(p3.representation_ablation) ? p3.representation_ablation : [];
  const calibration = Array.isArray(p3.calibration) ? p3.calibration : [];
  const robustness = Array.isArray(p3.robustness) ? p3.robustness : [];
  return (
    <>
      <div className="tile-row">
        <StatTile
          label="Phase 3 decision"
          value={f.replacement_gate_passed ? 'Candidate retained' : 'Frozen baseline retained'}
          sub={`Explored: ${f.selected_phase3_candidate || '—'}`}
          hero
        />
        <StatTile label="Macro-F1" value={fmt(f.MacroF1)} sub="Δ 0.000 vs frozen baseline" />
        <StatTile label="Macro-AUPRC" value={fmt(f.MacroAUPRC)} sub={`Macro-AUROC ${fmt(f.MacroAUROC)}`} />
        <StatTile label="N / F recall" value={`${pct(f.N_recall)} · ${pct(f.F_recall)}`} sub="No forced improvement" />
      </div>
      <div className="grid-3">
        <Card title="Representation ablation">
          <div className="table-wrap">
            <table className="data-table compact">
              <thead>
                <tr><th>Representation</th><th className="num">F1</th><th className="num">AUPRC</th><th className="num">AUROC</th></tr>
              </thead>
              <tbody>
                {ablation.map((r, i) => (
                  <tr key={r.representation || i}>
                    <td>{r.representation || '—'}</td>
                    <td className="num">{fmt(r.MacroF1)}</td>
                    <td className="num">{fmt(r.MacroAUPRC)}</td>
                    <td className="num">{fmt(r.MacroAUROC)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <Card title="Calibration" meta="Validation-fit temperature">
          <div className="table-wrap">
            <table className="data-table compact">
              <thead>
                <tr><th>Model</th><th className="num">ECE</th><th className="num">Brier</th><th className="num">F1</th></tr>
              </thead>
              <tbody>
                {calibration.map((r, i) => (
                  <tr key={r.model || i}>
                    <td>{r.model || '—'}</td>
                    <td className="num">{fmt(r.ECE)}</td>
                    <td className="num">{fmt(r.Brier)}</td>
                    <td className="num">{fmt(r.MacroF1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <Card title="Robustness" meta="Baseline vs Phase 3 candidate">
          <div className="table-wrap">
            <table className="data-table compact">
              <thead>
                <tr><th>Model</th><th>Condition</th><th className="num">F1</th><th className="num">Δ</th></tr>
              </thead>
              <tbody>
                {robustness.slice(0, 10).map((r, i) => (
                  <tr key={`${r.model}-${r.corruption}-${i}`}>
                    <td>{r.model || '—'}</td>
                    <td>{r.corruption || '—'}</td>
                    <td className="num">{fmt(r.MacroF1)}</td>
                    <td className="num">{fmt(r.MacroF1_delta, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  );
}
