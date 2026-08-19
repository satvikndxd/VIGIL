/**
 * Offline snapshot for when the FastAPI backend is unreachable.
 *
 * Every number here is a real, measured artifact copied verbatim from
 * `ml/metrics`, `ml/predictions` and `research_results` at build time
 * (see `src/demo-data.json`). Nothing is fabricated. Live inference
 * (prediction, probabilities, attention) is simply unavailable offline
 * and the UI renders explicit empty states for it.
 */
import data from './demo-data.json';
import sampleWaveform from '../ml/sample_waveform.json';

export function demoSnapshot() {
  return {
    health: { status: 'offline', model_online: false },
    model: data.model,
    metrics: data.metrics,
    records: data.records,
    sample: sampleWaveform,
    explanations: data.explanations,
    research: data.research,
  };
}
