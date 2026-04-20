import type { ExtractionAnalysis } from '../types/engine';

interface Props {
  extraction: ExtractionAnalysis;
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const cls = pct >= 90 ? 'high' : pct >= 75 ? 'medium' : 'low';
  return (
    <div className="pe-conf-bar">
      <div className={`pe-conf-bar__fill pe-conf-bar__fill--${cls}`} style={{ width: `${pct}%` }} />
      <span className="pe-conf-bar__label">{pct}%</span>
    </div>
  );
}

export function ExtractionStatus({ extraction }: Props) {
  const icon = extraction.success ? '✅' : '❌';
  return (
    <section className="pe-card">
      <h2 className="pe-section-title">Extraction Status</h2>
      <div className="pe-kv-grid">
        <span className="pe-kv__key">Result</span>
        <span className="pe-kv__val">{icon} {extraction.success ? 'SUCCESS' : 'FAILED'}</span>

        <span className="pe-kv__key">Source type</span>
        <span className="pe-kv__val pe-badge pe-badge--neutral">{extraction.sourceType || 'unknown'}</span>

        <span className="pe-kv__key">Sportsbook</span>
        <span className="pe-kv__val">{extraction.sportsbook ?? '—'}</span>

        <span className="pe-kv__key">Bet type</span>
        <span className="pe-kv__val">{extraction.betType ?? '—'}</span>

        <span className="pe-kv__key">Bet ID</span>
        <span className="pe-kv__val pe-mono">{extraction.betId ?? '—'}</span>

        <span className="pe-kv__key">Stake</span>
        <span className="pe-kv__val">{extraction.stake !== null ? `$${extraction.stake.toFixed(2)}` : '—'}</span>

        <span className="pe-kv__key">Total odds</span>
        <span className="pe-kv__val">
          {extraction.totalOddsAmerican !== null
            ? `${extraction.totalOddsAmerican > 0 ? '+' : ''}${extraction.totalOddsAmerican}`
            : '—'}
        </span>

        <span className="pe-kv__key">Doc confidence</span>
        <span className="pe-kv__val"><ConfidenceBar value={extraction.confidence.document} /></span>

        <span className="pe-kv__key">Legs parsed</span>
        <span className="pe-kv__val">{extraction.legCount}</span>
      </div>
    </section>
  );
}
