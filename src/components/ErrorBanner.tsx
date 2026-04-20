import type { EnrichedLeg, AnalysisError } from '../types/engine';

interface Props {
  removedLegs: EnrichedLeg[];
  warnings: AnalysisError[];
  errors: AnalysisError[];
}

export function ErrorBanner({ removedLegs, warnings, errors }: Props) {
  if (!removedLegs.length && !warnings.length && !errors.length) return null;

  return (
    <section className="pe-card pe-card--warn">
      <h2 className="pe-section-title">Error / Warning Banner</h2>

      {errors.map((e, i) => (
        <div key={i} className="pe-banner-row pe-banner-row--error">
          ❌ <strong>{e.code}</strong>: {e.message}
        </div>
      ))}

      {warnings.map((w, i) => (
        <div key={i} className="pe-banner-row pe-banner-row--warn">
          ⚠️ <strong>{w.code}</strong>: {w.message}
        </div>
      ))}

      {removedLegs.length > 0 && (
        <details className="pe-collapsible">
          <summary className="pe-collapsible__trigger">
            ❌ {removedLegs.length} leg{removedLegs.length !== 1 ? 's' : ''} removed (ATP below threshold)
          </summary>
          <ul className="pe-list">
            {removedLegs.map((leg, i) => (
              <li key={i} className="pe-list__item pe-list__item--removed">
                <strong>{leg.player}</strong> — {leg.market} {leg.line}+
                &nbsp;(ATP {(leg.adjustedTP * 100).toFixed(1)}%, required {(leg.adjustedTP * 100).toFixed(1)}%)
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
