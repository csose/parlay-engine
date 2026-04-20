import type { AllTiersResult, ParlayEvaluation } from '../types/engine';

interface Props {
  allTiers: AllTiersResult;
  evaluation: ParlayEvaluation;
}

export function SummaryMetrics({ allTiers, evaluation }: Props) {
  const totalRemoved = evaluation.removedLegs.length;
  const validTiers = Object.values(allTiers.tiers).filter((t) => t.valid).length;
  const maxEV = Math.max(
    ...Object.values(allTiers.tiers)
      .filter((t) => t.evaluation)
      .map((t) => t.evaluation!.ev),
    evaluation.ev,
  );
  const maxKelly = Math.max(
    ...Object.values(allTiers.tiers)
      .filter((t) => t.evaluation)
      .map((t) => t.evaluation!.kelly),
    evaluation.kelly,
  );

  return (
    <section className="pe-card">
      <h2 className="pe-section-title">Summary Metrics</h2>
      <div className="pe-metrics-grid">
        <div className="pe-metric">
          <span className="pe-metric__val">{allTiers.poolSize}</span>
          <span className="pe-metric__label">Raw data points</span>
        </div>
        <div className="pe-metric">
          <span className="pe-metric__val">{evaluation.legCount}</span>
          <span className="pe-metric__label">Eligible plays</span>
        </div>
        <div className="pe-metric">
          <span className="pe-metric__val pe-val--red">{totalRemoved}</span>
          <span className="pe-metric__label">Legs removed (ATP)</span>
        </div>
        <div className="pe-metric">
          <span className="pe-metric__val">{validTiers} / 5</span>
          <span className="pe-metric__label">Valid tiers</span>
        </div>
        <div className="pe-metric">
          <span className={`pe-metric__val ${maxEV >= 0 ? 'pe-val--green' : 'pe-val--red'}`}>
            {(maxEV * 100).toFixed(1)}%
          </span>
          <span className="pe-metric__label">Max EV</span>
        </div>
        <div className="pe-metric">
          <span className="pe-metric__val pe-val--gold">{(maxKelly * 100).toFixed(1)}%</span>
          <span className="pe-metric__label">Max Kelly</span>
        </div>
        <div className="pe-metric">
          <span className="pe-metric__val">{(evaluation.finalProb * 100).toFixed(2)}%</span>
          <span className="pe-metric__label">Adj. parlay prob.</span>
        </div>
        <div className="pe-metric">
          <span className="pe-metric__val">{(evaluation.corrSimProb * 100).toFixed(2)}%</span>
          <span className="pe-metric__label">MC correlated prob.</span>
        </div>
      </div>
    </section>
  );
}
