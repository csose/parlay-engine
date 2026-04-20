import type { ParlayEvaluation } from '../types/engine';

interface Props {
  evaluation: ParlayEvaluation;
}

export function Findings({ evaluation }: Props) {
  const { legs, removedLegs, correlation } = evaluation;

  const findings: { severity: 'info' | 'warn' | 'danger'; text: string }[] = [];

  findings.push({
    severity: 'info',
    text: `Pool: ${legs.length + removedLegs.length} legs analysed. ${legs.length} passed ATP gates; ${removedLegs.length} removed.`,
  });

  if (correlation.teamClusterTax > 0) {
    findings.push({
      severity: 'warn',
      text: `Team cluster tax applied: ${(correlation.teamClusterTax * 100).toFixed(1)}% — 3+ legs from the same team.`,
    });
  }
  if (correlation.gameClusterTax > 0) {
    findings.push({
      severity: 'warn',
      text: `Game cluster tax applied: ${(correlation.gameClusterTax * 100).toFixed(1)}% — multiple legs from the same game.`,
    });
  }
  if (correlation.volatilityStackTax > 0) {
    findings.push({
      severity: 'warn',
      text: `Volatility stack tax: ${(correlation.volatilityStackTax * 100).toFixed(1)}% — 2+ high-threshold scoring legs stacked.`,
    });
  }

  const ceilingLegs = legs.filter((l) => l.isCeilingProp);
  if (ceilingLegs.length > 0) {
    findings.push({
      severity: 'danger',
      text: `Ceiling props detected: ${ceilingLegs.map((l) => `${l.player} ${l.market}${l.line}+`).join(', ')}. These are banned in 10+ leg parlays.`,
    });
  }

  const volatileLegs = legs.filter((l) => l.isVolatile);
  if (volatileLegs.length > 0) {
    findings.push({
      severity: 'warn',
      text: `Volatile legs (vol. penalty ≥ 18%): ${volatileLegs.map((l) => `${l.player} ${l.market}${l.line}+`).join(', ')}.`,
    });
  }

  if (evaluation.ev < 0) {
    findings.push({
      severity: 'danger',
      text: `Negative expected value: ${(evaluation.ev * 100).toFixed(1)}%. Consider tightening leg selection.`,
    });
  }

  if (removedLegs.length > 0) {
    findings.push({
      severity: 'warn',
      text: `Removed legs (ATP < minimum): ${removedLegs.map((l) => `${l.player} ${l.market}${l.line}+ (ATP ${(l.adjustedTP * 100).toFixed(1)}%)`).join(', ')}.`,
    });
  }

  return (
    <section className="pe-card">
      <h2 className="pe-section-title">Findings</h2>
      <ul className="pe-findings-list">
        {findings.map((f, i) => (
          <li key={i} className={`pe-finding pe-finding--${f.severity}`}>
            {f.severity === 'danger' ? '🔴' : f.severity === 'warn' ? '🟡' : 'ℹ️'} {f.text}
          </li>
        ))}
      </ul>
    </section>
  );
}
