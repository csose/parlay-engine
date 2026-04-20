import type { AllTiersResult, TierLevel } from '../types/engine';
import { americanToDecimal } from '../engine/core';

interface Props {
  allTiers: AllTiersResult;
}

const TIER_ORDER: TierLevel[] = ['T1', 'T2', 'T3', 'T4', 'T5'];

function decimalToAmerican(dec: number): string {
  if (dec <= 1) return 'N/A';
  if (dec >= 2) {
    const v = Math.round((dec - 1) * 100);
    return `+${v}`;
  }
  const v = Math.round(-100 / (dec - 1));
  return `${v}`;
}

export function PayoutSummary({ allTiers }: Props) {
  return (
    <section className="pe-card">
      <h2 className="pe-section-title">Payout Summary</h2>
      <table className="pe-table">
        <thead>
          <tr>
            <th>Tier</th>
            <th>Label</th>
            <th>Legs</th>
            <th>American Odds</th>
            <th>Decimal Odds</th>
            <th>$5 Return</th>
          </tr>
        </thead>
        <tbody>
          {TIER_ORDER.map((tier) => {
            const t = allTiers.tiers[tier];
            const ev = t.evaluation;
            const decOdds = ev ? ev.decimalOdds : null;
            const americanStr = ev ? decimalToAmerican(decOdds!) : '—';
            const returnOn5 = decOdds ? (decOdds * 5).toFixed(2) : '—';
            return (
              <tr key={tier} className={t.valid ? '' : 'pe-table__row--invalid'}>
                <td><span className={`pe-tier-chip pe-tier-chip--${tier.toLowerCase()}`}>{tier}</span></td>
                <td>{t.label}</td>
                <td>{t.legs.length || '—'}</td>
                <td className="pe-mono">{americanStr}</td>
                <td className="pe-mono">{decOdds ? decOdds.toFixed(2) : '—'}</td>
                <td className="pe-mono pe-val--green">{returnOn5 !== '—' ? `$${returnOn5}` : '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="pe-footnote">
        Decimal odds = combined product of all leg odds. $5 return = decimal × $5 stake (includes stake).
        Powered by FanDuel SGP+ correlation rules.
      </p>
      <p className="pe-footnote">
        Odds shown use americanToDecimal conversion: +odds → 1 + odds/100; −odds → 1 + 100/|odds|.{' '}
        <em>Reference: <code>americanToDecimal</code> in engine/core.ts.</em>
      </p>
    </section>
  );
}

// re-export so App can use it without a separate import
export { americanToDecimal };
