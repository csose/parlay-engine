import { useState } from 'react';
import type { AllTiersResult, TierLevel, EnrichedLeg } from '../types/engine';

interface Props {
  allTiers: AllTiersResult;
}

const TIER_ORDER: TierLevel[] = ['T1', 'T2', 'T3', 'T4', 'T5'];

function GradeChip({ grade }: { grade: string }) {
  return <span className={`pe-grade pe-grade--${grade.toLowerCase()}`}>{grade}</span>;
}

function LegRow({ leg }: { leg: EnrichedLeg }) {
  return (
    <tr className="pe-table__subrow">
      <td>{leg.player}</td>
      <td>{leg.market} {leg.line}+</td>
      <td><GradeChip grade={leg.grade} /></td>
      <td className="pe-mono">{(leg.baseTP * 100).toFixed(1)}%</td>
      <td className="pe-mono">{(leg.adjustedTP * 100).toFixed(1)}%</td>
      <td className="pe-mono">{(leg.volatilityPenalty * 100).toFixed(0)}%</td>
      <td className="pe-mono">{(leg.contextPenalty * 100).toFixed(0)}%</td>
      <td className="pe-mono">{leg.score.toFixed(3)}</td>
      <td>{leg.isCeilingProp ? '⚠️' : ''}{leg.isVolatile ? '🔥' : ''}</td>
    </tr>
  );
}

function TierRow({ tier, allTiers }: { tier: TierLevel; allTiers: AllTiersResult }) {
  const [expanded, setExpanded] = useState(false);
  const t = allTiers.tiers[tier];
  const ev = t.evaluation;

  return (
    <>
      <tr
        className={`pe-table__tier-row pe-table__tier-row--${tier.toLowerCase()} ${t.valid ? '' : 'pe-table__row--invalid'}`}
        onClick={() => setExpanded((x) => !x)}
        style={{ cursor: 'pointer' }}
      >
        <td><span className={`pe-tier-chip pe-tier-chip--${tier.toLowerCase()}`}>{tier}</span></td>
        <td>{t.label}</td>
        <td>{t.valid ? '✅' : '❌'}</td>
        <td>{t.legs.length}</td>
        <td className="pe-mono">{ev ? (ev.rawProb * 100).toFixed(3) + '%' : '—'}</td>
        <td className="pe-mono">{ev ? (ev.finalProb * 100).toFixed(3) + '%' : '—'}</td>
        <td className={`pe-mono ${ev && ev.ev >= 0 ? 'pe-val--green' : 'pe-val--red'}`}>
          {ev ? (ev.ev >= 0 ? '+' : '') + (ev.ev * 100).toFixed(1) + '%' : '—'}
        </td>
        <td className="pe-mono">{ev ? (ev.kelly * 100).toFixed(1) + '%' : '—'}</td>
        <td className="pe-mono">{ev ? ev.decimalOdds.toFixed(2) + '×' : '—'}</td>
        <td className="pe-expand-btn">{expanded ? '▲' : '▼'}</td>
      </tr>

      {expanded && t.legs.length > 0 && (
        <>
          <tr className="pe-table__subheader">
            <td colSpan={10}>
              <table className="pe-table pe-table--inner">
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Prop</th>
                    <th>Grade</th>
                    <th>Base TP</th>
                    <th>Adj. TP</th>
                    <th>Vol.</th>
                    <th>Ctx.</th>
                    <th>Score</th>
                    <th>Flags</th>
                  </tr>
                </thead>
                <tbody>
                  {t.legs.map((leg, i) => <LegRow key={i} leg={leg} />)}
                </tbody>
              </table>
            </td>
          </tr>
        </>
      )}

      {expanded && t.rejected.length > 0 && (
        <tr className="pe-table__subheader">
          <td colSpan={10}>
            <div className="pe-rejected-list">
              <strong>Rejected:</strong>{' '}
              {t.rejected.map((r, i) => (
                <span key={i} className="pe-rejected-item">
                  {r.leg.player} ({r.reason})
                </span>
              ))}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export function TierMatrix({ allTiers }: Props) {
  return (
    <section className="pe-card">
      <h2 className="pe-section-title">Tier Matrix</h2>
      <p className="pe-footnote">Click a tier row to expand / collapse its legs.</p>
      <div className="pe-table-scroll">
        <table className="pe-table">
          <thead>
            <tr>
              <th>Tier</th>
              <th>Label</th>
              <th>Valid</th>
              <th>Legs</th>
              <th>Naive Prob</th>
              <th>Adj. Prob</th>
              <th>EV</th>
              <th>Kelly</th>
              <th>Dec. Odds</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {TIER_ORDER.map((tier) => (
              <TierRow key={tier} tier={tier} allTiers={allTiers} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
