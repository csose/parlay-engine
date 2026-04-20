export function Education() {
  return (
    <section className="pe-card">
      <h2 className="pe-section-title">Education</h2>

      <details className="pe-edu-block">
        <summary>FanDuel SGP+ Rules</summary>
        <p>
          Same Game Parlay+ (SGP+) allows combining player props from multiple games on a single slip.
          FanDuel applies correlation boosts or cuts based on how legs interact within a game.
          Duplicate players, directly conflicting outcomes (e.g. same player high points + low minutes),
          and rule-violating combinations are auto-rejected.
        </p>
      </details>

      <details className="pe-edu-block">
        <summary>Probability-First Tier Philosophy</summary>
        <p>
          The engine prioritises survivability over payout. T1 uses the highest-ATP legs with the
          tightest constraints; T5 uses more legs but enforces a higher per-leg ATP floor.
          Never maximise odds first — a high-payout parlay with weak legs is a losing strategy long-term.
          EV and Kelly fraction confirm whether the parlay has positive expectation; they are tie-breakers,
          not primary criteria.
        </p>
      </details>

      <details className="pe-edu-block">
        <summary>Why Longer Parlays Need Safer Legs</summary>
        <p>
          Parlay probability = product of all leg probabilities. Adding a 0.70 ATP leg to a 5-leg parlay
          multiplies your probability by 0.70. A 10-leg parlay where every leg is 0.80 ATP gives a
          joint probability of 0.80^10 ≈ 10.7%. Drop any leg to 0.68 ATP and that falls to ≈ 9.5%.
          The leg-count governor enforces tighter minimums as leg count rises to counteract this decay.
        </p>
      </details>

      <details className="pe-edu-block">
        <summary>Why Ceiling Props Are Dangerous in Long Ladders</summary>
        <p>
          Ceiling props (e.g. 30+ points, 10+ assists) have high volatility penalties because
          they depend on rare game states — blowouts, foul trouble, overtime, or exceptional hot
          streaks. One ceiling leg with a 0.22 volatility penalty and a 0.10 context penalty can
          drop an effective ATP from 0.75 to ≈ 0.51. In a 10-leg parlay that one leg alone
          reduces the joint probability by ~32%. They are banned in T4/T5 for this reason.
        </p>
      </details>

      <details className="pe-edu-block">
        <summary>ATP Formula</summary>
        <p>
          <code>ATP = BaseTP × (1 − VolatilityPenalty) × (1 − ContextPenalty)</code>
          <br />
          BaseTP is computed from a Normal CDF (points) or Poisson CDF (assists/rebounds),
          or a user-supplied override. Penalties are additive within each category (volatility
          and context) and then multiplicatively applied to BaseTP.
        </p>
      </details>

      <details className="pe-edu-block">
        <summary>Kelly Criterion</summary>
        <p>
          <code>f* = (p × (b+1) − 1) / b</code> where <code>b = decimalOdds − 1</code>.
          <br />
          The engine uses 25% fractional Kelly (max). A full-Kelly bet on a parlay is extremely
          aggressive; 25% fractional Kelly is conservative enough for long-term bankroll growth
          with reasonable drawdown risk.
        </p>
      </details>
    </section>
  );
}
