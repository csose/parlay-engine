import type { RawLeg, EnrichedLeg, CorrelationInfo, ParlayEvaluation, EngineConfig } from '../types/engine';

export const ENGINE_CONFIG: Readonly<EngineConfig> = {
  defaultStdDevPoints: 6,
  monteCarloIterations: 10000,
  maxKellyFraction: 0.25,
  minKellyFraction: 0,
};

// ─── Math utilities ───────────────────────────────────────────────────────────

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function americanToDecimal(odds: number): number {
  if (odds === 0) return 1;
  return odds > 0 ? 1 + odds / 100 : 1 + 100 / Math.abs(odds);
}

function factorial(n: number): number {
  // use log-space for n > 20 to avoid float overflow
  if (n <= 1) return 1;
  if (n <= 20) {
    let r = 1;
    for (let i = 2; i <= n; i++) r *= i;
    return r;
  }
  let logSum = 0;
  for (let i = 2; i <= n; i++) logSum += Math.log(i);
  return Math.exp(logSum);
}

// Abramowitz & Stegun 7.1.26 polynomial approximation
function erf(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  const ax = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax));
  return sign * y;
}

function normalCDF(x: number): number {
  return (1 + erf(x / Math.sqrt(2))) / 2;
}

// ─── Probability estimators ───────────────────────────────────────────────────

function probOverPoints(line: number, mean: number, stdDev?: number): number {
  const sd = stdDev ?? ENGINE_CONFIG.defaultStdDevPoints;
  if (sd <= 0) return 0.5;
  const z = (line - mean) / sd;
  return clamp(1 - normalCDF(z), 0.0001, 0.9999);
}

function poissonProbGE(lambda: number, k: number): number {
  if (lambda <= 0) return 0.0001;
  let cdf = 0;
  for (let i = 0; i < k; i++) {
    cdf += (Math.pow(lambda, i) * Math.exp(-lambda)) / factorial(i);
  }
  return clamp(1 - cdf, 0.0001, 0.9999);
}

// ─── Leg-count governor ───────────────────────────────────────────────────────

export function minATPForLegCount(nLegs: number): number {
  if (nLegs <= 4) return 0.62;
  if (nLegs <= 6) return 0.68;
  if (nLegs <= 9) return 0.72;
  if (nLegs <= 12) return 0.76;
  if (nLegs <= 16) return 0.80;
  return 0.84;
}

// ─── Penalty formulas ─────────────────────────────────────────────────────────

export function volatilityPenalty(leg: RawLeg): number {
  const { market, line } = leg;
  let v = 0;

  if (market === 'PTS') {
    if (line >= 40) v = 0.50;
    else if (line >= 35) v = 0.35;
    else if (line >= 30) v = 0.22;
    else if (line >= 25) v = 0.10;
    else if (line >= 20) v = 0.04;
  }

  if (market === 'AST') {
    if (line >= 12) v = Math.max(v, 0.35);
    else if (line >= 10) v = Math.max(v, 0.24);
    else if (line >= 8) v = Math.max(v, 0.16);
    else if (line >= 6) v = Math.max(v, 0.08);
    else if (line >= 4) v = Math.max(v, 0.04);
  }

  if (market === 'REB') {
    if (line >= 12) v = Math.max(v, 0.24);
    else if (line >= 10) v = Math.max(v, 0.15);
    else if (line >= 8) v = Math.max(v, 0.06);
    else if (line >= 6) v = Math.max(v, 0.04);
    else if (line >= 4) v = Math.max(v, 0.02);
  }

  if (market === '3PM') {
    if (line >= 5) v = Math.max(v, 0.42);
    else if (line >= 4) v = Math.max(v, 0.30);
    else if (line >= 3) v = Math.max(v, 0.18);
    else if (line >= 2) v = Math.max(v, 0.10);
    else if (line >= 1) v = Math.max(v, 0.04);
  }

  return clamp(v, 0, 0.9);
}

export function contextPenalty(leg: RawLeg): number {
  let p = 0;
  if (leg.isRookieStarter) p += 0.08;
  if (leg.isBenchOrRotationRisk) p += 0.12;
  if (leg.minutesCapRisk) p += 0.15;
  if (leg.injuryReturn || leg.isQuestionable) p += 0.12;
  if (leg.backToBack) p += 0.06;
  if (leg.highFoulRiskBig) p += 0.06;
  if (leg.isUnderdog && leg.market === 'PTS') p += 0.05;
  if (leg.blowoutRisk) p += 0.10;
  if (leg.usageSplitRisk) p += 0.06;
  return clamp(p, 0, 0.9);
}

function computeBaseTrueProb(leg: RawLeg): number {
  const { market, line, mean, stdDev, baseProb } = leg;
  if (typeof baseProb === 'number') return clamp(baseProb, 0.0001, 0.9999);
  if (market === 'PTS' && typeof mean === 'number') return probOverPoints(line, mean, stdDev);
  if ((market === 'REB' || market === 'AST') && typeof mean === 'number') return poissonProbGE(mean, line);
  return 0.5;
}

export function computeAdjustedProb(leg: RawLeg): { baseTP: number; adjustedTP: number; vPen: number; cPen: number } {
  const baseTP = computeBaseTrueProb(leg);
  const vPen = volatilityPenalty(leg);
  const cPen = contextPenalty(leg);
  const adjusted = baseTP * (1 - vPen) * (1 - cPen);
  return { baseTP, adjustedTP: clamp(adjusted, 0.0001, 0.9999), vPen, cPen };
}

// ─── Correlation penalty ──────────────────────────────────────────────────────

export function computeCorrelationPenalty(legs: EnrichedLeg[]): CorrelationInfo {
  const gameCounts: Record<string, number> = {};
  const teamCounts: Record<string, number> = {};
  let highThresholdCount = 0;

  for (const leg of legs) {
    if (leg.gameId) gameCounts[leg.gameId] = (gameCounts[leg.gameId] ?? 0) + 1;
    if (leg.team) teamCounts[leg.team] = (teamCounts[leg.team] ?? 0) + 1;
    const highPTS = leg.market === 'PTS' && leg.line >= 30;
    const highOther =
      (leg.market === 'AST' && leg.line >= 10) ||
      (leg.market === 'REB' && leg.line >= 12) ||
      (leg.market === '3PM' && leg.line >= 4);
    if (highPTS || highOther) highThresholdCount++;
  }

  let gameClusterTax = 0;
  for (const gid in gameCounts) {
    if (gameCounts[gid] >= 3) gameClusterTax += 0.08;
    else if (gameCounts[gid] === 2) gameClusterTax += 0.04;
  }

  let teamClusterTax = 0;
  for (const team in teamCounts) {
    if (teamCounts[team] > 2) teamClusterTax += 0.06;
  }

  const volatilityStackTax = highThresholdCount >= 2 ? 0.05 : 0;
  const totalTax = clamp(teamClusterTax + gameClusterTax + volatilityStackTax, 0, 0.5);

  return {
    teamClusterTax,
    gameClusterTax,
    volatilityStackTax,
    totalTax,
    correlationFactor: clamp(1 - totalTax, 0.5, 1),
  };
}

// ─── Monte Carlo simulation ───────────────────────────────────────────────────

// Box-Muller standard normal sample
function randNormal(): number {
  return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

// Cholesky-Banachiewicz for small n×n positive-definite matrices
function cholesky(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
      L[i][j] = i === j
        ? Math.sqrt(Math.max(0, A[i][i] - sum))
        : (A[i][j] - sum) / (L[j][j] || 1e-12);
    }
  }
  return L;
}

export function simulateParlay(legs: EnrichedLeg[], iterations?: number): number {
  if (!legs.length) return 0;
  const iters = iterations ?? ENGINE_CONFIG.monteCarloIterations;
  let wins = 0;
  for (let i = 0; i < iters; i++) {
    let hit = true;
    for (const leg of legs) {
      if (Math.random() > leg.adjustedTP) { hit = false; break; }
    }
    if (hit) wins++;
  }
  return wins / iters;
}

export function simulateCorrelatedParlay(legs: EnrichedLeg[], iterations?: number): number {
  if (!legs.length) return 0;
  const iters = iterations ?? ENGINE_CONFIG.monteCarloIterations;
  const n = legs.length;

  // Build correlation matrix: same gameId → 0.15, else 0
  const C: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 1;
      if (legs[i].gameId && legs[i].gameId === legs[j].gameId) return 0.15;
      return 0;
    })
  );
  const L = cholesky(C);

  let wins = 0;
  for (let iter = 0; iter < iters; iter++) {
    const z = Array.from({ length: n }, () => randNormal());
    // correlated sample = L @ z
    let hit = true;
    for (let i = 0; i < n; i++) {
      let corr = 0;
      for (let k = 0; k <= i; k++) corr += L[i][k] * z[k];
      const u = normalCDF(corr);
      if (u > legs[i].adjustedTP) { hit = false; break; }
    }
    if (hit) wins++;
  }
  return wins / iters;
}

// ─── Kelly criterion ──────────────────────────────────────────────────────────

export function kellyFraction(prob: number, decimalOdds: number): number {
  const b = decimalOdds - 1;
  if (b <= 0) return 0;
  const k = (prob * (b + 1) - 1) / b;
  return clamp(k, ENGINE_CONFIG.minKellyFraction, ENGINE_CONFIG.maxKellyFraction);
}

// ─── EV ───────────────────────────────────────────────────────────────────────

function expectedValue(prob: number, decimalOdds: number): number {
  return prob * (decimalOdds - 1) - (1 - prob);
}

// ─── Grading ─────────────────────────────────────────────────────────────────

export function gradeLeg(adjustedTP: number): 'A' | 'B' | 'C' | 'D' {
  if (adjustedTP >= 0.80) return 'A';
  if (adjustedTP >= 0.74) return 'B';
  if (adjustedTP >= 0.68) return 'C';
  return 'D';
}

export function isCeilingProp(leg: RawLeg): boolean {
  if (leg.market === 'PTS' && leg.line >= 30) return true;
  if (leg.market === 'AST' && leg.line >= 10) return true;
  if (leg.market === 'REB' && leg.line >= 12) return true;
  if (leg.market === '3PM' && leg.line >= 4) return true;
  return false;
}

export function legScore(leg: EnrichedLeg): number {
  const roleBoost = leg.roleStability ?? 1.0;
  const minutesBoost = leg.minutesSecurity ?? 1.0;
  const usageBoost = leg.usageStability ?? 1.0;
  const corrTax = leg.localCorrTax ?? 0;
  return (leg.adjustedTP * roleBoost * minutesBoost * usageBoost) - corrTax;
}

// ─── Evaluation ───────────────────────────────────────────────────────────────

export function evaluateParlay(parlay: RawLeg[]): Omit<ParlayEvaluation, 'simProb' | 'corrSimProb' | 'kelly'> {
  const enriched: EnrichedLeg[] = parlay.map((leg) => {
    const { baseTP, adjustedTP, vPen, cPen } = computeAdjustedProb(leg);
    return {
      ...leg,
      baseTP,
      adjustedTP,
      volatilityPenalty: vPen,
      contextPenalty: cPen,
      grade: gradeLeg(adjustedTP),
      isCeilingProp: isCeilingProp(leg),
      isVolatile: vPen >= 0.18,
      score: 0, // filled below
    };
  });

  for (const leg of enriched) leg.score = legScore(leg);

  const nLegs = enriched.length;
  const minATP = minATPForLegCount(nLegs);

  const eligible = enriched.filter((l) => l.adjustedTP >= minATP);
  const removed = enriched.filter((l) => l.adjustedTP < minATP);

  let rawProb = 1;
  let decimalOdds = 1;
  for (const leg of eligible) {
    rawProb *= leg.adjustedTP;
    decimalOdds *= americanToDecimal(leg.odds);
  }
  rawProb = clamp(rawProb, 1e-7, 0.999999);

  const corrInfo = computeCorrelationPenalty(eligible);
  const finalProb = clamp(rawProb * corrInfo.correlationFactor, 1e-7, 0.999999);
  const ev = expectedValue(finalProb, decimalOdds);

  return {
    legs: eligible,
    removedLegs: removed,
    legCount: nLegs,
    minATPRequired: minATP,
    rawProb,
    finalProb,
    decimalOdds,
    correlation: corrInfo,
    ev,
  };
}

export function runParlayAnalysis(parlay: RawLeg[], options?: { iterations?: number }): ParlayEvaluation {
  const base = evaluateParlay(parlay);
  const simProb = simulateParlay(base.legs, options?.iterations);
  const corrSimProb = simulateCorrelatedParlay(base.legs, options?.iterations);
  const kelly = kellyFraction(base.finalProb, base.decimalOdds);
  return { ...base, simProb, corrSimProb, kelly };
}
