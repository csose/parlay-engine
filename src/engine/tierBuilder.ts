import type { RawLeg, EnrichedLeg, TierConfig, TierLevel, TierResult, AllTiersResult, RejectedLeg } from '../types/engine';
import {
  computeAdjustedProb,
  gradeLeg,
  isCeilingProp,
  legScore,
  runParlayAnalysis,
} from './core';

export const TIER_CONFIG: Record<TierLevel, TierConfig> = {
  T1: { label: 'Safe Entry',  minLegs: 4,  maxLegs: 6,  minATP: 0.74, maxVolatileLegs: 1, maxSameGameLegs: 2, maxSameTeamLegs: 2, allowCeilingProps: false, preferredGrades: ['A'] },
  T2: { label: 'Balanced',    minLegs: 5,  maxLegs: 7,  minATP: 0.72, maxVolatileLegs: 1, maxSameGameLegs: 2, maxSameTeamLegs: 2, allowCeilingProps: false, preferredGrades: ['A', 'B'] },
  T3: { label: 'Aggressive',  minLegs: 7,  maxLegs: 9,  minATP: 0.70, maxVolatileLegs: 2, maxSameGameLegs: 2, maxSameTeamLegs: 2, allowCeilingProps: false, preferredGrades: ['A', 'B'] },
  T4: { label: 'High Risk',   minLegs: 9,  maxLegs: 12, minATP: 0.76, maxVolatileLegs: 2, maxSameGameLegs: 3, maxSameTeamLegs: 2, allowCeilingProps: false, preferredGrades: ['A', 'B'] },
  T5: { label: 'Moonshot',    minLegs: 12, maxLegs: 16, minATP: 0.80, maxVolatileLegs: 1, maxSameGameLegs: 2, maxSameTeamLegs: 2, allowCeilingProps: false, preferredGrades: ['A'] },
};

function countBy<T>(arr: T[], keyFn: (item: T) => string): Record<string, number> {
  const map: Record<string, number> = {};
  for (const item of arr) {
    const key = keyFn(item);
    map[key] = (map[key] ?? 0) + 1;
  }
  return map;
}

export function prepareCandidatePool(rawLegs: RawLeg[]): EnrichedLeg[] {
  return rawLegs.map((leg) => {
    const { baseTP, adjustedTP, vPen, cPen } = computeAdjustedProb(leg);
    const enriched: EnrichedLeg = {
      ...leg,
      baseTP,
      adjustedTP,
      volatilityPenalty: vPen,
      contextPenalty: cPen,
      grade: gradeLeg(adjustedTP),
      isCeilingProp: isCeilingProp(leg),
      isVolatile: vPen >= 0.18,
      score: 0,
    };
    enriched.score = legScore(enriched);
    return enriched;
  });
}

export function filterPoolForTier(pool: EnrichedLeg[], config: TierConfig): EnrichedLeg[] {
  return pool
    .filter((leg) => {
      if (leg.adjustedTP < config.minATP) return false;
      if (!config.preferredGrades.includes(leg.grade)) return false;
      if (!config.allowCeilingProps && leg.isCeilingProp) return false;
      return true;
    })
    .sort((a, b) => b.score - a.score);
}

export function validateLegAddition(
  current: EnrichedLeg[],
  candidate: EnrichedLeg,
  config: TierConfig,
): { ok: boolean; reason?: string } {
  if (current.some((l) => l.player === candidate.player)) {
    return { ok: false, reason: 'Duplicate player' };
  }

  const next = [...current, candidate];
  const sameGame = countBy(next, (l) => l.gameId ?? 'UNKNOWN');
  const sameTeam = countBy(next, (l) => l.team ?? 'UNKNOWN');

  for (const gid in sameGame) {
    if (sameGame[gid] > config.maxSameGameLegs) return { ok: false, reason: 'Same-game cap exceeded' };
  }
  for (const team in sameTeam) {
    if (sameTeam[team] > config.maxSameTeamLegs) return { ok: false, reason: 'Same-team cap exceeded' };
  }

  const volatileCount = next.filter((l) => l.isVolatile).length;
  if (volatileCount > config.maxVolatileLegs) return { ok: false, reason: 'Too many volatile legs' };

  return { ok: true };
}

export function buildTierFromPool(pool: EnrichedLeg[], tierName: TierLevel, config: TierConfig): TierResult {
  const filtered = filterPoolForTier(pool, config);
  const selected: EnrichedLeg[] = [];
  const rejected: RejectedLeg[] = [];

  for (const leg of filtered) {
    if (selected.length >= config.maxLegs) break;
    const v = validateLegAddition(selected, leg, config);
    if (v.ok) selected.push(leg);
    else rejected.push({ leg, reason: v.reason ?? 'Unknown' });
  }

  const valid = selected.length >= config.minLegs && selected.length <= config.maxLegs;
  const evaluation = valid ? runParlayAnalysis(selected) : null;

  return { tier: tierName, label: config.label, valid, legs: selected, rejected, evaluation };
}

export function buildAllTiers(rawLegs: RawLeg[]): AllTiersResult {
  const pool = prepareCandidatePool(rawLegs);
  return {
    generatedAt: new Date().toISOString(),
    titleDate: `PARLAY-ENGINE-v7.3 — ${new Date().toISOString().slice(0, 10)}`,
    poolSize: pool.length,
    tiers: {
      T1: buildTierFromPool(pool, 'T1', TIER_CONFIG.T1),
      T2: buildTierFromPool(pool, 'T2', TIER_CONFIG.T2),
      T3: buildTierFromPool(pool, 'T3', TIER_CONFIG.T3),
      T4: buildTierFromPool(pool, 'T4', TIER_CONFIG.T4),
      T5: buildTierFromPool(pool, 'T5', TIER_CONFIG.T5),
    },
  };
}
