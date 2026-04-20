export type TierLevel = 'T1' | 'T2' | 'T3' | 'T4' | 'T5';
export type LegGrade = 'A' | 'B' | 'C' | 'D';
export type Market = 'PTS' | 'AST' | 'REB' | '3PM' | 'UNKNOWN';
export type AppMode = 'analyze' | 'results';

export interface EngineConfig {
  defaultStdDevPoints: number;
  monteCarloIterations: number;
  maxKellyFraction: number;
  minKellyFraction: number;
}

export interface TierConfig {
  label: string;
  minLegs: number;
  maxLegs: number;
  minATP: number;
  maxVolatileLegs: number;
  maxSameGameLegs: number;
  maxSameTeamLegs: number;
  allowCeilingProps: boolean;
  preferredGrades: LegGrade[];
}

export interface RawLeg {
  player: string;
  team?: string;
  opponent?: string;
  market: Market;
  line: number;
  odds: number;
  gameId?: string;
  // context flags
  isRookieStarter?: boolean;
  isBenchOrRotationRisk?: boolean;
  minutesCapRisk?: boolean;
  injuryReturn?: boolean;
  isQuestionable?: boolean;
  backToBack?: boolean;
  highFoulRiskBig?: boolean;
  isUnderdog?: boolean;
  blowoutRisk?: boolean;
  usageSplitRisk?: boolean;
  // optional overrides
  baseProb?: number;
  mean?: number;
  stdDev?: number;
  roleStability?: number;
  minutesSecurity?: number;
  usageStability?: number;
  localCorrTax?: number;
}

export interface EnrichedLeg extends RawLeg {
  baseTP: number;
  adjustedTP: number;
  volatilityPenalty: number;
  contextPenalty: number;
  grade: LegGrade;
  score: number;
  isCeilingProp: boolean;
  isVolatile: boolean;
}

export interface CorrelationInfo {
  teamClusterTax: number;
  gameClusterTax: number;
  volatilityStackTax: number;
  totalTax: number;
  correlationFactor: number;
}

export interface RejectedLeg {
  leg: EnrichedLeg;
  reason: string;
}

export interface ParlayEvaluation {
  legs: EnrichedLeg[];
  removedLegs: EnrichedLeg[];
  legCount: number;
  minATPRequired: number;
  rawProb: number;
  finalProb: number;
  decimalOdds: number;
  correlation: CorrelationInfo;
  ev: number;
  simProb: number;
  corrSimProb: number;
  kelly: number;
}

export interface TierResult {
  tier: TierLevel;
  label: string;
  valid: boolean;
  legs: EnrichedLeg[];
  rejected: RejectedLeg[];
  evaluation: ParlayEvaluation | null;
}

export interface AllTiersResult {
  generatedAt: string;
  titleDate: string;
  poolSize: number;
  tiers: Record<TierLevel, TierResult>;
}

// Ingestion types

export interface ConfidenceFields {
  sportsbook: number;
  betType: number;
  betId: number;
  totalOddsAmerican: number;
  stake: number;
  legs: number;
}

export interface AnalysisError {
  code: string;
  message: string;
  field: string | null;
  confidence: number | null;
}

export interface ParsedLeg {
  player: string;
  market: Market;
  line: number;
  direction: 'OVER' | 'UNDER';
  rawText: string;
  actualHit?: boolean | null;
  predictedProb?: number | null;
  adjustedTP?: number | null;
  matched?: boolean;
}

export interface ExtractionAnalysis {
  titleDate: string;
  sourceType: string;
  action: string;
  success: boolean;
  sportsbook: string | null;
  sport: string;
  betType: string | null;
  betId: string | null;
  placedAt: string | null;
  settledAt: string | null;
  stake: number | null;
  totalOddsAmerican: number | null;
  totalOddsDecimal: number | null;
  projectedReturn: number | null;
  actualReturn: number | null;
  tierLabel: string | null;
  legCount: number;
  legs: ParsedLeg[];
  confidence: {
    document: number;
    fields: Partial<ConfidenceFields>;
  };
  warnings: AnalysisError[];
  errors: AnalysisError[];
  predictedFinalProb?: number | null;
}

export interface ReconciledLeg {
  player: string;
  market: Market;
  line: number;
  direction: 'OVER' | 'UNDER';
  predictedProb: number | null;
  adjustedTP: number | null;
  actualHit: boolean | null;
  matched: boolean;
}

export interface ReconcileResult {
  titleDate: string;
  betId: string | null;
  sportsbook: string | null;
  predictedFinalProb: number | null;
  actualReturn: number | null;
  actualHit: boolean;
  legs: ReconciledLeg[];
}

export interface AnalysisState {
  status: 'idle' | 'running' | 'done' | 'error';
  allTiers: AllTiersResult | null;
  evaluation: ParlayEvaluation | null;
  error: string | null;
}

export interface IngestionState {
  file: File | null;
  extraction: ExtractionAnalysis | null;
  loading: boolean;
  error: string | null;
}

export function isCompleteLeg(leg: Partial<RawLeg>): leg is RawLeg {
  return (
    typeof leg.player === 'string' && leg.player.trim().length > 0 &&
    typeof leg.market === 'string' && leg.market !== 'UNKNOWN' &&
    typeof leg.line === 'number' && !isNaN(leg.line) &&
    typeof leg.odds === 'number' && !isNaN(leg.odds)
  );
}
