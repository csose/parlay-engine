import type {
  ExtractionAnalysis,
  ParsedLeg,
  Market,
  AnalysisError,
  ReconcileResult,
  ReconciledLeg,
} from '../types/engine';
import { americanToDecimal } from './core';

// ─── Factories ────────────────────────────────────────────────────────────────

export function createEmptyAnalysis(): ExtractionAnalysis {
  return {
    titleDate: '',
    sourceType: '',
    action: '',
    success: false,
    sportsbook: null,
    sport: 'NBA',
    betType: null,
    betId: null,
    placedAt: null,
    settledAt: null,
    stake: null,
    totalOddsAmerican: null,
    totalOddsDecimal: null,
    projectedReturn: null,
    actualReturn: null,
    tierLabel: null,
    legCount: 0,
    legs: [],
    confidence: { document: 0, fields: {} },
    warnings: [],
    errors: [],
  };
}

export function makeTitleDate(prefix = 'PARLAY-ENGINE'): string {
  return `${prefix} — ${new Date().toISOString().slice(0, 10)}`;
}

// ─── File detection ───────────────────────────────────────────────────────────

export function detectFileType(file: File): 'image' | 'pdf' | 'text' | 'unknown' {
  const name = file.name.toLowerCase();
  if (file.type.startsWith('image/') || /\.(png|jpg|jpeg|webp|gif)$/.test(name)) return 'image';
  if (file.type === 'application/pdf' || name.endsWith('.pdf')) return 'pdf';
  if (file.type.startsWith('text/') || name.endsWith('.txt')) return 'text';
  return 'unknown';
}

// ─── Text normalisation ───────────────────────────────────────────────────────

export function normalizeText(raw: string): string {
  return (raw ?? '')
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

// ─── Field parsers ────────────────────────────────────────────────────────────

export function parseAmericanOdds(text: string): number | null {
  const m = text.match(/([+-]\d{3,6})/);
  return m ? Number(m[1]) : null;
}

export function parseDollarAmount(text: string): number | null {
  const m = text.match(/\$\s?(\d+(?:\.\d+)?)/);
  return m ? Number(m[1]) : null;
}

export function parseBetId(text: string): string | null {
  const m = text.match(/BET\s*ID[:\s]+([A-Z0-9/.\-]+)/i);
  return m ? m[1] : null;
}

export function detectSportsbook(text: string): string | null {
  const t = text.toLowerCase();
  if (t.includes('fanduel')) return 'FanDuel';
  if (t.includes('draftkings')) return 'DraftKings';
  if (t.includes('betmgm')) return 'BetMGM';
  if (t.includes('caesars')) return 'Caesars';
  if (t.includes('pointsbet')) return 'PointsBet';
  if (t.includes('bet365')) return 'Bet365';
  return null;
}

export function detectBetType(text: string): string | null {
  const t = text.toLowerCase();
  if (t.includes('same game parlay+') || t.includes('sgp+')) return 'SGP+';
  if (t.includes('same game parlay') || t.includes('sgp')) return 'SGP';
  if (t.includes('parlay')) return 'Parlay';
  return null;
}

export function normalizeMarket(raw: string): Market {
  const r = raw.toLowerCase();
  if (r.includes('point') || r === 'pts') return 'PTS';
  if (r.includes('assist') || r === 'ast') return 'AST';
  if (r.includes('rebound') || r === 'reb') return 'REB';
  if (r.includes('three') || r.includes('3-point') || r === '3pm') return '3PM';
  return 'UNKNOWN';
}

// ─── Leg parser ───────────────────────────────────────────────────────────────

export function parseLegsFromText(text: string): ParsedLeg[] {
  const legs: ParsedLeg[] = [];
  const normalized = text.replace(/\n/g, ' ');

  // Pattern: "Player Name TO SCORE/TO RECORD/OVER N+ POINTS/ASSISTS/REBOUNDS/THREES"
  const pattern =
    /([A-Z][a-z]+(?:\s+[A-Z][a-z.'`-]+)+)\s+(?:TO SCORE|TO RECORD|OVER)\s+(\d+(?:\.\d+)?)\+?\s+(POINTS|ASSISTS|REBOUNDS|THREES|THREE-POINTERS)/gi;

  let match: RegExpExecArray | null;
  while ((match = pattern.exec(normalized)) !== null) {
    legs.push({
      player: match[1].trim(),
      market: normalizeMarket(match[3].trim()),
      line: Number(match[2]),
      direction: 'OVER',
      rawText: match[0],
    });
  }

  return legs;
}

export function inferResultMarkers(text: string, legs: ParsedLeg[]): ParsedLeg[] {
  const t = text.toLowerCase();
  return legs.map((leg) => {
    let actualHit: boolean | null = null;
    if (t.includes(leg.player.toLowerCase())) {
      if (t.includes('won') || t.includes('winner')) actualHit = true;
      if (t.includes('lost') || t.includes('loser')) actualHit = false;
    }
    return { ...leg, actualHit };
  });
}

// ─── Error/warning helpers ────────────────────────────────────────────────────

function pushError(a: ExtractionAnalysis, code: string, message: string, field: string | null = null, confidence: number | null = null): void {
  a.errors.push({ code, message, field, confidence } as AnalysisError);
}

function pushWarning(a: ExtractionAnalysis, code: string, message: string, field: string | null = null, confidence: number | null = null): void {
  a.warnings.push({ code, message, field, confidence } as AnalysisError);
}

// ─── Main analysis ────────────────────────────────────────────────────────────

export function analyzeExtractedText(
  rawExtract: { text: string; confidence: number; warnings?: string[] },
  action = 'Analyze Parlay',
  sourceType = 'unknown',
): ExtractionAnalysis {
  const analysis = createEmptyAnalysis();
  analysis.titleDate = makeTitleDate(action === 'Analyze Results' ? 'ANALYZE-RESULTS' : 'ANALYZE-PARLAY');
  analysis.sourceType = sourceType;
  analysis.action = action;
  analysis.confidence.document = rawExtract.confidence ?? 0;

  const text = normalizeText(rawExtract.text ?? '');
  if (!text) {
    pushError(analysis, 'EMPTY_TEXT', 'No readable text extracted from file', 'document', analysis.confidence.document);
    return analysis;
  }

  analysis.sportsbook = detectSportsbook(text);
  analysis.betType = detectBetType(text);
  analysis.betId = parseBetId(text);
  analysis.totalOddsAmerican = parseAmericanOdds(text);
  analysis.totalOddsDecimal = analysis.totalOddsAmerican !== null
    ? americanToDecimal(analysis.totalOddsAmerican)
    : null;
  analysis.stake = parseDollarAmount(text);

  const legs = parseLegsFromText(text);
  analysis.legs = action === 'Analyze Results' ? inferResultMarkers(text, legs) : legs;
  analysis.legCount = analysis.legs.length;

  const docConf = analysis.confidence.document;
  analysis.confidence.fields = {
    sportsbook: analysis.sportsbook !== null ? docConf : 0.5,
    betType: analysis.betType !== null ? docConf : 0.5,
    betId: analysis.betId !== null ? docConf : 0.4,
    totalOddsAmerican: analysis.totalOddsAmerican !== null ? docConf : 0.4,
    stake: analysis.stake !== null ? docConf : 0.4,
    legs: analysis.legs.length > 0 ? docConf : 0.3,
  };

  if (!analysis.sportsbook) pushWarning(analysis, 'UNKNOWN_SPORTSBOOK', 'Could not detect sportsbook', 'sportsbook', 0.5);
  if (!analysis.betType) pushWarning(analysis, 'UNKNOWN_BET_TYPE', 'Could not detect bet type', 'betType', 0.5);
  if (!analysis.legs.length) pushError(analysis, 'NO_LEGS_PARSED', 'Could not parse any bet legs', 'legs', 0.3);

  for (const w of rawExtract.warnings ?? []) {
    pushWarning(analysis, 'EXTRACTOR_WARNING', w);
  }

  analysis.success = analysis.errors.length === 0;
  return analysis;
}

export async function analyzeFile(file: File, action = 'Analyze Parlay'): Promise<ExtractionAnalysis> {
  const sourceType = detectFileType(file);

  if (sourceType === 'image' || sourceType === 'pdf') {
    // Image/PDF OCR is handled server-side (Supabase edge function).
    // Return a stub with a warning so the UI can prompt the user.
    const stub = createEmptyAnalysis();
    stub.titleDate = makeTitleDate(action === 'Analyze Results' ? 'ANALYZE-RESULTS' : 'ANALYZE-PARLAY');
    stub.sourceType = sourceType;
    stub.action = action;
    stub.success = false;
    pushWarning(stub, 'OCR_NOT_CONNECTED', `${sourceType.toUpperCase()} files require the OCR edge function. Paste text instead.`, 'document', 0);
    return stub;
  }

  const text = await file.text();
  return analyzeExtractedText({ text, confidence: 0.95 }, action, sourceType);
}

// ─── Reconciliation ───────────────────────────────────────────────────────────

export function fingerprintLeg(leg: ParsedLeg): string {
  return [leg.player ?? '', leg.market ?? '', leg.line ?? '', leg.direction ?? ''].join('|').toUpperCase();
}

export function fingerprintBet(analysis: ExtractionAnalysis): string {
  const legPrint = (analysis.legs ?? []).map(fingerprintLeg).sort().join('||');
  return [
    analysis.sportsbook ?? '',
    analysis.betType ?? '',
    analysis.stake ?? '',
    analysis.totalOddsAmerican ?? '',
    legPrint,
  ].join('###').toUpperCase();
}

export function matchResultToOriginal(
  resultAnalysis: ExtractionAnalysis,
  originalAnalyses: ExtractionAnalysis[],
): { matched: ExtractionAnalysis | null; matchConfidence: number; method: string } {
  if (resultAnalysis.betId) {
    const exact = originalAnalyses.find((x) => x.betId && x.betId === resultAnalysis.betId);
    if (exact) return { matched: exact, matchConfidence: 0.99, method: 'betId' };
  }

  const resultFinger = fingerprintBet(resultAnalysis);
  let best: ExtractionAnalysis | null = null;
  let bestScore = 0;

  for (const candidate of originalAnalyses) {
    let score = 0;
    if (candidate.sportsbook === resultAnalysis.sportsbook) score += 0.15;
    if (candidate.betType === resultAnalysis.betType) score += 0.15;
    if (candidate.stake === resultAnalysis.stake) score += 0.10;
    if (candidate.totalOddsAmerican === resultAnalysis.totalOddsAmerican) score += 0.15;

    const resultLegs = new Set((resultAnalysis.legs ?? []).map(fingerprintLeg));
    const candLegs = new Set((candidate.legs ?? []).map(fingerprintLeg));
    let overlap = 0;
    for (const leg of resultLegs) { if (candLegs.has(leg)) overlap++; }
    score += candidate.legs.length ? (overlap / candidate.legs.length) * 0.45 : 0;
    if (fingerprintBet(candidate) === resultFinger) score = 0.98;

    if (score > bestScore) { bestScore = score; best = candidate; }
  }

  if (!best || bestScore < 0.75) return { matched: null, matchConfidence: bestScore, method: 'none' };
  return { matched: best, matchConfidence: bestScore, method: 'fuzzy' };
}

export function reconcileBetResults(
  originalAnalysis: ExtractionAnalysis,
  resultAnalysis: ExtractionAnalysis,
): ReconcileResult {
  const originalMap = new Map<string, ParsedLeg>(
    (originalAnalysis.legs ?? []).map((leg) => [fingerprintLeg(leg), leg]),
  );

  const reconciledLegs: ReconciledLeg[] = (resultAnalysis.legs ?? []).map((resultLeg) => {
    const fp = fingerprintLeg(resultLeg);
    const orig = originalMap.get(fp);
    return {
      player: resultLeg.player,
      market: resultLeg.market,
      line: resultLeg.line,
      direction: resultLeg.direction,
      predictedProb: orig?.predictedProb ?? null,
      adjustedTP: orig?.adjustedTP ?? null,
      actualHit: resultLeg.actualHit ?? null,
      matched: !!orig,
    };
  });

  return {
    titleDate: makeTitleDate('RECONCILED-BET'),
    betId: originalAnalysis.betId ?? resultAnalysis.betId ?? null,
    sportsbook: originalAnalysis.sportsbook ?? resultAnalysis.sportsbook,
    predictedFinalProb: originalAnalysis.predictedFinalProb ?? null,
    actualReturn: resultAnalysis.actualReturn ?? null,
    actualHit: reconciledLegs.every((l) => l.actualHit === true),
    legs: reconciledLegs,
  };
}
