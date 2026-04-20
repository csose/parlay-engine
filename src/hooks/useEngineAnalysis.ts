import { useState, useCallback, useTransition } from 'react';
import type { RawLeg, AnalysisState } from '../types/engine';
import { runParlayAnalysis } from '../engine/core';
import { buildAllTiers } from '../engine/tierBuilder';

export function useEngineAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    status: 'idle',
    allTiers: null,
    evaluation: null,
    error: null,
  });
  const [isPending, startTransition] = useTransition();

  const run = useCallback((legs: RawLeg[]) => {
    setState((s) => ({ ...s, status: 'running', error: null }));
    startTransition(() => {
      try {
        const evaluation = runParlayAnalysis(legs);
        const allTiers = buildAllTiers(legs);
        setState({ status: 'done', evaluation, allTiers, error: null });
      } catch (err) {
        setState({ status: 'error', evaluation: null, allTiers: null, error: String(err) });
      }
    });
  }, []);

  const reset = useCallback(() => {
    setState({ status: 'idle', allTiers: null, evaluation: null, error: null });
  }, []);

  return { state, isPending, run, reset };
}
