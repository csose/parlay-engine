import { useState, useCallback } from 'react';
import type { IngestionState } from '../types/engine';
import { analyzeFile } from '../engine/ingestion';

export function useIngestion() {
  const [state, setState] = useState<IngestionState>({
    file: null,
    extraction: null,
    loading: false,
    error: null,
  });

  const processFile = useCallback(async (file: File, action = 'Analyze Parlay') => {
    setState({ file, extraction: null, loading: true, error: null });
    try {
      const extraction = await analyzeFile(file, action);
      setState({ file, extraction, loading: false, error: null });
    } catch (err) {
      setState({ file, extraction: null, loading: false, error: String(err) });
    }
  }, []);

  const reset = useCallback(() => {
    setState({ file: null, extraction: null, loading: false, error: null });
  }, []);

  return { state, processFile, reset };
}
