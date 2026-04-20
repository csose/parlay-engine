import { useState } from 'react';
import type { AppMode, RawLeg, ExtractionAnalysis } from './types/engine';
import { useEngineAnalysis } from './hooks/useEngineAnalysis';
import { useIngestion } from './hooks/useIngestion';
import { reconcileBetResults } from './engine/ingestion';
import { Header } from './components/Header';
import { ExtractionStatus } from './components/ExtractionStatus';
import { ErrorBanner } from './components/ErrorBanner';
import { PayoutSummary } from './components/PayoutSummary';
import { SummaryMetrics } from './components/SummaryMetrics';
import { TierMatrix } from './components/TierMatrix';
import { Findings } from './components/Findings';
import { Education } from './components/Education';
import { LegInputForm } from './components/LegInputForm';
import { FileUploadZone } from './components/FileUploadZone';
import './index.css';

function parsedLegsToRaw(extraction: ExtractionAnalysis): Partial<RawLeg>[] {
  return extraction.legs.map((l) => ({
    player: l.player,
    market: l.market,
    line: l.line,
    odds: -110,
  }));
}

export default function App() {
  const [mode, setMode] = useState<AppMode>('analyze');
  const [timestamp] = useState(() => new Date().toISOString());
  const [priorExtraction, setPriorExtraction] = useState<ExtractionAnalysis | null>(null);
  const [reconcileResult, setReconcileResult] = useState<ReturnType<typeof reconcileBetResults> | null>(null);

  const engine = useEngineAnalysis();
  const ingestion = useIngestion();
  const resultsIngestion = useIngestion();

  function handleLegsSubmit(legs: RawLeg[]) {
    engine.run(legs);
  }

  function handleParlayFile(file: File) {
    ingestion.processFile(file, 'Analyze Parlay');
  }

  function handleResultsFile(file: File) {
    resultsIngestion.processFile(file, 'Analyze Results').then(() => {
      if (priorExtraction && resultsIngestion.state.extraction) {
        const rec = reconcileBetResults(priorExtraction, resultsIngestion.state.extraction);
        setReconcileResult(rec);
      }
    });
  }

  const engineDone = engine.state.status === 'done';
  const ev = engine.state.evaluation;
  const allTiers = engine.state.allTiers;

  // Persist extraction so "Analyze Results" can reconcile against it
  if (ingestion.state.extraction && ingestion.state.extraction !== priorExtraction) {
    setPriorExtraction(ingestion.state.extraction);
  }

  return (
    <div className="pe-root">
      <Header timestamp={timestamp} mode={mode} />

      {/* Mode toggle */}
      <div className="pe-mode-toggle">
        <button
          className={`pe-btn pe-btn--tab ${mode === 'analyze' ? 'pe-btn--tab-active' : ''}`}
          onClick={() => setMode('analyze')}
        >
          Analyze Parlay
        </button>
        <button
          className={`pe-btn pe-btn--tab ${mode === 'results' ? 'pe-btn--tab-active' : ''}`}
          onClick={() => setMode('results')}
        >
          Analyze Results
        </button>
      </div>

      {/* ── ANALYZE PARLAY MODE ── */}
      {mode === 'analyze' && (
        <div className="pe-input-section">
          <div className="pe-input-cols">
            <div className="pe-input-col">
              <h2 className="pe-section-title">Manual Leg Entry</h2>
              <LegInputForm
                onSubmit={handleLegsSubmit}
                disabled={engine.state.status === 'running' || engine.isPending}
                initialLegs={ingestion.state.extraction ? parsedLegsToRaw(ingestion.state.extraction) : undefined}
              />
            </div>
            <div className="pe-input-col pe-input-col--upload">
              <h2 className="pe-section-title">Or Upload Parlay File</h2>
              <FileUploadZone
                mode="parlay"
                onFile={handleParlayFile}
                disabled={ingestion.state.loading}
              />
              {ingestion.state.error && (
                <p className="pe-error-text">{ingestion.state.error}</p>
              )}
            </div>
          </div>

          {ingestion.state.extraction && (
            <ExtractionStatus extraction={ingestion.state.extraction} />
          )}
        </div>
      )}

      {/* ── ANALYZE RESULTS MODE ── */}
      {mode === 'results' && (
        <div className="pe-input-section">
          <h2 className="pe-section-title">Upload Result Screenshot</h2>
          {!priorExtraction && (
            <p className="pe-warn-text">
              ⚠️ No prior parlay analysed yet. Run an analysis first so results can be reconciled.
            </p>
          )}
          <FileUploadZone
            mode="results"
            onFile={handleResultsFile}
            disabled={resultsIngestion.state.loading}
          />
          {resultsIngestion.state.extraction && (
            <ExtractionStatus extraction={resultsIngestion.state.extraction} />
          )}
          {reconcileResult && (
            <section className="pe-card">
              <h2 className="pe-section-title">Reconciliation</h2>
              <div className="pe-kv-grid">
                <span className="pe-kv__key">Bet ID</span>
                <span className="pe-kv__val pe-mono">{reconcileResult.betId ?? '—'}</span>
                <span className="pe-kv__key">Sportsbook</span>
                <span className="pe-kv__val">{reconcileResult.sportsbook ?? '—'}</span>
                <span className="pe-kv__key">Overall hit</span>
                <span className="pe-kv__val">{reconcileResult.actualHit ? '✅ WON' : '❌ LOST'}</span>
                <span className="pe-kv__key">Actual return</span>
                <span className="pe-kv__val">{reconcileResult.actualReturn !== null ? `$${reconcileResult.actualReturn}` : '—'}</span>
                <span className="pe-kv__key">Predicted prob.</span>
                <span className="pe-kv__val">
                  {reconcileResult.predictedFinalProb !== null
                    ? `${(reconcileResult.predictedFinalProb * 100).toFixed(2)}%`
                    : '—'}
                </span>
              </div>
              <table className="pe-table" style={{ marginTop: '1rem' }}>
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Market</th>
                    <th>Line</th>
                    <th>Hit?</th>
                    <th>Matched</th>
                  </tr>
                </thead>
                <tbody>
                  {reconcileResult.legs.map((leg, i) => (
                    <tr key={i}>
                      <td>{leg.player}</td>
                      <td>{leg.market}</td>
                      <td>{leg.line}+</td>
                      <td>{leg.actualHit === true ? '✅' : leg.actualHit === false ? '❌' : '—'}</td>
                      <td>{leg.matched ? '✅' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </div>
      )}

      {/* ── ENGINE OUTPUT ── */}
      {engine.state.status === 'running' && (
        <div className="pe-running-banner">⚙️ Running Monte Carlo simulation…</div>
      )}

      {engine.state.status === 'error' && (
        <div className="pe-error-banner">❌ Engine error: {engine.state.error}</div>
      )}

      {engineDone && ev && allTiers && (
        <>
          <ErrorBanner
            removedLegs={ev.removedLegs}
            warnings={ingestion.state.extraction?.warnings ?? []}
            errors={ingestion.state.extraction?.errors ?? []}
          />
          <PayoutSummary allTiers={allTiers} />
          <SummaryMetrics allTiers={allTiers} evaluation={ev} />
          <TierMatrix allTiers={allTiers} />
          <Findings evaluation={ev} />
          <Education />
        </>
      )}
    </div>
  );
}
