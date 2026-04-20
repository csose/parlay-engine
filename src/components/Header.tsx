import type { AppMode } from '../types/engine';

interface Props {
  timestamp: string;
  mode: AppMode;
}

export function Header({ timestamp, mode }: Props) {
  return (
    <header className="pe-header">
      <div className="pe-header__brand">
        <span className="pe-header__title">PARLAY ENGINE</span>
        <span className="pe-badge pe-badge--version">v7.3.0</span>
        <span className="pe-badge pe-badge--tag">PROBABILITY-FIRST</span>
        <span className={`pe-badge pe-badge--mode pe-badge--${mode}`}>
          {mode === 'analyze' ? 'ANALYZE PARLAY' : 'ANALYZE RESULTS'}
        </span>
      </div>
      <div className="pe-header__meta">
        <span className="pe-header__timestamp">{timestamp}</span>
      </div>
    </header>
  );
}
