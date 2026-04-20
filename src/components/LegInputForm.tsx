import { useState } from 'react';
import type { RawLeg, Market } from '../types/engine';
import { isCompleteLeg } from '../types/engine';

interface Props {
  onSubmit: (legs: RawLeg[]) => void;
  disabled: boolean;
  initialLegs?: Partial<RawLeg>[];
}

const MARKETS: { value: Market; label: string }[] = [
  { value: 'PTS', label: 'Points' },
  { value: 'AST', label: 'Assists' },
  { value: 'REB', label: 'Rebounds' },
  { value: '3PM', label: 'Threes' },
];

function emptyLeg(): Partial<RawLeg> {
  return { player: '', market: 'PTS', line: undefined, odds: undefined, team: '', gameId: '' };
}

export function LegInputForm({ onSubmit, disabled, initialLegs }: Props) {
  const [rows, setRows] = useState<Partial<RawLeg>[]>(
    initialLegs?.length ? initialLegs : [emptyLeg(), emptyLeg()],
  );

  function updateRow(i: number, patch: Partial<RawLeg>) {
    setRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  }

  function addRow() {
    setRows((prev) => [...prev, emptyLeg()]);
  }

  function removeRow(i: number) {
    setRows((prev) => prev.filter((_, idx) => idx !== i));
  }

  function handleSubmit() {
    const complete = rows.filter(isCompleteLeg);
    if (complete.length >= 2) onSubmit(complete);
  }

  const completeCount = rows.filter(isCompleteLeg).length;
  const canSubmit = !disabled && completeCount >= 2;

  return (
    <div className="pe-leg-form">
      <div className="pe-leg-form__rows">
        {rows.map((row, i) => (
          <div key={i} className="pe-leg-row">
            <span className="pe-leg-row__num">{i + 1}</span>

            <input
              className="pe-input pe-input--player"
              placeholder="Player name"
              value={row.player ?? ''}
              onChange={(e) => updateRow(i, { player: e.target.value })}
              disabled={disabled}
            />

            <select
              className="pe-select"
              value={row.market ?? 'PTS'}
              onChange={(e) => updateRow(i, { market: e.target.value as Market })}
              disabled={disabled}
            >
              {MARKETS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>

            <input
              className="pe-input pe-input--line"
              type="number"
              min={0}
              step={0.5}
              placeholder="Line"
              value={row.line ?? ''}
              onChange={(e) => updateRow(i, { line: Number(e.target.value) })}
              disabled={disabled}
            />

            <input
              className="pe-input pe-input--odds"
              type="number"
              placeholder="Odds (e.g. -110)"
              value={row.odds ?? ''}
              onChange={(e) => updateRow(i, { odds: Number(e.target.value) })}
              disabled={disabled}
            />

            <input
              className="pe-input pe-input--mean"
              type="number"
              step={0.1}
              placeholder="Season avg (opt)"
              value={row.mean ?? ''}
              onChange={(e) => updateRow(i, { mean: e.target.value ? Number(e.target.value) : undefined })}
              disabled={disabled}
            />

            <input
              className="pe-input pe-input--team"
              placeholder="Team (opt)"
              value={row.team ?? ''}
              onChange={(e) => updateRow(i, { team: e.target.value })}
              disabled={disabled}
            />

            <input
              className="pe-input pe-input--game"
              placeholder="Game ID (opt)"
              value={row.gameId ?? ''}
              onChange={(e) => updateRow(i, { gameId: e.target.value })}
              disabled={disabled}
            />

            <button
              className="pe-btn pe-btn--ghost pe-btn--remove"
              onClick={() => removeRow(i)}
              disabled={disabled || rows.length <= 2}
              title="Remove leg"
            >
              ✕
            </button>
          </div>
        ))}
      </div>

      <div className="pe-leg-form__actions">
        <button className="pe-btn pe-btn--secondary" onClick={addRow} disabled={disabled}>
          + Add Leg
        </button>
        <span className="pe-leg-form__count">
          {completeCount} / {rows.length} legs complete
        </span>
        <button
          className="pe-btn pe-btn--primary"
          onClick={handleSubmit}
          disabled={!canSubmit}
        >
          {disabled ? 'Analysing…' : 'Analyse'}
        </button>
      </div>
    </div>
  );
}
