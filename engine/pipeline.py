import logging

import numpy as np
import pandas as pd

from .odds_api import fetch_odds
from .parlays import MC_TRIALS, Bet, build_candidate_parlays, select_exclusive_portfolio
from .ratings import InformationAggregator, MasterProbabilityModel, placeholder_training_set

logger = logging.getLogger(__name__)

MODEL_CAVEAT = (
    "Model probabilities come from placeholder ratings for six NBA teams and synthetic training "
    "outcomes. Treat any edge as unvalidated until real ratings and game history are wired in."
)


def score_slate(odds: pd.DataFrame, aggregator: InformationAggregator, model: MasterProbabilityModel) -> pd.DataFrame:
    df = odds.copy()
    df["proprietary_score_difference"] = np.nan
    df["my_prob"] = np.nan
    df["certainty"] = np.nan
    for _, game in df.groupby("game_id"):
        row = game.iloc[0]
        features = aggregator.get_quant_features(row["home_team"], row["away_team"])
        if features:
            df.loc[game.index, "proprietary_score_difference"] = features["proprietary_score_difference"]

    covered = df["proprietary_score_difference"].notna()
    if covered.any():
        X = df.loc[covered, MasterProbabilityModel.FEATURES]
        home_win = pd.Series(model.predict_proba(X), index=X.index)
        is_home = df.loc[covered, "bet"] == df.loc[covered, "home_team"]
        df.loc[covered, "my_prob"] = home_win.where(is_home, 1 - home_win)
        df.loc[covered, "certainty"] = model.certainty(X)
    return df


def positive_ev_bets(scored: pd.DataFrame) -> list[Bet]:
    bets = []
    for _, r in scored[scored["my_prob"].notna()].iterrows():
        bet = Bet(
            description=r["bet"],
            odds=float(r["odds"]),
            prob=float(r["my_prob"]),
            game_id=r["game_id"],
            sport=r["sport"],
            matchup=f"{r['away_team']} @ {r['home_team']}",
            certainty=float(r["certainty"]),
        )
        if bet.ev > 0:
            bets.append(bet)
    return sorted(bets, key=lambda b: b.ev, reverse=True)


def run_engine(
    bankroll: float,
    sports: list[str] | None = None,
    leg_counts: tuple[int, ...] = (2, 3),
    max_parlays: int = 3,
    leg_pool: int = 15,
    trials: int = MC_TRIALS,
    odds: pd.DataFrame | None = None,
) -> dict:
    """Full pipeline: odds -> proprietary ratings -> +EV singles -> correlated parlays -> exclusive portfolio."""
    aggregator = InformationAggregator()
    model = MasterProbabilityModel()
    model.train(*placeholder_training_set(aggregator))

    if odds is None:
        odds = fetch_odds(sports)
    report = {"bankroll": bankroll, "caveat": MODEL_CAVEAT, "model_brier": model.brier}
    if odds.empty:
        return {**report, "games": 0, "status": "no_odds", "portfolio": []}

    scored = score_slate(odds, aggregator, model)
    singles = positive_ev_bets(scored)[:leg_pool]
    portfolio = select_exclusive_portfolio(build_candidate_parlays(singles, list(leg_counts), trials), max_parlays)

    return {
        **report,
        "games": int(scored["game_id"].nunique()),
        "games_with_model_coverage": int(scored.loc[scored["my_prob"].notna(), "game_id"].nunique()),
        "positive_ev_singles": [_bet_dict(b, bankroll) for b in singles],
        "status": "ok" if portfolio else "no_positive_ev_parlays",
        "portfolio": [
            {
                "legs": [_bet_dict(leg, bankroll) for leg in p.legs],
                "decimal_odds": round(p.decimal_odds, 3),
                "win_prob": round(p.prob, 4),
                "ev_pct": round(p.ev * 100, 2),
                "kelly_pct": round(p.kelly * 100, 3),
                "stake_usd": round(bankroll * p.kelly, 2),
                **{k: round(v, 4) for k, v in p.quality.items()},
            }
            for p in portfolio
        ],
    }


def _bet_dict(b: Bet, bankroll: float) -> dict:
    return {
        "bet": b.description,
        "matchup": b.matchup,
        "sport": b.sport,
        "odds": int(b.odds),
        "model_prob": round(b.prob, 4),
        "ev_pct": round(b.ev * 100, 2),
        "kelly_stake_usd": round(bankroll * b.kelly, 2),
    }
