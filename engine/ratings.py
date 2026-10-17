import logging
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

# Placeholder ratings: only these six NBA teams get model probabilities.
# Replace with a real, regularly updated data source.
BASELINE_TEAMS = pd.DataFrame({
    "team_id": ["Los Angeles Lakers", "Boston Celtics", "Golden State Warriors", "Denver Nuggets", "Miami Heat", "New York Knicks"],
    "base_elo": [1550, 1580, 1575, 1600, 1565, 1540],
    "offensive_efficiency": [115.2, 114.8, 118.1, 117.5, 113.0, 112.1],
    "defensive_efficiency": [112.5, 110.1, 115.3, 113.0, 109.5, 111.0],
    "last_5_games_win_pct": [0.6, 0.8, 0.4, 1.0, 0.8, 0.6],
    "avg_points_last_3_games": [118.0, 112.5, 125.0, 121.0, 110.0, 108.0],
    "is_on_road_trip": [1, 0, 0, 0, 1, 0],
    "days_since_last_game": [2, 3, 2, 4, 1, 3],
}).set_index("team_id")


class InformationAggregator:
    """Multi-stage internal ranking engine producing a proprietary score per team."""

    def __init__(self, baseline: pd.DataFrame = BASELINE_TEAMS):
        self.scores = self._run_full_ranking_cycle(baseline.copy())["final_proprietary_score"]

    def covers(self, team: str) -> bool:
        return team in self.scores.index

    def get_quant_features(self, team_a: str, team_b: str) -> dict | None:
        if not (self.covers(team_a) and self.covers(team_b)):
            return None
        return {"proprietary_score_difference": float(self.scores[team_a] - self.scores[team_b])}

    def _run_full_ranking_cycle(self, df: pd.DataFrame) -> pd.DataFrame:
        for stage in (
            self._strategy_recalculate_base_elo,
            self._strategy_factor_in_recent_form,
            self._strategy_adjust_for_offensive_momentum,
            self._strategy_apply_situational_modifiers,
            self._strategy_normalize_final_scores,
        ):
            df = stage(df)
        return df

    def _strategy_recalculate_base_elo(self, df):
        df["efficiency_diff"] = df["offensive_efficiency"] - df["defensive_efficiency"]
        df["intermediate_score"] = df["base_elo"] + df["efficiency_diff"] * 10
        return df

    def _strategy_factor_in_recent_form(self, df):
        df["intermediate_score"] += (df["last_5_games_win_pct"] - 0.5) * 100
        return df

    def _strategy_adjust_for_offensive_momentum(self, df):
        df["intermediate_score"] += (df["avg_points_last_3_games"] - df["offensive_efficiency"]) * 2
        return df

    def _strategy_apply_situational_modifiers(self, df):
        df.loc[df["is_on_road_trip"] == 1, "intermediate_score"] -= 25
        df.loc[df["days_since_last_game"] < 3, "intermediate_score"] -= 10
        return df

    def _strategy_normalize_final_scores(self, df):
        s = df["intermediate_score"]
        std = s.std()
        df["final_proprietary_score"] = 1500 + 100 * (s - s.mean()) / std if std > 0 else 1500.0
        return df


class MasterProbabilityModel:
    FEATURES = ["proprietary_score_difference"]

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=250, random_state=42)
        self.brier: float | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X[self.FEATURES], y)
        self.brier = float(-np.mean(cross_val_score(self.model, X[self.FEATURES], y, cv=3, scoring="neg_brier_score")))
        logger.info("MasterProbabilityModel trained. Brier loss: %.4f", self.brier)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.FEATURES])[:, 1]

    def certainty(self, X: pd.DataFrame) -> np.ndarray:
        """Per-row agreement across trees, scaled to [0, 1] (1 = trees agree most)."""
        values = X[self.FEATURES].to_numpy()
        preds = np.array([tree.predict_proba(values)[:, 1] for tree in self.model.estimators_])
        c = 1 / (np.std(preds, axis=0) + 1e-6)
        return (c - c.min()) / (c.max() - c.min() + 1e-6)


def placeholder_training_set(aggregator: InformationAggregator, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic outcomes drawn from Elo win expectancy. Stand-in until real game history is wired up."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for a, b in permutations(aggregator.scores.index, 2):
        diff = aggregator.get_quant_features(a, b)["proprietary_score_difference"]
        rows.append(diff)
        labels.append(int(rng.random() < 1 / (1 + 10 ** (-diff / 400))))
    return pd.DataFrame({"proprietary_score_difference": rows}), pd.Series(labels)
