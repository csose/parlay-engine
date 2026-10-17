from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.stats import norm

from .odds import american_to_decimal, kelly

MC_TRIALS = 100_000
RHO_SAME_GAME = 0.15
RHO_CROSS_GAME = 0.00


@dataclass
class Bet:
    description: str
    odds: float
    prob: float
    game_id: str = ""
    sport: str = ""
    matchup: str = ""
    certainty: float = 1.0

    @property
    def decimal(self) -> float:
        return american_to_decimal(self.odds)

    @property
    def ev(self) -> float:
        return self.prob * self.decimal - 1

    @property
    def kelly(self) -> float:
        return kelly(self.prob, self.decimal)

    @property
    def key(self) -> tuple[str, str]:
        return (self.game_id, self.description)


@dataclass
class Parlay:
    legs: list[Bet]
    decimal_odds: float
    prob: float
    quality: dict = field(default_factory=dict)

    @property
    def ev(self) -> float:
        return self.prob * self.decimal_odds - 1

    @property
    def kelly(self) -> float:
        return kelly(self.prob, self.decimal_odds)


def corr_matrix(legs: list[Bet], rho_same_game: float = RHO_SAME_GAME, rho_cross_game: float = RHO_CROSS_GAME) -> np.ndarray:
    n = len(legs)
    rho = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            same = legs[i].game_id and legs[i].game_id == legs[j].game_id
            rho[i, j] = rho[j, i] = rho_same_game if same else rho_cross_game
    return rho


def joint_probability(probs: np.ndarray, rho: np.ndarray, trials: int = MC_TRIALS, seed: int | None = 0) -> float:
    """Probability every leg hits, via a Gaussian copula Monte Carlo."""
    try:
        L = np.linalg.cholesky(rho)
    except np.linalg.LinAlgError:
        L = np.eye(len(probs))
    z = np.random.default_rng(seed).standard_normal((trials, len(probs))) @ L.T
    return float((norm.cdf(z) < probs).all(axis=1).mean())


def price_parlay(legs: list[Bet], trials: int = MC_TRIALS) -> Parlay:
    probs = np.array([leg.prob for leg in legs])
    return Parlay(
        legs=list(legs),
        decimal_odds=float(np.prod([leg.decimal for leg in legs])),
        prob=joint_probability(probs, corr_matrix(legs), trials),
    )


def quality_score(parlay: Parlay) -> dict:
    W_VALUE, W_CONVICTION, W_CERTAINTY = 0.40, 0.30, 0.30
    value = 1 / (1 - parlay.ev) if parlay.ev < 1 else 1.0
    certainties = np.array([leg.certainty for leg in parlay.legs])
    certainty = float(np.prod(certainties) ** (1 / len(certainties))) if len(certainties) else 0.0
    total = W_VALUE * value + W_CONVICTION * parlay.prob + W_CERTAINTY * certainty
    return {"quality_score": total, "value_score": value, "conviction_score": parlay.prob, "certainty_score": certainty}


def build_candidate_parlays(bets: list[Bet], leg_counts: list[int], trials: int = MC_TRIALS) -> list[Parlay]:
    parlays = []
    for n_legs in leg_counts:
        for combo in combinations(bets, n_legs):
            # Both sides of one moneyline can't both win.
            if len({leg.game_id for leg in combo}) < n_legs:
                continue
            parlay = price_parlay(list(combo), trials)
            if parlay.ev > 0:
                parlay.quality = quality_score(parlay)
                parlays.append(parlay)
    return sorted(parlays, key=lambda p: p.quality["quality_score"], reverse=True)


def select_exclusive_portfolio(parlays: list[Parlay], max_size: int) -> list[Parlay]:
    portfolio, used = [], set()
    for parlay in parlays:
        if len(portfolio) >= max_size:
            break
        keys = {leg.key for leg in parlay.legs}
        if used.isdisjoint(keys):
            portfolio.append(parlay)
            used |= keys
    return portfolio
