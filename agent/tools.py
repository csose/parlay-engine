import json
from typing import NotRequired, TypedDict

from anthropic import beta_tool

from engine import run_engine
from engine.odds import PROP_VIG_FACTOR, decimal_to_american, implied_prob
from engine.odds_api import fetch_odds
from engine.parlays import Bet, price_parlay


class Leg(TypedDict):
    description: str
    odds: int
    game: str
    estimated_prob: NotRequired[float]


def _tier(prob: float, ev: float) -> str:
    if ev <= 0:
        return "PASS"
    if prob >= 0.73:
        return "LOCK"
    if prob >= 0.60:
        return "STRONG"
    return "VALUE"


def _to_bet(leg: Leg) -> tuple[Bet, bool]:
    estimated = leg.get("estimated_prob")
    if estimated is not None and not 0 < estimated < 1:
        raise ValueError(f"estimated_prob for '{leg['description']}' must be between 0 and 1, got {estimated}.")
    prob = estimated if estimated is not None else implied_prob(leg["odds"]) * PROP_VIG_FACTOR
    return Bet(description=leg["description"], odds=leg["odds"], prob=prob, game_id=leg["game"], matchup=leg["game"]), estimated is not None


@beta_tool
def run_parlay_engine(bankroll: float, sports: list[str] | None = None, leg_counts: list[int] | None = None, max_parlays: int = 3) -> str:
    """Run the full quant parlay engine on live moneyline odds and return a mutually exclusive parlay portfolio.

    Fetches the best US moneyline per side, scores games with the proprietary ratings model, keeps +EV singles,
    prices 2- and 3-leg parlays with correlated Monte Carlo, and selects the highest-quality non-overlapping parlays.
    The result includes a caveat and model coverage counts that must be relayed to the user.

    Args:
        bankroll: Total bankroll in USD; stakes are quarter-Kelly, capped at 5% of bankroll per bet.
        sports: The Odds API sport keys, e.g. basketball_nba, icehockey_nhl. Defaults to NBA and NHL.
        leg_counts: Parlay sizes to consider. Defaults to [2, 3].
        max_parlays: Maximum number of parlays in the portfolio.
    """
    if bankroll <= 0:
        raise ValueError("bankroll must be positive.")
    result = run_engine(bankroll, sports=sports, leg_counts=tuple(leg_counts or (2, 3)), max_parlays=max_parlays)
    return json.dumps(result)


@beta_tool
def get_slate(sports: list[str] | None = None) -> str:
    """List upcoming games with the best available moneyline per side and the devigged market probability.

    Args:
        sports: The Odds API sport keys, e.g. basketball_nba, icehockey_nhl. Defaults to NBA and NHL.
    """
    df = fetch_odds(sports)
    if df.empty:
        return json.dumps({"games": []})
    cols = ["sport", "commence_time", "away_team", "home_team", "bet", "odds", "sportsbook", "market_prob"]
    return df[cols].round({"market_prob": 4}).to_json(orient="records")


@beta_tool
def evaluate_props(props: list[Leg], bankroll: float) -> str:
    """Price individual bets (e.g. player props read from sportsbook screenshots): EV, quarter-Kelly stake, tier.

    Without estimated_prob, a leg is priced at the market's no-vig probability, so its EV is negative by
    construction; supply estimated_prob only when you have a reasoned view, and say it is your estimate.

    Args:
        props: Bets to price. game is the matchup, e.g. "Lakers vs Celtics"; odds are American odds.
        bankroll: Total bankroll in USD.
    """
    out = []
    for leg in props:
        bet, estimated = _to_bet(leg)
        out.append({
            "description": bet.description,
            "game": leg["game"],
            "odds": leg["odds"],
            "implied_prob": round(implied_prob(leg["odds"]), 4),
            "no_vig_prob": round(implied_prob(leg["odds"]) * PROP_VIG_FACTOR, 4),
            "prob_used": round(bet.prob, 4),
            "prob_source": "estimate" if estimated else "market_no_vig",
            "ev_pct": round(bet.ev * 100, 2),
            "kelly_stake_usd": round(bankroll * bet.kelly, 2),
            "tier": _tier(bet.prob, bet.ev),
        })
    return json.dumps(out)


@beta_tool
def price_custom_parlay(legs: list[Leg], bankroll: float) -> str:
    """Price one specific parlay: joint win probability (legs in the same game are positively correlated), payout, EV, stake.

    Args:
        legs: Two or more legs. Legs with the same game string are treated as same-game legs.
        bankroll: Total bankroll in USD.
    """
    if len(legs) < 2:
        raise ValueError("A parlay needs at least two legs.")
    bets = [_to_bet(leg)[0] for leg in legs]
    parlay = price_parlay(bets)
    return json.dumps({
        "legs": [{"description": b.description, "game": b.matchup, "odds": int(b.odds), "prob_used": round(b.prob, 4)} for b in bets],
        "decimal_odds": round(parlay.decimal_odds, 3),
        "american_odds": round(decimal_to_american(parlay.decimal_odds)),
        "win_prob": round(parlay.prob, 4),
        "ev_pct": round(parlay.ev * 100, 2),
        "kelly_stake_usd": round(bankroll * parlay.kelly, 2),
    })


ALL_TOOLS = [run_parlay_engine, get_slate, evaluate_props, price_custom_parlay]
