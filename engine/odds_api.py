import logging
import os

import pandas as pd
import requests

from .odds import devig_two_way, implied_prob, PROP_VIG_FACTOR

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4/sports"
DEFAULT_SPORTS = ["basketball_nba", "icehockey_nhl"]


class OddsAPIError(RuntimeError):
    pass


def fetch_odds(sports: list[str] | None = None) -> pd.DataFrame:
    """Best available moneyline per (game, side) across US books, with devigged market probability."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise OddsAPIError("ODDS_API_KEY is not set.")

    records = []
    for sport in sports or DEFAULT_SPORTS:
        params = {"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"}
        try:
            r = requests.get(f"{BASE_URL}/{sport}/odds", params=params, timeout=10)
            r.raise_for_status()
            for game in r.json():
                for book in game["bookmakers"]:
                    for market in book["markets"]:
                        for outcome in market["outcomes"]:
                            records.append({
                                "sport": sport,
                                "game_id": game["id"],
                                "commence_time": game.get("commence_time", ""),
                                "home_team": game["home_team"],
                                "away_team": game["away_team"],
                                "sportsbook": book["title"],
                                "bet": outcome["name"],
                                "odds": outcome["price"],
                            })
        except requests.RequestException as exc:
            logger.warning("Network error fetching %s: %s", sport, exc)
        except (KeyError, ValueError) as exc:
            logger.warning("Unexpected API response format for %s: %s", sport, exc)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.sort_values("odds", ascending=False).drop_duplicates(subset=["game_id", "bet"]).reset_index(drop=True)
    df["market_prob"] = df["odds"].apply(implied_prob)
    return apply_devig(df)


def apply_devig(df: pd.DataFrame) -> pd.DataFrame:
    devigged = df["market_prob"].copy()
    for _, group in df.groupby("game_id"):
        if len(group) == 2:
            i, j = group.index
            devigged.loc[i], devigged.loc[j] = devig_two_way(df.loc[i, "market_prob"], df.loc[j, "market_prob"])
        else:
            devigged.loc[group.index] *= PROP_VIG_FACTOR
    df = df.copy()
    df["market_prob"] = devigged
    return df
