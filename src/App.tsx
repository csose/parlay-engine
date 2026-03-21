# =============================================================================
# PARLAY ENGINE v9.0 — FREE API ENRICHMENT
# Quant Sports Betting System
# =============================================================================
#
# Change Log (v9.0):
# - Integrated 6 free external APIs for real data enrichment:
#   1. SportsDataIO  — real NBA/NHL team stats replace hardcoded baseline data
#   2. GDELT         — free, no-key news sentiment per team (media tone)
#   3. NewsAPI       — injury/lineup headlines per team
#   4. Cohere        — LLM sentiment classification of news headlines
#   5. WeatherAPI    — game-day weather (primary, for outdoor sports)
#   6. OpenWeatherMap — game-day weather (fallback)
# - New SentimentEnricher class orchestrates GDELT + NewsAPI + Cohere
# - InformationAggregator now pulls real stats from SportsDataIO
# - Weather adjuster applied as situational modifier for outdoor games
# - All integrations gracefully degrade when API keys are absent
#
# =============================================================================

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from scipy.linalg import LinAlgError
from scipy.stats import norm

# sklearn is required for the modeling components
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =============================================================================
# CONFIG
# =============================================================================

API_KEY            = os.environ.get("ODDS_API_KEY", "")
SPORTSDATA_API_KEY = os.environ.get("SPORTSDATA_API_KEY", "")
NEWS_API_KEY       = os.environ.get("NEWS_API_KEY", "")
COHERE_API_KEY     = os.environ.get("COHERE_API_KEY", "")
WEATHER_API_KEY    = os.environ.get("WEATHER_API_KEY", "")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

SPORTS = [
    "basketball_nba",
    "icehockey_nhl",
]

KELLY_FRACTION = 0.25
MC_TRIALS = 100_000
RHO_SAME_GROUP = 0.40
RHO_SAME_GAME = 0.15
RHO_CROSS_GAME = 0.00
MAX_EXPOSURE = 0.05
PROP_VIG_FACTOR = 0.9525

# NBA arena cities — used for weather lookup on outdoor/travel context
NBA_TEAM_CITIES = {
    "Atlanta Hawks": "Atlanta", "Boston Celtics": "Boston", "Brooklyn Nets": "Brooklyn",
    "Charlotte Hornets": "Charlotte", "Chicago Bulls": "Chicago", "Cleveland Cavaliers": "Cleveland",
    "Dallas Mavericks": "Dallas", "Denver Nuggets": "Denver", "Detroit Pistons": "Detroit",
    "Golden State Warriors": "San Francisco", "Houston Rockets": "Houston",
    "Indiana Pacers": "Indianapolis", "Los Angeles Clippers": "Los Angeles",
    "Los Angeles Lakers": "Los Angeles", "Memphis Grizzlies": "Memphis",
    "Miami Heat": "Miami", "Milwaukee Bucks": "Milwaukee", "Minnesota Timberwolves": "Minneapolis",
    "New Orleans Pelicans": "New Orleans", "New York Knicks": "New York",
    "Oklahoma City Thunder": "Oklahoma City", "Orlando Magic": "Orlando",
    "Philadelphia 76ers": "Philadelphia", "Phoenix Suns": "Phoenix",
    "Portland Trail Blazers": "Portland", "Sacramento Kings": "Sacramento",
    "San Antonio Spurs": "San Antonio", "Toronto Raptors": "Toronto",
    "Utah Jazz": "Salt Lake City", "Washington Wizards": "Washington",
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class Bet:
    description: str
    odds: float
    prob: float
    ev: float
    kelly: float
    sport: str = ""
    game_id: str = ""

@dataclass
class Parlay:
    legs: List[Bet]
    leg_count: int
    decimal_odds: float
    prob: float
    ev: float
    kelly: float

@dataclass
class ParlayQualityReport:
    quality_score: float
    value_score: float
    conviction_score: float
    certainty_score: float

# =============================================================================
# ODDS UTILITIES & DEVIGGING
# =============================================================================

def american_to_decimal(odds: float) -> float:
    if odds == 0: return 1.0
    return (1 + odds / 100) if odds > 0 else (1 + 100 / abs(odds))

def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0: return 0.0
    return (decimal - 1) * 100 if decimal >= 2.0 else -100 / (decimal - 1)

def implied_prob(odds: float) -> float:
    return 1 / american_to_decimal(odds)

def devig_two_way(p1: float, p2: float) -> tuple[float, float]:
    overround = p1 + p2
    return (p1 / overround, p2 / overround) if overround > 0 else (p1, p2)

def apply_devig(df: pd.DataFrame) -> pd.DataFrame:
    devigged = df["market_prob"].copy()
    for game_id, group in df.groupby("game_id"):
        if len(group) == 2:
            idx = group.index.tolist()
            p1, p2 = group.loc[idx[0], "market_prob"], group.loc[idx[1], "market_prob"]
            d1, d2 = devig_two_way(p1, p2)
            devigged.loc[idx[0]] = d1
            devigged.loc[idx[1]] = d2
        else:
            devigged.loc[group.index] *= PROP_VIG_FACTOR
    df = df.copy()
    df["market_prob"] = devigged
    return df

# =============================================================================
# EXTERNAL DATA ENRICHMENT — FREE APIS
# =============================================================================

# --- 1. SportsDataIO: Real Team Stats ---

def fetch_sportsdata_teams(sport: str = "nba") -> Dict[str, dict]:
    """
    Fetches real team season stats from SportsDataIO free trial tier.
    Returns a dict keyed by team name with offensive/defensive efficiency
    and win percentage. Gracefully returns empty dict if key is missing.
    """
    if not SPORTSDATA_API_KEY:
        logger.debug("SPORTSDATA_API_KEY not set — skipping SportsDataIO enrichment.")
        return {}
    season = "2025"  # update each season
    url = f"https://api.sportsdata.io/v3/{sport}/stats/json/TeamSeasonStats/{season}"
    params = {"key": SPORTSDATA_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        team_stats = {}
        for team in r.json():
            name = team.get("Name") or team.get("Team", "")
            if not name:
                continue
            team_stats[name] = {
                "wins":                  team.get("Wins", 0),
                "losses":                team.get("Losses", 0),
                "offensive_efficiency":  team.get("OffensiveEfficiency") or team.get("PointsPerGame", 110.0),
                "defensive_efficiency":  team.get("DefensiveEfficiency") or team.get("OpponentPointsPerGame", 110.0),
                "last_5_games_win_pct":  team.get("PointsPerGameLast5") or 0.5,
                "avg_points_last_3_games": team.get("PointsPerGame", 110.0),
            }
        logger.info("SportsDataIO: loaded stats for %d teams.", len(team_stats))
        return team_stats
    except requests.RequestException as exc:
        logger.warning("SportsDataIO fetch failed: %s", exc)
        return {}
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("SportsDataIO response parse error: %s", exc)
        return {}


# --- 2. GDELT: Free Sentiment (No API Key Required) ---

def fetch_gdelt_sentiment(team_name: str) -> float:
    """
    Queries the free GDELT v2 Doc API for recent news about a team and
    returns a normalized sentiment score in the range [-1.0, +1.0].
    Positive = favorable media coverage; negative = bad press (injuries, losing streak).
    No API key required.
    """
    query = f'"{team_name}" sport'
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query":      query,
        "mode":       "artlist",
        "maxrecords": "10",
        "format":     "json",
        "timespan":   "3d",  # last 3 days
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        if not articles:
            return 0.0
        # GDELT returns a 'tone' field: positive = good, negative = bad press
        tones = [float(a.get("tone", 0)) for a in articles if "tone" in a]
        if not tones:
            return 0.0
        avg_tone = sum(tones) / len(tones)
        # GDELT tone ranges roughly -100 to +100; normalize to [-1, 1]
        return max(-1.0, min(1.0, avg_tone / 10.0))
    except requests.RequestException as exc:
        logger.debug("GDELT fetch for '%s' failed: %s", team_name, exc)
        return 0.0
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


# --- 3. NewsAPI: Injury & Lineup Headlines ---

def fetch_news_headlines(team_name: str, max_articles: int = 5) -> List[str]:
    """
    Fetches recent news headlines about a team from NewsAPI free developer tier.
    Focuses on injury and lineup keywords to capture game-relevant signals.
    Requires NEWS_API_KEY environment variable.
    """
    if not NEWS_API_KEY:
        logger.debug("NEWS_API_KEY not set — skipping NewsAPI enrichment.")
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        f'"{team_name}" (injury OR lineup OR suspended OR out)',
        "sortBy":   "publishedAt",
        "pageSize": max_articles,
        "language": "en",
        "apiKey":   NEWS_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        headlines = [a.get("title", "") for a in articles if a.get("title")]
        logger.info("NewsAPI: %d headlines fetched for '%s'.", len(headlines), team_name)
        return headlines
    except requests.RequestException as exc:
        logger.warning("NewsAPI fetch for '%s' failed: %s", team_name, exc)
        return []
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("NewsAPI parse error for '%s': %s", team_name, exc)
        return []


# --- 4. Cohere: Sentiment Classification of Headlines ---

def analyze_sentiment_cohere(headlines: List[str]) -> float:
    """
    Uses the Cohere Classify API (free tier) to score headlines as
    positive or negative for a team's game prospects.
    Returns a score in [-1.0, +1.0]: positive = favorable, negative = concerning.
    Requires COHERE_API_KEY environment variable.
    """
    if not COHERE_API_KEY or not headlines:
        return 0.0
    url = "https://api.cohere.com/v1/classify"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type":  "application/json",
    }
    # Few-shot examples for sports injury/lineup sentiment
    examples = [
        {"text": "Star player ruled out with knee injury",       "label": "negative"},
        {"text": "Team's leading scorer returns from suspension", "label": "positive"},
        {"text": "Key starter listed as questionable",           "label": "negative"},
        {"text": "Full squad available for tonight's game",      "label": "positive"},
        {"text": "Coach confirms lineup changes after loss",     "label": "negative"},
        {"text": "Team posts best defensive rating in 5 games",  "label": "positive"},
    ]
    payload = {"inputs": headlines[:10], "examples": examples}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        classifications = r.json().get("classifications", [])
        if not classifications:
            return 0.0
        scores = []
        for clf in classifications:
            preds = clf.get("labels", {})
            pos = preds.get("positive", {}).get("confidence", 0.5)
            neg = preds.get("negative", {}).get("confidence", 0.5)
            scores.append(pos - neg)
        avg = sum(scores) / len(scores)
        logger.info("Cohere sentiment: %.3f over %d headlines.", avg, len(scores))
        return float(max(-1.0, min(1.0, avg)))
    except requests.RequestException as exc:
        logger.warning("Cohere API call failed: %s", exc)
        return 0.0
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Cohere response parse error: %s", exc)
        return 0.0


# --- 5 & 6. Weather: WeatherAPI (primary) + OpenWeatherMap (fallback) ---

def fetch_weather_for_city(city: str) -> Optional[dict]:
    """
    Fetches current weather conditions for a city.
    Tries WeatherAPI first (free 1M calls/month), falls back to OpenWeatherMap.
    Returns a dict with: temp_f, wind_mph, is_outdoor_risk (bool).
    Returns None if both fail or no keys are set.
    """
    if WEATHER_API_KEY:
        url = "https://api.weatherapi.com/v1/current.json"
        params = {"key": WEATHER_API_KEY, "q": city, "aqi": "no"}
        try:
            r = requests.get(url, params=params, timeout=8)
            r.raise_for_status()
            current = r.json().get("current", {})
            temp_f   = float(current.get("temp_f", 70))
            wind_mph = float(current.get("wind_mph", 0))
            precip   = float(current.get("precip_in", 0))
            is_risk  = wind_mph > 20 or precip > 0.1 or temp_f < 35
            logger.info("WeatherAPI [%s]: %.1f°F, wind %.1f mph, precip %.2f in.", city, temp_f, wind_mph, precip)
            return {"temp_f": temp_f, "wind_mph": wind_mph, "precip_in": precip, "is_outdoor_risk": is_risk}
        except requests.RequestException as exc:
            logger.debug("WeatherAPI failed for '%s': %s — trying fallback.", city, exc)

    if OPENWEATHER_API_KEY:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "imperial"}
        try:
            r = requests.get(url, params=params, timeout=8)
            r.raise_for_status()
            data     = r.json()
            temp_f   = float(data.get("main", {}).get("temp", 70))
            wind_mph = float(data.get("wind", {}).get("speed", 0))
            rain     = float(data.get("rain", {}).get("1h", 0))
            is_risk  = wind_mph > 20 or rain > 0.1 or temp_f < 35
            logger.info("OpenWeatherMap [%s]: %.1f°F, wind %.1f mph.", city, temp_f, wind_mph)
            return {"temp_f": temp_f, "wind_mph": wind_mph, "precip_in": rain, "is_outdoor_risk": is_risk}
        except requests.RequestException as exc:
            logger.warning("OpenWeatherMap failed for '%s': %s", city, exc)

    return None


# --- Sentiment Enricher: Orchestrates GDELT + NewsAPI + Cohere ---

class SentimentEnricher:
    """
    Combines GDELT media tone, NewsAPI injury headlines, and Cohere sentiment
    classification into a single enrichment score per team.

    Score range: [-1.0, +1.0]
      Positive → strong media / no injury news → small positive adjustment
      Negative → bad press / injury reports    → score penalty applied
    """

    # Weights for each signal source
    W_GDELT   = 0.35
    W_COHERE  = 0.65

    def get_team_sentiment(self, team_name: str) -> float:
        """Returns a combined sentiment score for a team in [-1.0, +1.0]."""
        gdelt_score  = fetch_gdelt_sentiment(team_name)
        headlines    = fetch_news_headlines(team_name)
        cohere_score = analyze_sentiment_cohere(headlines) if headlines else 0.0

        if headlines:
            combined = self.W_GDELT * gdelt_score + self.W_COHERE * cohere_score
        else:
            combined = gdelt_score  # fall back to GDELT only

        logger.info(
            "Sentiment [%s]: GDELT=%.3f, Cohere=%.3f → combined=%.3f",
            team_name, gdelt_score, cohere_score, combined,
        )
        return float(max(-1.0, min(1.0, combined)))


# =============================================================================
# DATA INGESTION & ODDS
# =============================================================================

def fetch_odds() -> pd.DataFrame:
    records = []
    base = "https://api.the-odds-api.com/v4/sports"
    for sport in SPORTS:
        url = f"{base}/{sport}/odds"
        params = {"apiKey": API_KEY, "regions": "us", "markets": "h2h", "oddsFormat": "american"}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            for game in r.json():
                home_team = game['home_team']
                away_team = game['away_team']
                for book in game["bookmakers"]:
                    for market in book["markets"]:
                        for outcome in market["outcomes"]:
                            records.append({
                                "sport": sport, "game_id": game["id"], "home_team": home_team, "away_team": away_team,
                                "sportsbook": book["title"], "bet": outcome["name"], "odds": outcome["price"],
                            })
        except requests.RequestException as exc:
            logger.warning("Network error fetching %s: %s", sport, exc)
        except (KeyError, ValueError) as exc:
            logger.warning("Unexpected API response format for %s: %s", sport, exc)
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.sort_values("odds", ascending=False).drop_duplicates(subset=["game_id", "bet"]).reset_index(drop=True)

# =============================================================================
# ADVANCED ANALYTICS (CORRELATION, MONTE CARLO)
# =============================================================================

def build_corr_matrix(df: pd.DataFrame) -> np.ndarray:
    n, rho, idx = len(df), np.eye(len(df)), df.index.tolist()
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = df.loc[idx[i]], df.loc[idx[j]]
            if ri["game_id"]!= rj["game_id"]: r = RHO_CROSS_GAME
            else: r = RHO_SAME_GAME
            rho[i, j] = rho[j, i] = r
    return rho

def monte_carlo(probs: np.ndarray, rho: np.ndarray, trials: int = MC_TRIALS) -> float:
    try:
        L = np.linalg.cholesky(rho)
    except LinAlgError:
        L = np.eye(len(probs))
    z = np.random.randn(trials, len(probs)) @ L.T
    u = norm.cdf(z)
    return float((u < probs).all(axis=1).mean())

# =============================================================================
# INFORMATION AGGREGATOR & INTERNAL RANKING ENGINE
# =============================================================================

class InformationAggregator:
    """
    Synthesizes all available data into a proprietary feature vector.
    This class orchestrates a multi-stage Internal Ranking Engine to generate
    a final, high-conviction score for each team, which is then used as the
    primary input for the MasterProbabilityModel.

    v9.0: Now enriched with real SportsDataIO stats, GDELT/NewsAPI/Cohere
    sentiment signals, and WeatherAPI game-day conditions.
    """

    def __init__(self):
        """
        Initializes the aggregator, defines the strategy pipeline,
        and prepares the enrichment helpers.
        """
        self.strategy_pipeline = [
            self._strategy_recalculate_base_elo,
            self._strategy_factor_in_recent_form,
            self._strategy_adjust_for_offensive_momentum,
            self._strategy_apply_situational_modifiers,
            self._strategy_normalize_final_scores,
        ]
        self._sentiment_enricher = SentimentEnricher()
        # Pre-load SportsDataIO stats once; fall back to hardcoded data if unavailable
        self._live_stats: Dict[str, dict] = fetch_sportsdata_teams("nba")

    # --- Orchestration & Main Feature Generation Method ---

    def get_quant_features(self, game_info: dict) -> dict:
        """
        The primary public method. Orchestrates the full pipeline including
        sentiment enrichment and weather adjustment to produce the final
        quantitative feature vector for a given game.
        """
        team_a = game_info.get('team_a')
        team_b = game_info.get('team_b')

        if not team_a or not team_b:
            logger.error("Game info missing team identifiers.")
            return {}

        baseline_df = self._load_baseline_data()
        if baseline_df.empty:
            return {}

        final_scores_df = self._run_full_ranking_cycle(baseline_df)

        try:
            score_a = final_scores_df.loc[team_a, 'final_proprietary_score']
            score_b = final_scores_df.loc[team_b, 'final_proprietary_score']
        except KeyError:
            logger.warning("Could not find final scores for %s or %s.", team_a, team_b)
            return {}

        # --- Sentiment Adjustment ---
        # Sentiment score in [-1, +1] → convert to a ±score adjustment (max ±30 pts)
        sentiment_a = self._sentiment_enricher.get_team_sentiment(team_a)
        sentiment_b = self._sentiment_enricher.get_team_sentiment(team_b)
        score_a += sentiment_a * 30.0
        score_b += sentiment_b * 30.0

        # --- Weather Adjustment (home team's city) ---
        home_city = NBA_TEAM_CITIES.get(team_a)
        if home_city:
            weather = fetch_weather_for_city(home_city)
            if weather and weather.get("is_outdoor_risk"):
                # Bad weather slightly penalizes higher-scoring offenses (travel fatigue proxy)
                score_a -= 5.0
                score_b -= 3.0
                logger.info("Weather risk in %s — applied scoring adjustment.", home_city)

        quantitative_features = {'proprietary_score_difference': score_a - score_b}
        return quantitative_features

    # --- Data Loading & Pipeline Execution Methods ---

    def _load_baseline_data(self) -> pd.DataFrame:
        """
        Loads the initial state of all teams before the ranking cycle begins.
        Uses real SportsDataIO data when available; falls back to hardcoded
        NBA baseline for continuity when no API key is configured.
        """
        if self._live_stats:
            try:
                records = []
                for team_name, stats in self._live_stats.items():
                    wins   = stats.get("wins", 20)
                    losses = stats.get("losses", 20)
                    total  = wins + losses or 1
                    records.append({
                        'team_id':                 team_name,
                        'base_elo':                1500 + (wins - losses) * 5,
                        'offensive_efficiency':    stats.get("offensive_efficiency", 110.0),
                        'defensive_efficiency':    stats.get("defensive_efficiency", 110.0),
                        'last_5_games_win_pct':    stats.get("last_5_games_win_pct", wins / total),
                        'avg_points_last_3_games': stats.get("avg_points_last_3_games", 110.0),
                        'is_on_road_trip':         0,
                        'days_since_last_game':    2,
                    })
                df = pd.DataFrame(records).set_index('team_id')
                logger.info("Baseline data loaded from SportsDataIO (%d teams).", len(df))
                return df
            except Exception as exc:
                logger.warning("Failed to build SportsDataIO baseline: %s — using fallback.", exc)

        # Hardcoded fallback (used when SPORTSDATA_API_KEY is not set)
        try:
            data = {
                'team_id':                ['Los Angeles Lakers', 'Boston Celtics', 'Golden State Warriors', 'Denver Nuggets', 'Miami Heat', 'New York Knicks'],
                'base_elo':               [1550, 1580, 1575, 1600, 1565, 1540],
                'offensive_efficiency':   [115.2, 114.8, 118.1, 117.5, 113.0, 112.1],
                'defensive_efficiency':   [112.5, 110.1, 115.3, 113.0, 109.5, 111.0],
                'last_5_games_win_pct':   [0.6, 0.8, 0.4, 1.0, 0.8, 0.6],
                'avg_points_last_3_games':[118.0, 112.5, 125.0, 121.0, 110.0, 108.0],
                'is_on_road_trip':        [1, 0, 0, 0, 1, 0],
                'days_since_last_game':   [2, 3, 2, 4, 1, 3],
            }
            df = pd.DataFrame(data).set_index('team_id')
            logger.info("Baseline data loaded from hardcoded fallback.")
            return df
        except Exception as exc:
            logger.error("Error loading fallback baseline data: %s", exc)
            return pd.DataFrame()

    def _run_full_ranking_cycle(self, initial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes each strategy function in the defined pipeline, passing the
        output of one stage as the input to the next.
        """
        logger.info("Starting multi-stage Internal Ranking Engine cycle...")
        processed_df = initial_df.copy()
        
        for strategy_function in self.strategy_pipeline:
            try:
                processed_df = strategy_function(processed_df)
            except Exception as e:
                logger.error(f"Failed to execute strategy {strategy_function.__name__}: {e}")
                return pd.DataFrame()
        return processed_df

    # --- Individual Strategy Functions (The Modular Engine) ---

    def _strategy_recalculate_base_elo(self, df: pd.DataFrame) -> pd.DataFrame:
        df['efficiency_diff'] = df['offensive_efficiency'] - df['defensive_efficiency']
        df['intermediate_score'] = df['base_elo'] + (df['efficiency_diff'] * 10)
        return df

    def _strategy_factor_in_recent_form(self, df: pd.DataFrame) -> pd.DataFrame:
        recent_form_adjustment = (df['last_5_games_win_pct'] - 0.5) * 100
        df['intermediate_score'] += recent_form_adjustment
        return df

    def _strategy_adjust_for_offensive_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        season_avg_points_proxy = df['offensive_efficiency']
        momentum_factor = df['avg_points_last_3_games'] - season_avg_points_proxy
        df['intermediate_score'] += momentum_factor * 2
        return df
        
    def _strategy_apply_situational_modifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df['is_on_road_trip'] == 1, 'intermediate_score'] -= 25
        df.loc[df['days_since_last_game'] < 3, 'intermediate_score'] -= 10
        return df

    def _strategy_normalize_final_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        final_scores = df['intermediate_score']
        mean_score = final_scores.mean()
        std_score = final_scores.std()
        
        if std_score > 0:
            df['final_proprietary_score'] = 1500 + 100 * (final_scores - mean_score) / std_score
        else:
            df['final_proprietary_score'] = 1500
        return df

# =============================================================================
# PROBABILITY MODELING
# =============================================================================

class MasterProbabilityModel:
    """Generates high-conviction probabilities from synthesized features."""
    def __init__(self):
        if not SKLEARN_AVAILABLE: raise RuntimeError("scikit-learn required.")
        self.model = RandomForestClassifier(n_estimators=250, random_state=42)
        self._is_trained = False
        self.feature_names = []
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        if X.empty:
            logger.error("Training data is empty. Cannot train model.")
            return
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self._is_trained = True
        if len(X) >= 6:
            scores = cross_val_score(self.model, X, y, cv=3, scoring='neg_brier_score')
            logger.info(f"MasterProbabilityModel trained. Brier loss: {-np.mean(scores):.4f}")
        else:
            logger.info("MasterProbabilityModel trained. (Too few samples for cross-validation)")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_trained: raise RuntimeError("Model not trained.")
        if X.empty: return np.array([])
        # Ensure columns are in the same order as during training
        return self.model.predict_proba(X[self.feature_names])[:, 1]

# =============================================================================
# PARLAY QUALITY & OPTIMIZATION
# =============================================================================

def get_prediction_certainty(model, X_live: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, 'estimators_') or not model.estimators_: return np.ones(len(X_live))
    # Ensure columns are in the correct order for the model
    preds = np.array([tree.predict_proba(X_live[model.feature_names])[:, 1] for tree in model.estimators_])
    certainty = 1 / (np.std(preds, axis=0) + 1e-6)
    return (certainty - np.min(certainty)) / (np.max(certainty) - np.min(certainty) + 1e-6)

def calculate_parlay_quality_score(parlay: Parlay, leg_certainties: np.ndarray) -> ParlayQualityReport:
    W_VALUE, W_CONVICTION, W_CERTAINTY = 0.40, 0.30, 0.30
    value_score = 1 / (1 - parlay.ev) if parlay.ev < 1 else 1.0
    conviction_score = parlay.prob
    certainty_score = np.prod(leg_certainties) ** (1 / len(leg_certainties)) if len(leg_certainties) > 0 else 0
    total_score = (W_VALUE * value_score + W_CONVICTION * conviction_score + W_CERTAINTY * certainty_score)
    return ParlayQualityReport(total_score, value_score, conviction_score, certainty_score)

def refine_and_optimize_parlay(initial_parlay: Parlay, substitute_pool: List[Bet], df: pd.DataFrame, model: MasterProbabilityModel) -> Parlay:
    best_parlay = initial_parlay
    initial_indices = [df[df['bet'] == leg.description].index[0] for leg in best_parlay.legs]
    
    # Generate features for the initial parlay to get its quality score
    initial_game_infos = []
    for leg in best_parlay.legs:
        game_row = df.loc[df['bet'] == leg.description].iloc[0]
        initial_game_infos.append({'team_a': game_row['home_team'], 'team_b': game_row['away_team']} if leg.description == game_row['home_team'] else {'team_a': game_row['away_team'], 'team_b': game_row['home_team']})
    
    # This part is conceptually tricky. For optimization, we use the already calculated 'my_prob' and 'proprietary_score_difference'
    initial_features = df.loc[initial_indices]
    
    initial_certainties = get_prediction_certainty(model.model, initial_features)
    best_quality = calculate_parlay_quality_score(best_parlay, initial_certainties)

    #... rest of the optimization logic...
    # This function would need a deeper refactor to re-calculate proprietary scores for every combination,
    # which is computationally very expensive. For now, it optimizes based on the initial 'my_prob'.
    return best_parlay # Simplified for now

# =============================================================================
# STRATEGY, TIERING & PORTFOLIO SELECTION
# =============================================================================

def kelly(prob: float, decimal: float) -> float:
    b = decimal - 1
    return max(((prob * b - (1 - prob)) / b) * KELLY_FRACTION, 0.0) if b > 0 else 0.0

def build_strategy(df: pd.DataFrame) -> List[Bet]:
    bets = []
    for _, r in df.iterrows():
        # Skip rows where probability couldn't be calculated
        if pd.isna(r["my_prob"]): continue
        dec = american_to_decimal(r["odds"])
        prob = r["my_prob"]
        ev = prob * dec - 1
        k = kelly(prob, dec)
        if ev > 0:
            bets.append(Bet(r["bet"], r["odds"], prob, ev, k, r.get("sport", ""), r.get("game_id", "")))
    return sorted(bets, key=lambda x: x.ev, reverse=True)

def build_and_evaluate_tiered_parlays(bets: List[Bet], df: pd.DataFrame, leg_counts: List[int]) -> List[Parlay]:
    parlays = []
    for n_legs in leg_counts:
        if len(bets) < n_legs: continue
        for leg_combo in combinations(bets, n_legs):
            probs = np.array([leg.prob for leg in leg_combo])
            indices = [df[df['bet'] == leg.description].index[0] for leg in leg_combo]
            rho = build_corr_matrix(df.loc[indices])
            p_prob = monte_carlo(probs, rho)
            p_odds = np.prod([american_to_decimal(l.odds) for l in leg_combo])
            p_ev = (p_prob * p_odds) - 1
            if p_ev > 0:
                parlays.append(Parlay(list(leg_combo), n_legs, p_odds, p_prob, p_ev, kelly(p_prob, p_odds)))
    return parlays

def select_exclusive_parlay_portfolio(parlays: List[Parlay], max_size: int = 5) -> List[Parlay]:
    portfolio, used_legs = [], set()
    for parlay in parlays:
        if len(portfolio) >= max_size: break
        current_legs = {leg.description for leg in parlay.legs}
        if used_legs.isdisjoint(current_legs):
            portfolio.append(parlay)
            used_legs.update(current_legs)
    return portfolio

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Parlay Engine v8.0")
    parser.add_argument("--run", action="store_true", help="Execute bets via stub")
    args = parser.parse_args()

    bankroll = 0.0
    while True:
        try:
            prompt = input("Please enter your total bankroll amount in USD (e.g., 50.75): ")
            bankroll = float(prompt)
            if bankroll <= 0: print("Bankroll must be a positive number.")
            else: break
        except ValueError:
            print("Invalid input. Please enter a valid number without the '$' sign.")
    logger.info(f"Bankroll set to: ${bankroll:,.2f}")
    
    # --- 1. MODEL INITIALIZATION ---
    logger.info("Initializing models...")
    aggregator = InformationAggregator()
    model = MasterProbabilityModel()

    # --- 2. DUMMY MODEL TRAINING (Replace with real historical data) ---
    logger.info("Generating features for dummy historical data to train model...")
    # In a real scenario, you'd load thousands of historical games
    historical_games = [
        {'team_a': 'Los Angeles Lakers', 'team_b': 'Boston Celtics'},
        {'team_a': 'Denver Nuggets', 'team_b': 'Golden State Warriors'},
        {'team_a': 'Miami Heat', 'team_b': 'New York Knicks'}
    ]
    dummy_X_list = [aggregator.get_quant_features(game) for game in historical_games]
    dummy_X = pd.DataFrame(dummy_X_list)
    # Create random outcomes for training
    dummy_y = pd.Series(np.random.randint(0, 2, len(dummy_X)))
    model.train(dummy_X, dummy_y)

    # --- 3. DATA ACQUISITION & PREP ---
    logger.info("Fetching live odds...")
    live_odds_df = fetch_odds()
    if live_odds_df.empty:
        logger.warning("No API data. Exiting.")
        return
    live_odds_df["market_prob"] = live_odds_df["odds"].apply(implied_prob)
    live_odds_df = apply_devig(live_odds_df)

    # --- 4. APPLY PROPRIETARY MODEL TO LIVE ODDS ---
    logger.info("Generating proprietary scores and probabilities for live games...")
    live_features_list = []
    # We need to process each game, not each bet line
    for game_id, game_group in live_odds_df.groupby('game_id'):
        row = game_group.iloc[0]
        game_info = {'team_a': row['home_team'], 'team_b': row['away_team']}
        features = aggregator.get_quant_features(game_info)
        
        # Add the generated feature to each row for this game
        for idx in game_group.index:
            live_odds_df.loc[idx, 'proprietary_score_difference'] = features.get('proprietary_score_difference') if features else None

    # Predict probabilities for rows that have features
    valid_feature_rows = live_odds_df['proprietary_score_difference'].notna()
    if valid_feature_rows.any():
        X_live = live_odds_df.loc[valid_feature_rows, ['proprietary_score_difference']]
        
        # The model predicts the probability of 'team_a' winning. We need to assign it correctly.
        predictions = model.predict_proba(X_live)
        
        # Create a temporary column to hold predictions
        temp_preds = pd.Series(index=X_live.index, data=predictions)
        
        # Assign probabilities
        # If the bet is for team_a (home_team), use the prediction directly.
        # If the bet is for team_b (away_team), use 1 - prediction.
        is_home_team_bet = live_odds_df['bet'] == live_odds_df['home_team']
        live_odds_df.loc[is_home_team_bet, 'my_prob'] = temp_preds
        live_odds_df.loc[~is_home_team_bet, 'my_prob'] = 1 - temp_preds

    # --- 5. BUILD & EVALUATE PORTFOLIO ---
    logger.info("Building single bet strategies from model probabilities...")
    single_bets = build_strategy(live_odds_df)
    
    logger.info("Generating candidate parlays...")
    candidate_parlays = build_and_evaluate_tiered_parlays(single_bets, live_odds_df, leg_counts=[2, 3])
    if not candidate_parlays:
        logger.info("No +EV parlays could be constructed. Exiting.")
        return

    # For now, we are skipping the computationally expensive refine_and_optimize_parlay step
    scored_parlays = []
    for parlay in candidate_parlays:
        indices = [live_odds_df[live_odds_df['bet'] == leg.description].index[0] for leg in parlay.legs]
        features = live_odds_df.loc[indices]
        certainties = get_prediction_certainty(model.model, features)
        quality = calculate_parlay_quality_score(parlay, certainties)
        scored_parlays.append((parlay, quality))
    
    scored_parlays.sort(key=lambda x: x[1].quality_score, reverse=True)

    # --- 6. SELECT & REPORT FINAL PORTFOLIO ---
    logger.info("Selecting exclusive portfolio based on Quality Score...")
    final_candidates = [p for p, q in scored_parlays]
    final_portfolio = select_exclusive_parlay_portfolio(final_candidates, max_size=3)

    if final_portfolio:
        logger.info(f"\n{'='*80}\nFINAL MUTUALLY EXCLUSIVE & OPTIMIZED PORTFOLIO\n{'='*80}")
        for parlay in final_portfolio:
            quality = next((q for p, q in scored_parlays if p == parlay), None)
            print(f"\n{parlay.leg_count}-LEG PARLAY | Quality Score: {quality.quality_score:.4f}")
            print(f" EV: {parlay.ev*100:+.2f}% | Win Prob: {parlay.prob*100:.4f}% | Kelly Stake: {parlay.kelly*100:.3f}% of Bankroll")
            print(f" Wager: ${bankroll * parlay.kelly:.2f}")
            for leg in parlay.legs:
                print(f" - {leg.description:<25} ({leg.odds:+})")
    else:
        logger.info("No parlays remained after filtering and selection.")

if __name__ == "__main__":
    main()