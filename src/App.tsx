# =============================================================================
# PARLAY ENGINE v7.2 — INTERACTIVE BANKROLL
# Quant Sports Betting System
# =============================================================================
#
# Change Log:
# - Bankroll is no longer a command-line argument.
# - The script now interactively prompts the user to enter their bankroll at runtime.
# - This makes the engine more user-friendly and removes complex commands.
#
# All other logic remains, including:
# 1. Stubs for InformationAggregator and MasterProbabilityModel for deep analysis.
# 2. Tiered parlay construction based on specific odds ranges.
# 3. A multi-factor ParlayQualityScore (Value, Conviction, Certainty).
# 4. An optimization engine to "scrub" and refine each parlay.
# 5. A redundancy rule for creating a diversified, exclusive portfolio.
# =============================================================================

import argparse
import logging
from dataclasses import dataclass
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import requests
from scipy.linalg import LinAlgError
from scipy.stats import norm, multivariate_normal

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

ODDS_API_KEY = "268e7e91ea671f4710b057dde90bb5f2" # Replace with your actual key

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
MAX_DAILY_LOSS = 0.15
PROP_VIG_FACTOR = 0.9525

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
# DATA INGESTION & ODDS
# =============================================================================

def fetch_odds() -> pd.DataFrame:
    records = []
    base = "https://api.the-odds-api.com/v4/sports"
    for sport in SPORTS:
        url = f"{base}/{sport}/odds"
        params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h", "oddsFormat": "american"}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            for game in r.json():
                for book in game["bookmakers"]:
                    for market in book["markets"]:
                        for outcome in market["outcomes"]:
                            records.append({
                                "sport": sport, "game_id": game["id"], "sportsbook": book["title"],
                                "bet": outcome["name"], "odds": outcome["price"], "corr_group": "none",
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
            elif "corr_group" in df.columns and ri["corr_group"]!= "none" and ri["corr_group"] == rj["corr_group"]: r = RHO_SAME_GROUP
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
# INFORMATION & PROBABILITY MODELING STUBS
# =============================================================================

class InformationAggregator:
    """STUB: Gathers and synthesizes all data sources into a feature vector."""
    def generate_feature_vector(self, game_info: dict) -> dict:
        return {"elo_diff": 120.5, "key_player_injury_score": 0.8, "line_move_abs": 45}

class MasterProbabilityModel:
    """STUB: Generates high-conviction probabilities from synthesized features."""
    def __init__(self):
        if not SKLEARN_AVAILABLE: raise RuntimeError("scikit-learn required.")
        self.model = RandomForestClassifier(n_estimators=250, random_state=42)
        self._is_trained = False
        self.feature_names = []
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self._is_trained = True
        scores = cross_val_score(self.model, X, y, cv=5, scoring='brier_score_loss')
        logger.info(f"MasterProbabilityModel trained. Brier loss: {-np.mean(scores):.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_trained: raise RuntimeError("Model not trained.")
        return self.model.predict_proba(X[self.feature_names])[:, 1]

# =============================================================================
# PARLAY QUALITY & OPTIMIZATION
# =============================================================================

def get_prediction_certainty(model, X_live: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, 'estimators_'): return np.ones(len(X_live))
    preds = np.array([tree.predict_proba(X_live)[:, 1] for tree in model.estimators_])
    certainty = 1 / (np.std(preds, axis=0) + 1e-6)
    return (certainty - np.min(certainty)) / (np.max(certainty) - np.min(certainty) + 1e-6)

def calculate_parlay_quality_score(parlay: Parlay, leg_certainties: np.ndarray) -> ParlayQualityReport:
    W_VALUE, W_CONVICTION, W_CERTAINTY = 0.40, 0.30, 0.30
    value_score = 1 / (1 - parlay.ev) if parlay.ev < 1 else 1.0
    conviction_score = parlay.prob
    certainty_score = np.prod(leg_certainties) ** (1 / len(leg_certainties))
    total_score = (W_VALUE * value_score + W_CONVICTION * conviction_score + W_CERTAINTY * certainty_score)
    return ParlayQualityReport(total_score, value_score, conviction_score, certainty_score)

def refine_and_optimize_parlay(initial_parlay: Parlay, substitute_pool: List[Bet], df: pd.DataFrame, model: MasterProbabilityModel) -> Parlay:
    best_parlay = initial_parlay
    initial_indices = [df[df['bet'] == leg.description].index[0] for leg in best_parlay.legs]
    initial_features = df.loc[initial_indices]
    initial_certainties = get_prediction_certainty(model.model, initial_features[model.feature_names])
    best_quality = calculate_parlay_quality_score(best_parlay, initial_certainties)

    for i, leg_to_replace in enumerate(initial_parlay.legs):
        base_legs = [leg for j, leg in enumerate(initial_parlay.legs) if i!= j]
        for substitute in substitute_pool:
            if substitute.description in [l.description for l in base_legs]: continue
            new_legs = base_legs + [substitute]
            
            probs = np.array([leg.prob for leg in new_legs])
            indices = [df[df['bet'] == leg.description].index[0] for leg in new_legs]
            rho = build_corr_matrix(df.loc[indices])
            new_prob = monte_carlo(probs, rho, trials=10000)
            
            new_odds = np.prod([american_to_decimal(l.odds) for l in new_legs])
            new_ev = (new_prob * new_odds) - 1
            if new_ev <= 0: continue
            
            candidate = Parlay(new_legs, len(new_legs), new_odds, new_prob, new_ev, kelly(new_prob, new_odds))
            new_features = df.loc[indices]
            new_certainties = get_prediction_certainty(model.model, new_features[model.feature_names])
            new_quality = calculate_parlay_quality_score(candidate, new_certainties)

            if new_quality.quality_score > best_quality.quality_score:
                logger.info(f"OPTIMIZED: Swapping '{leg_to_replace.description}' for '{substitute.description}' improved score.")
                best_parlay, best_quality = candidate, new_quality
    return best_parlay

# =============================================================================
# STRATEGY, TIERING & PORTFOLIO SELECTION
# =============================================================================

def kelly(prob: float, decimal: float) -> float:
    b = decimal - 1
    return max(((prob * b - (1 - prob)) / b) * KELLY_FRACTION, 0.0) if b > 0 else 0.0

def build_strategy(df: pd.DataFrame) -> List[Bet]:
    bets = []
    for _, r in df.iterrows():
        dec = american_to_decimal(r["odds"])
        prob = r["my_prob"]
        ev = prob * dec - 1
        k = kelly(prob, dec)
        if ev > 0:
            bets.append(Bet(r["bet"], r["odds"], prob, ev, k, r.get("sport", ""), r.get("game_id", "")))
    return sorted(bets, key=lambda x: x.ev, reverse=True)

def categorize_bets_by_tier(bets: List[Bet]) -> dict:
    TIERS = {"tier1": (1200, 2000), "tier2": (2001, 3000), "tier3": (3001, 4000), "tier4": (4001, 5000), "tier5": (5001, float('inf'))}
    tiered = {name: [] for name in TIERS}
    for bet in bets:
        for name, (lower, upper) in TIERS.items():
            if lower <= bet.odds <= upper:
                tiered[name].append(bet)
                break
    return tiered

def build_and_evaluate_tiered_parlays(tiered_bets: dict, df: pd.DataFrame, leg_counts: List[int]) -> List[Parlay]:
    parlays = []
    pool = [bet for tier_list in tiered_bets.values() for bet in tier_list]
    for n_legs in leg_counts:
        if len(pool) < n_legs: continue
        for leg_combo in combinations(pool, n_legs):
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

def execute_bets(bets: List, bankroll: float) -> List:
    executed = []
    for bet in bets:
        stake = min(bankroll * bet.kelly, bankroll * MAX_EXPOSURE)
        if stake > 0:
            logger.info(f"Placing bet: {bet.description}, Stake: ${stake:.2f}")
            executed.append({"description": bet.description, "stake": stake})
    return executed

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Parlay Engine v7.2")
    parser.add_argument("--run", action="store_true", help="Execute bets via stub")
    args = parser.parse_args()

    # --- NEW: INTERACTIVE BANKROLL INPUT ---
    bankroll = 0.0
    while True:
        try:
            prompt = input("Please enter your total bankroll amount in USD (e.g., 50.75): ")
            bankroll = float(prompt)
            if bankroll <= 0:
                print("Bankroll must be a positive number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number without the '$' sign.")
    logger.info(f"Bankroll set to: ${bankroll:,.2f}")
    
    # --- 1. MODEL TRAINING (STUB) ---
    logger.info("Initializing models...")
    aggregator = InformationAggregator()
    model = MasterProbabilityModel()
    dummy_X = pd.DataFrame([aggregator.generate_feature_vector({}) for _ in range(100)])
    dummy_y = pd.Series(np.random.randint(0, 2, 100))
    model.train(dummy_X, dummy_y)

    # --- 2. DATA ACQUISITION & PREP ---
    logger.info("Fetching live odds...")
    df = fetch_odds()
    if df.empty:
        logger.warning("No API data. Exiting.")
        return
    df["market_prob"] = df["odds"].apply(implied_prob)
    df = apply_devig(df)

    # --- 3. APPLY MASTER PROBABILITY MODEL ---
    logger.info("Generating high-conviction probabilities...")
    live_features = pd.DataFrame([aggregator.generate_feature_vector({}) for _ in range(len(df))])
    df["my_prob"] = model.predict_proba(live_features)

    # --- 4. BUILD & EVALUATE ---
    logger.info("Building single bet strategies...")
    single_bets = build_strategy(df)
    tiered_bets = categorize_bets_by_tier(single_bets)
    
    logger.info("Generating all candidate parlays...")
    candidate_parlays = build_and_evaluate_tiered_parlays(tiered_bets, df, leg_counts=[3, 4, 5])
    if not candidate_parlays:
        logger.info("No +EV parlays could be constructed. Exiting.")
        return

    # --- 5. REFINE, OPTIMIZE & SCORE ---
    logger.info("Optimizing and refining candidate parlays...")
    optimized_parlays = []
    for parlay in candidate_parlays:
        sub_pool = [b for b in single_bets if b.description not in {l.description for l in parlay.legs}]
        optimized = refine_and_optimize_parlay(parlay, sub_pool, df, model)
        optimized_parlays.append(optimized)

    logger.info("Scoring final optimized parlays...")
    scored_parlays = []
    for parlay in optimized_parlays:
        indices = [df[df['bet'] == leg.description].index[0] for leg in parlay.legs]
        features = df.loc[indices]
        certainties = get_prediction_certainty(model.model, features[model.feature_names])
        quality = calculate_parlay_quality_score(parlay, certainties)
        scored_parlays.append((parlay, quality))

    scored_parlays.sort(key=lambda x: x[1].quality_score, reverse=True)

    # --- 6. SELECT FINAL PORTFOLIO ---
    logger.info("Selecting exclusive portfolio based on Quality Score...")
    final_candidates = [p for p, q in scored_parlays]
    final_portfolio = select_exclusive_parlay_portfolio(final_candidates)

    # --- 7. REPORT & EXECUTE ---
    if final_portfolio:
        logger.info(f"\n{'='*80}\nFINAL MUTUALLY EXCLUSIVE & OPTIMIZED PORTFOLIO\n{'='*80}")
        for parlay in final_portfolio:
            quality = next(q for p, q in scored_parlays if p == parlay)
            print(f"\n{parlay.leg_count}-LEG PARLAY | Quality Score: {quality.quality_score:.4f}")
            print(f" EV: {parlay.ev*100:+.2f}% | Win Prob: {parlay.prob*100:.4f}% | Kelly: {parlay.kelly*100:.3f}%")
            for leg in parlay.legs:
                print(f" - {leg.description:<25} ({leg.odds:+})")
        
        if args.run:
            logger.info("Executing final portfolio with the provided bankroll...")
            bets_to_execute = [Bet(
                " & ".join([l.description for l in p.legs]), decimal_to_american(p.decimal_odds), 
                p.prob, p.ev, p.kelly) for p in final_portfolio]
            execute_bets(bets_to_execute, bankroll=bankroll) # Using the interactive bankroll here
    else:
        logger.info("No parlays remained after optimization and selection.")

if __name__ == "__main__":
    main()