import json

import pandas as pd
import pytest

from agent.runner import SYSTEM_PROMPT, image_block, run_agent
from agent.tools import ALL_TOOLS, evaluate_props, price_custom_parlay
from engine import run_engine
from engine.odds import MAX_EXPOSURE, american_to_decimal, devig_two_way, implied_prob, kelly
from engine.odds_api import OddsAPIError, apply_devig, fetch_odds
from engine.parlays import Bet, build_candidate_parlays, joint_probability, select_exclusive_portfolio
from engine.ratings import InformationAggregator, placeholder_training_set


def slate(rows):
    df = pd.DataFrame(rows, columns=["game_id", "home_team", "away_team", "bet", "odds"])
    df["sport"] = "basketball_nba"
    df["market_prob"] = df["odds"].apply(implied_prob)
    return apply_devig(df)


ODDS = slate([
    ("g1", "Denver Nuggets", "New York Knicks", "Denver Nuggets", 150),
    ("g1", "Denver Nuggets", "New York Knicks", "New York Knicks", -170),
    ("g2", "Boston Celtics", "Los Angeles Lakers", "Boston Celtics", 140),
    ("g2", "Boston Celtics", "Los Angeles Lakers", "Los Angeles Lakers", -160),
    ("g3", "Miami Heat", "Golden State Warriors", "Miami Heat", 130),
    ("g3", "Miami Heat", "Golden State Warriors", "Golden State Warriors", -150),
    ("g4", "Chicago Bulls", "Brooklyn Nets", "Chicago Bulls", -110),
    ("g4", "Chicago Bulls", "Brooklyn Nets", "Brooklyn Nets", -110),
])


def test_odds_conversions():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert implied_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert sum(devig_two_way(implied_prob(-110), implied_prob(-110))) == pytest.approx(1.0)


def test_kelly_is_fractional_and_capped():
    assert kelly(0.4, 2.0) == 0.0
    assert kelly(0.55, 2.0) == pytest.approx(0.025)
    assert kelly(0.9, 3.0) == MAX_EXPOSURE


def test_joint_probability_independent_and_correlated():
    import numpy as np
    probs = np.array([0.6, 0.5])
    assert joint_probability(probs, np.eye(2)) == pytest.approx(0.30, abs=0.01)
    assert joint_probability(probs, np.array([[1, 0.5], [0.5, 1]])) > 0.32


def test_placeholder_training_has_both_classes():
    X, y = placeholder_training_set(InformationAggregator())
    assert len(X) == 30
    assert y.value_counts().min() >= 3


def test_moneyline_parlays_never_pair_both_sides_of_a_game():
    bets = [Bet("A", 200, 0.5, game_id="g1"), Bet("B", 200, 0.5, game_id="g1"), Bet("C", 200, 0.5, game_id="g2")]
    parlays = build_candidate_parlays(bets, [2], trials=20_000)
    assert parlays
    assert all(len({leg.game_id for leg in p.legs}) == 2 for p in parlays)


def test_portfolio_legs_are_exclusive():
    bets = [Bet(x, 200, 0.5, game_id=x) for x in "ABCD"]
    portfolio = select_exclusive_portfolio(build_candidate_parlays(bets, [2], trials=20_000), max_size=5)
    keys = [leg.key for p in portfolio for leg in p.legs]
    assert len(keys) == len(set(keys))


def test_run_engine_on_fixture_slate():
    result = run_engine(1000, odds=ODDS, trials=20_000)
    assert result["games"] == 4
    assert result["games_with_model_coverage"] == 3
    assert "placeholder" in result["caveat"]
    assert result["status"] in {"ok", "no_positive_ev_parlays"}
    for p in result["portfolio"]:
        assert p["ev_pct"] > 0
        assert p["stake_usd"] <= 1000 * MAX_EXPOSURE
    json.dumps(result)


def test_run_engine_with_no_odds():
    assert run_engine(100, odds=pd.DataFrame())["status"] == "no_odds"


def test_fetch_odds_requires_key(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(OddsAPIError):
        fetch_odds()


def test_evaluate_props_tiers_and_market_default():
    out = json.loads(evaluate_props.call({"bankroll": 1000, "props": [
        {"description": "Jokic 25+ pts", "odds": -150, "game": "DEN vs NYK", "estimated_prob": 0.75},
        {"description": "Brunson 30+ pts", "odds": 200, "game": "DEN vs NYK"},
    ]}))
    assert out[0]["tier"] == "LOCK" and out[0]["ev_pct"] > 0
    assert out[1]["prob_source"] == "market_no_vig" and out[1]["tier"] == "PASS" and out[1]["ev_pct"] < 0


def test_evaluate_props_rejects_bad_probability():
    with pytest.raises(ValueError):
        evaluate_props.call({"bankroll": 100, "props": [{"description": "x", "odds": 100, "game": "g", "estimated_prob": 75}]})


def test_price_custom_parlay():
    out = json.loads(price_custom_parlay.call({"bankroll": 500, "legs": [
        {"description": "A", "odds": 100, "game": "g1", "estimated_prob": 0.6},
        {"description": "B", "odds": 100, "game": "g2", "estimated_prob": 0.6},
    ]}))
    assert out["decimal_odds"] == pytest.approx(4.0)
    assert out["win_prob"] == pytest.approx(0.36, abs=0.01)
    assert out["american_odds"] == 300


def test_tool_schemas_build():
    assert {t.to_dict()["name"] for t in ALL_TOOLS} == {"run_parlay_engine", "get_slate", "evaluate_props", "price_custom_parlay"}


def test_image_block(tmp_path):
    png = tmp_path / "slip.png"
    png.write_bytes(b"\x89PNG")
    assert image_block(png)["source"]["media_type"] == "image/png"
    with pytest.raises(ValueError):
        image_block(tmp_path / "slip.txt")


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeClient:
    def __init__(self, messages):
        self.calls = []
        self.messages_out = messages
        self.beta = _Block(messages=_Block(tool_runner=self._runner))

    def _runner(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.messages_out)


def test_run_agent_wires_runner_and_returns_text(tmp_path):
    png = tmp_path / "s.png"
    png.write_bytes(b"\x89PNG")
    msgs = [
        _Block(stop_reason="tool_use", content=[_Block(type="tool_use", name="evaluate_props", input={"bankroll": 1})]),
        _Block(stop_reason="end_turn", content=[_Block(type="text", text="No bet tonight.")]),
    ]
    client, seen = _FakeClient(msgs), []
    assert run_agent("price these", [png], client=client, on_tool_call=lambda n, a: seen.append(n)) == "No bet tonight."
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == SYSTEM_PROMPT
    assert call["fallbacks"] == "default"
    assert call["messages"][0]["content"][0]["type"] == "image"
    assert seen == ["evaluate_props"]


def test_run_agent_surfaces_refusal():
    msgs = [_Block(stop_reason="refusal", stop_details=_Block(category=None, explanation="no"), content=[])]
    with pytest.raises(RuntimeError, match="declined"):
        run_agent("x", client=_FakeClient(msgs))
