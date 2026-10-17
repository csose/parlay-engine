# parlay-engine

A sports betting parlay agent, plus the original screenshot dashboard.

## Agent

`agent/` is a Claude agent that runs the quant engine as tools. Ask it in plain English, or hand it sportsbook screenshots, and it fetches odds, prices bets, builds parlays, and sizes stakes.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # Claude
export ODDS_API_KEY=...        # the-odds-api.com, needed for live slates

.venv/bin/python -m agent "Build tonight's NBA parlays with a \$500 bankroll"
.venv/bin/python -m agent "Which of these props are worth it? Bankroll \$300" -s slip1.png -s slip2.png
```

Or call it from Python: `from agent import run_agent`.

| Tool | What it does |
|---|---|
| `run_parlay_engine` | Full pipeline on live moneylines: ratings model, +EV singles, correlated Monte Carlo parlays, exclusive portfolio |
| `get_slate` | Upcoming games, best line per side, devigged market probability |
| `evaluate_props` | EV, quarter-Kelly stake, and tier (LOCK / STRONG / VALUE / PASS) for individual bets |
| `price_custom_parlay` | Joint probability, payout, EV, and stake for one specific parlay |

The model defaults to `claude-opus-5-5` (override with `PARLAY_AGENT_MODEL`). It runs with server-side refusal fallbacks (`fallbacks="default"`).

Stakes are quarter-Kelly, capped at 5% of bankroll per bet.

**Known limitation:** the ratings model in `engine/ratings.py` uses placeholder data for six NBA teams and is trained on synthetic outcomes. Its probabilities are not validated, and the agent says so in every answer. Wire in real ratings and game history before trusting any edge.

## Engine

`engine/` is the quant core: odds math and devigging (`odds.py`), The Odds API client (`odds_api.py`), the ratings model (`ratings.py`), parlay pricing and selection (`parlays.py`), and the pipeline (`pipeline.py`).

```bash
.venv/bin/python -m pytest
```

## Dashboard

`src/` is the Vite/React screenshot dashboard. It calls the `supabase/functions/analyze-screenshots` edge function.

```bash
npm install && npm run dev
```
