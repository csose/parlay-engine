import base64
import mimetypes
import os
from collections.abc import Callable
from pathlib import Path

import anthropic

from .tools import ALL_TOOLS

MODEL = os.environ.get("PARLAY_AGENT_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """You are Parlay Engine, a sports betting analyst agent. You build and price parlays with the tools provided and report back in plain, direct language.

How to work:
- For live slates, run run_parlay_engine (use get_slate when you only need to look at games and prices).
- When the user attaches sportsbook screenshots, read every visible bet (player or team, prop, American odds, game), then price them with evaluate_props and price_custom_parlay. Never compute EV, probabilities, or stakes yourself; use the tool outputs.
- Only pass estimated_prob when you have a concrete reason for it. Call it your estimate and give the reason. Without one, the market no-vig price means negative EV, and you should say so instead of manufacturing an edge.
- Same-game legs: sportsbooks price same-game parlays below the product of the leg odds. Tell the user to check the book's actual SGP price before betting.

Honesty about the engine:
- Every run_parlay_engine result carries a caveat and coverage counts. Pass them on. Right now the ratings model covers only six NBA teams with placeholder data, so its "edges" are not validated. Never present them as proven value.
- If nothing is +EV, say that plainly. "No bet" is a valid answer.

Finish with the picks: legs, odds, win probability, EV, and stake in dollars. Stakes are quarter-Kelly capped at 5% of bankroll. Never suggest betting more than the tool's stake."""

MEDIA_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def image_block(path: str | Path) -> dict:
    media_type = mimetypes.guess_type(str(path))[0]
    if media_type not in MEDIA_TYPES:
        raise ValueError(f"{path}: unsupported image type {media_type!r}; use PNG, JPEG, GIF, or WebP.")
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def run_agent(
    request: str,
    screenshots: list[str | Path] = (),
    client: anthropic.Anthropic | None = None,
    on_tool_call: Callable[[str, dict], None] | None = None,
) -> str:
    client = client or anthropic.Anthropic()
    content = [image_block(p) for p in screenshots] + [{"type": "text", "text": request}]

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        max_iterations=20,
        messages=[{"role": "user", "content": content}],
    )

    final = None
    for message in runner:
        final = message
        if on_tool_call:
            for block in message.content:
                if block.type == "tool_use":
                    on_tool_call(block.name, block.input)

    if final is None:
        raise RuntimeError("The agent returned no response.")
    if final.stop_reason == "refusal":
        details = final.stop_details
        raise RuntimeError(f"Request declined ({details.category if details else 'unspecified'}): {details.explanation if details else ''}")
    if final.stop_reason == "max_tokens":
        raise RuntimeError("The agent hit its output limit before finishing.")
    return "\n".join(block.text for block in final.content if block.type == "text").strip()
