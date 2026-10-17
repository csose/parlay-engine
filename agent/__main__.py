import argparse
import json
import sys

import anthropic

from .runner import run_agent


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agent", description="Parlay Engine agent")
    parser.add_argument("request", help='What you want, e.g. "Build tonight\'s NBA parlays with a $500 bankroll"')
    parser.add_argument("-s", "--screenshot", action="append", default=[], help="Sportsbook screenshot to analyze (repeatable)")
    args = parser.parse_args()

    def show(name: str, args: dict) -> None:
        print(f"→ {name} {json.dumps(args)[:200]}", file=sys.stderr)

    try:
        print(run_agent(args.request, args.screenshot, on_tool_call=show))
    except (anthropic.APIError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
