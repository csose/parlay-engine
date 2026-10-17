KELLY_FRACTION = 0.25
MAX_EXPOSURE = 0.05
PROP_VIG_FACTOR = 0.9525


def american_to_decimal(odds: float) -> float:
    if odds == 0:
        return 1.0
    return (1 + odds / 100) if odds > 0 else (1 + 100 / abs(odds))


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        return 0.0
    return (decimal - 1) * 100 if decimal >= 2.0 else -100 / (decimal - 1)


def implied_prob(odds: float) -> float:
    return 1 / american_to_decimal(odds)


def devig_two_way(p1: float, p2: float) -> tuple[float, float]:
    overround = p1 + p2
    return (p1 / overround, p2 / overround) if overround > 0 else (p1, p2)


def kelly(prob: float, decimal: float) -> float:
    """Fractional Kelly stake as a share of bankroll, capped at MAX_EXPOSURE."""
    b = decimal - 1
    if b <= 0:
        return 0.0
    full = (prob * b - (1 - prob)) / b
    return min(max(full * KELLY_FRACTION, 0.0), MAX_EXPOSURE)
