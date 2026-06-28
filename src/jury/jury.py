from dataclasses import dataclass

from src.agents.base_agent import AgentVerdict

SUPERMAJORITY = 3  # at least 3/5 agents must agree to trigger a trade


@dataclass
class JuryDecision:
    action: str         # BUY | SELL | HOLD
    confidence: float   # weighted average confidence of the winning camp
    reasoning: str
    vote_summary: str   # e.g. "BUY: 3, SELL: 1, HOLD: 1"
    votes: dict[str, int]
    dissenting_views: list[str]


def deliberate(verdicts: list[AgentVerdict]) -> JuryDecision:
    tally: dict[str, list[int]] = {"BUY": [], "SELL": [], "HOLD": []}

    for v in verdicts:
        tally[v.action].append(v.confidence)

    counts = {k: len(v) for k, v in tally.items()}
    leading = max(counts, key=counts.get)
    max_votes = counts[leading]

    if max_votes < SUPERMAJORITY:
        action = "HOLD"
        avg_conf = _avg(tally["HOLD"] or tally["BUY"] or tally["SELL"])
        reasoning = (
            f"The jury reached no supermajority. "
            f"Votes: BUY {counts['BUY']}, SELL {counts['SELL']}, HOLD {counts['HOLD']}. "
            f"Without at least {SUPERMAJORITY}/5 agents in agreement, the prudent move is to hold."
        )
    else:
        action = leading
        avg_conf = _avg(tally[action])
        verb = {"BUY": "buy", "SELL": "sell", "HOLD": "hold"}[action]
        reasoning = (
            f"{max_votes} of 5 jury members recommend to {verb} "
            f"with an average conviction of {avg_conf:.0f}%. "
            + _dissent_summary(action, verdicts)
        )

    vote_summary = f"BUY: {counts['BUY']}  SELL: {counts['SELL']}  HOLD: {counts['HOLD']}"
    dissenting = [
        f"{v.agent_name} ({v.action}, {v.confidence}%): {v.reasoning}"
        for v in verdicts
        if v.action != action
    ]

    return JuryDecision(
        action=action,
        confidence=avg_conf,
        reasoning=reasoning,
        vote_summary=vote_summary,
        votes=counts,
        dissenting_views=dissenting,
    )


def _avg(values: list[int]) -> float:
    return sum(values) / len(values) if values else 50.0


def _dissent_summary(action: str, verdicts: list[AgentVerdict]) -> str:
    dissenters = [v.agent_name for v in verdicts if v.action != action]
    if not dissenters:
        return "Unanimous verdict."
    return f"Dissenting: {', '.join(dissenters)}."
