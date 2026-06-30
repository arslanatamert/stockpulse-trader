"""Daily review engine for the auto-managed EUR portfolio.

Each run the jury reviews every current holding plus the watchlist, and a
deterministic confidence-based rule turns each verdict into a share count. No
extra LLM call is made for sizing — the math is fully explainable.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from src.agents.buffett import BuffettAgent
from src.agents.dalio import DalioAgent
from src.agents.soros import SorosAgent
from src.agents.lynch import LynchAgent
from src.agents.simons import SimonsAgent
from src.jury import jury as jury_module
from src.market.data import get_stock_data, get_eur_quotes, to_eur
from src.portfolio.managed import ManagedPortfolio

# Single source of truth for the jury line-up (also imported by app.py).
AGENTS = [
    ("Warren Buffett", BuffettAgent),
    ("Ray Dalio",      DalioAgent),
    ("George Soros",   SorosAgent),
    ("Peter Lynch",    LynchAgent),
    ("Jim Simons",     SimonsAgent),
]

# Buys never deploy more than this share of free cash in one go.
_MAX_BUY_FRACTION = 0.40


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def size_trade(action: str, confidence: float, cash: float, price: float, held_shares: int) -> int:
    """Translate a jury verdict into a whole-share quantity (0 = no trade).

    BUY  — fraction of free cash scales with conviction (~55%→10%, 80%→30%, cap 40%).
    SELL — fraction of the held position scales with conviction (~55%→25%, 80%→75%).
    """
    if action == "BUY":
        if price <= 0 or cash <= 0:
            return 0
        frac = _clamp(0.008 * confidence - 0.34, 0.0, _MAX_BUY_FRACTION)
        budget = frac * cash
        return int(math.floor(budget / price))

    if action == "SELL":
        if held_shares <= 0:
            return 0
        frac = _clamp(0.02 * confidence - 0.85, 0.0, 1.0)
        return int(round(frac * held_shares))

    return 0  # HOLD


def run_daily_cycle(
    portfolio: ManagedPortfolio,
    agents=AGENTS,
    *,
    force: bool = False,
    today: str | None = None,
) -> dict:
    """Run one review pass over holdings + watchlist.

    Returns a dict with ``skipped`` plus a per-ticker ``results`` list describing
    what the jury decided and whether a trade was executed.
    """
    today = today or date.today().isoformat()

    if not force and portfolio.get_last_run() == today:
        return {"skipped": True, "reason": "already_ran_today", "date": today, "results": []}

    positions = {p["symbol"]: p for p in portfolio.get_positions()}
    tickers = sorted(set(positions) | set(portfolio.get_watchlist()))

    results: list[dict] = []
    for ticker in tickers:
        entry = {"ticker": ticker, "action": "HOLD", "shares": 0, "executed": False, "note": ""}
        try:
            market_data = get_stock_data(ticker)
            # Account in EUR: convert the native quote so cash/positions stay consistent.
            price = to_eur(market_data["price"], market_data.get("currency") or "EUR")

            # Run the 5 agents concurrently — each is an independent network call,
            # so a thread pool collapses ~5 sequential calls into ~1 round-trip.
            verdicts = []
            with ThreadPoolExecutor(max_workers=len(agents)) as pool:
                futures = {
                    pool.submit(AgentClass().analyze, ticker, market_data): name
                    for name, AgentClass in agents
                }
                for future, name in futures.items():
                    try:
                        verdicts.append(future.result())
                    except Exception as exc:  # one agent failing shouldn't sink the ticker
                        entry["note"] = f"{name} errored: {exc}"
            if not verdicts:
                entry["note"] = "All agents failed."
                results.append(entry)
                continue

            decision = jury_module.deliberate(verdicts)
            held = positions.get(ticker, {}).get("shares", 0)
            cash = portfolio.get_cash()
            shares = size_trade(decision.action, decision.confidence, cash, price, held)

            entry.update(
                action=decision.action,
                confidence=round(decision.confidence),
                price=price,
                shares=shares,
                vote_summary=decision.vote_summary,
            )

            if decision.action in ("BUY", "SELL") and shares > 0:
                try:
                    portfolio.execute_trade(
                        symbol=ticker,
                        action=decision.action,
                        shares=shares,
                        price=price,
                        decision=decision,
                        verdicts=verdicts,
                    )
                    entry["executed"] = True
                    entry["note"] = f"{decision.action} {shares} @ €{price:,.2f}"
                except ValueError as exc:  # e.g. not enough cash / shares
                    entry["note"] = str(exc)
            elif decision.action in ("BUY", "SELL"):
                entry["note"] = "Sized to 0 shares — skipped."
            else:
                entry["note"] = "Jury holds."

        except Exception as exc:
            entry["note"] = f"Data error: {exc}"

        results.append(entry)

    # Capture an equity snapshot (EUR, at current prices) for the performance chart.
    try:
        held = [p["symbol"] for p in portfolio.get_positions()]
        snap_prices = {
            s: q["price_eur"] for s, q in get_eur_quotes(held).items() if q["price_eur"] is not None
        } if held else {}
        portfolio.record_snapshot(snap_prices)
    except Exception:
        pass

    portfolio.set_last_run(today)
    return {"skipped": False, "date": today, "results": results}
