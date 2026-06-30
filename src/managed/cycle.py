"""Daily review engine for the auto-managed EUR portfolio.

Each run the jury reviews every current holding plus the watchlist, and a
deterministic confidence-based rule turns each verdict into a share count. No
extra LLM call is made for sizing — the math is fully explainable.
"""

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from src.agents.base_agent import BaseAgent
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


def _apply_verdicts(portfolio, ticker, verdicts, price, held, entry) -> None:
    """Deliberate, size, and execute for one ticker — shared by sync and batch paths."""
    decision = jury_module.deliberate(verdicts)
    shares = size_trade(decision.action, decision.confidence, portfolio.get_cash(), price, held)
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
                symbol=ticker, action=decision.action, shares=shares,
                price=price, decision=decision, verdicts=verdicts,
            )
            entry["executed"] = True
            entry["note"] = f"{decision.action} {shares} @ €{price:,.2f}"
        except ValueError as exc:  # e.g. not enough cash / shares
            entry["note"] = str(exc)
    elif decision.action in ("BUY", "SELL"):
        entry["note"] = "Sized to 0 shares — skipped."
    else:
        entry["note"] = "Jury holds."


def _record_snapshot(portfolio) -> None:
    """Capture an equity snapshot (EUR, current prices) for the performance chart."""
    try:
        held = [p["symbol"] for p in portfolio.get_positions()]
        snap_prices = {
            s: q["price_eur"] for s, q in get_eur_quotes(held).items() if q["price_eur"] is not None
        } if held else {}
        portfolio.record_snapshot(snap_prices)
    except Exception:
        pass


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

            held = positions.get(ticker, {}).get("shares", 0)
            _apply_verdicts(portfolio, ticker, verdicts, price, held, entry)

        except Exception as exc:
            entry["note"] = f"Data error: {exc}"

        results.append(entry)

    _record_snapshot(portfolio)
    portfolio.set_last_run(today)
    return {"skipped": False, "date": today, "results": results}


def run_daily_cycle_batched(
    portfolio,
    agents=AGENTS,
    *,
    force: bool = False,
    today: str | None = None,
    poll_interval: float = 15.0,
    max_wait: float = 3600.0,
) -> dict:
    """Same review as run_daily_cycle, but all agent calls go through the Anthropic
    Batch API (50% cheaper, asynchronous). Suited to the background daily run where
    latency doesn't matter — not the interactive UI.
    """
    today = today or date.today().isoformat()
    if not force and portfolio.get_last_run() == today:
        return {"skipped": True, "reason": "already_ran_today", "date": today, "results": []}

    positions = {p["symbol"]: p for p in portfolio.get_positions()}
    tickers = sorted(set(positions) | set(portfolio.get_watchlist()))

    entries = {
        t: {"ticker": t, "action": "HOLD", "shares": 0, "executed": False, "note": ""}
        for t in tickers
    }
    instances = [(name, AgentClass()) for name, AgentClass in agents]

    # ── Build one request per (ticker, agent); map custom_id back to both ──
    eur_price: dict[str, float] = {}
    requests: list[Request] = []
    id_map: dict[str, tuple[str, str]] = {}
    counter = 0
    for ticker in tickers:
        try:
            market_data = get_stock_data(ticker)
            eur_price[ticker] = to_eur(market_data["price"], market_data.get("currency") or "EUR")
        except Exception as exc:
            entries[ticker]["note"] = f"Data error: {exc}"
            continue
        for name, inst in instances:
            cid = f"r{counter}"
            counter += 1
            id_map[cid] = (ticker, name)
            requests.append(Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(**inst.build_params(ticker, market_data)),
            ))

    if not requests:  # nothing to review (no tickers, or every fetch failed)
        portfolio.set_last_run(today)
        return {"skipped": False, "date": today, "via": "batch", "results": [entries[t] for t in tickers]}

    # ── Submit the batch and poll until it ends ──
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    batch = client.messages.batches.create(requests=requests)
    waited = 0.0
    while True:
        status = client.messages.batches.retrieve(batch.id).processing_status
        if status == "ended":
            break
        if waited >= max_wait:
            raise TimeoutError(f"Batch {batch.id} still '{status}' after {max_wait:.0f}s")
        time.sleep(poll_interval)
        waited += poll_interval

    # ── Collect verdicts per ticker ──
    verdicts: dict[str, list] = {t: [] for t in tickers}
    for result in client.messages.batches.results(batch.id):
        ticker, name = id_map.get(result.custom_id, (None, None))
        if ticker is None:
            continue
        if result.result.type == "succeeded":
            try:
                text = next(b.text for b in result.result.message.content if b.type == "text")
                verdicts[ticker].append(BaseAgent.parse(name, text))
            except Exception as exc:
                entries[ticker]["note"] = f"{name} parse error: {exc}"
        else:
            entries[ticker]["note"] = f"{name} {result.result.type}"

    # ── Decide + trade per ticker ──
    for ticker in tickers:
        if ticker not in eur_price:  # data error already recorded
            continue
        vs = verdicts[ticker]
        if not vs:
            entries[ticker]["note"] = entries[ticker]["note"] or "All agents failed."
            continue
        held = positions.get(ticker, {}).get("shares", 0)
        _apply_verdicts(portfolio, ticker, vs, eur_price[ticker], held, entries[ticker])

    _record_snapshot(portfolio)
    portfolio.set_last_run(today)
    return {"skipped": False, "date": today, "via": "batch", "results": [entries[t] for t in tickers]}
