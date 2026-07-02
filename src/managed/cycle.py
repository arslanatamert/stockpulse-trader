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
from src.jury.jury import JuryDecision
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

# ── Invested-fraction floor ─────────────────────────────────────────────
# Idle cash loses to the index by default, so after each cycle any cash above
# this buffer is swept into a broad MSCI World ETF. The ETF acts as the
# portfolio's "default position": the jury never votes on it, and the buffer
# is what remains available for the jury's own stock picks.
_CASH_BUFFER_FRACTION = 0.10
BENCHMARK_ETF = os.getenv("BENCHMARK_ETF", "IWDA.AS")  # iShares Core MSCI World, EUR

# ── Trade cooldown (hysteresis) ─────────────────────────────────────────
# A stateless daily vote re-reaches yesterday's verdict and re-sizes it as a
# fraction of what's left, salami-slicing positions over days. After an
# executed trade a ticker is not re-reviewed until either the cooldown passes
# or the price moves enough to constitute new information.
_TRADE_COOLDOWN_DAYS = 5
_COOLDOWN_BREAK_MOVE_PCT = 5.0


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


def fetch_benchmark_price() -> float | None:
    """Current EUR price of the benchmark ETF (None if the quote fails)."""
    quote = get_eur_quotes([BENCHMARK_ETF]).get(BENCHMARK_ETF)
    return quote["price_eur"] if quote else None


def _cooldown_note(last_trade: dict | None, price_now: float, today: str) -> str | None:
    """Skip-note if the ticker traded recently and the price hasn't moved — else None."""
    if not last_trade:
        return None
    try:
        traded = date.fromisoformat(last_trade["timestamp"][:10])
        days_ago = (date.fromisoformat(today) - traded).days
    except ValueError:
        return None
    if days_ago >= _TRADE_COOLDOWN_DAYS:
        return None
    ref = last_trade["price"]
    move_pct = abs(price_now / ref - 1) * 100 if ref else 100.0
    if move_pct >= _COOLDOWN_BREAK_MOVE_PCT:
        return None
    return (
        f"Cooldown — {last_trade['action']} executed {days_ago}d ago, "
        f"price moved only {move_pct:.1f}% since."
    )


def _portfolio_context(positions: dict, ticker: str, price_eur: float,
                       last_trade: dict | None) -> dict:
    """Position + trade-history context handed to each agent's prompt."""
    pos = positions.get(ticker)
    ctx: dict = {"held_shares": 0}
    if pos and pos["avg_cost"]:
        ctx = {
            "held_shares": pos["shares"],
            "avg_cost": pos["avg_cost"],
            "unrealized_pnl_pct": (price_eur - pos["avg_cost"]) / pos["avg_cost"] * 100,
        }
    if last_trade:
        ctx["last_trade"] = {
            "action": last_trade["action"],
            "shares": last_trade["shares"],
            "price": last_trade["price"],
            "date": last_trade["timestamp"][:10],
            "confidence": last_trade["confidence"],
        }
    return ctx


def _sweep_excess_cash(portfolio, prices: dict[str, float], results: list[dict]) -> None:
    """Buy the benchmark ETF with any cash above the buffer fraction of equity."""
    price_map = dict(prices)
    if BENCHMARK_ETF not in price_map:
        bench = fetch_benchmark_price()
        if bench is not None:
            price_map[BENCHMARK_ETF] = bench
    bench_price = price_map.get(BENCHMARK_ETF)
    if not bench_price or bench_price <= 0:
        return

    cash = portfolio.get_cash()
    holdings_value = sum(
        p["shares"] * price_map.get(p["symbol"], p["avg_cost"])
        for p in portfolio.get_positions()
    )
    equity = cash + holdings_value
    excess = cash - _CASH_BUFFER_FRACTION * equity
    shares = int(excess // bench_price)
    if shares <= 0:
        return

    decision = JuryDecision(
        action="BUY",
        confidence=0.0,
        reasoning=(
            f"Automatic cash sweep: keep at least {1 - _CASH_BUFFER_FRACTION:.0%} of equity "
            f"invested. Cash above the {_CASH_BUFFER_FRACTION:.0%} buffer goes into the "
            f"MSCI World ETF ({BENCHMARK_ETF}) rather than dragging on returns."
        ),
        vote_summary="cash sweep (no jury vote)",
        votes={},
        dissenting_views=[],
    )
    entry = {"ticker": BENCHMARK_ETF, "action": "BUY", "shares": shares, "executed": False,
             "note": ""}
    try:
        portfolio.execute_trade(
            symbol=BENCHMARK_ETF, action="BUY", shares=shares,
            price=bench_price, decision=decision, verdicts=[],
        )
        entry["executed"] = True
        entry["note"] = f"Cash sweep → BUY {shares} @ €{bench_price:,.2f}"
    except ValueError as exc:
        entry["note"] = f"Cash sweep failed: {exc}"
    results.append(entry)


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
    """Capture an equity snapshot (EUR, current prices) plus the benchmark price."""
    try:
        held = [p["symbol"] for p in portfolio.get_positions()]
        symbols = sorted(set(held) | {BENCHMARK_ETF})
        snap_prices = {
            s: q["price_eur"]
            for s, q in get_eur_quotes(symbols).items()
            if q["price_eur"] is not None
        }
        portfolio.record_snapshot(snap_prices, benchmark_price=snap_prices.get(BENCHMARK_ETF))
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
    # The benchmark ETF is the cash-sweep reserve, not a jury matter.
    tickers = sorted((set(positions) | set(portfolio.get_watchlist())) - {BENCHMARK_ETF})

    results: list[dict] = []
    eur_price: dict[str, float] = {}
    for ticker in tickers:
        entry = {"ticker": ticker, "action": "HOLD", "shares": 0, "executed": False, "note": ""}
        try:
            market_data = get_stock_data(ticker)
            # Account in EUR: convert the native quote so cash/positions stay consistent.
            price = to_eur(market_data["price"], market_data.get("currency") or "EUR")
            eur_price[ticker] = price

            last_trade = portfolio.get_last_trade(ticker)
            skip = _cooldown_note(last_trade, price, today)
            if skip:
                entry["note"] = skip
                results.append(entry)
                continue
            context = _portfolio_context(positions, ticker, price, last_trade)

            # Run the 5 agents concurrently — each is an independent network call,
            # so a thread pool collapses ~5 sequential calls into ~1 round-trip.
            verdicts = []
            with ThreadPoolExecutor(max_workers=len(agents)) as pool:
                futures = {
                    pool.submit(AgentClass().analyze, ticker, market_data, context): name
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

    _sweep_excess_cash(portfolio, eur_price, results)
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
    # The benchmark ETF is the cash-sweep reserve, not a jury matter.
    tickers = sorted((set(positions) | set(portfolio.get_watchlist())) - {BENCHMARK_ETF})

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
        last_trade = portfolio.get_last_trade(ticker)
        skip = _cooldown_note(last_trade, eur_price[ticker], today)
        if skip:
            entries[ticker]["note"] = skip
            continue
        context = _portfolio_context(positions, ticker, eur_price[ticker], last_trade)
        for name, inst in instances:
            cid = f"r{counter}"
            counter += 1
            id_map[cid] = (ticker, name)
            requests.append(Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(
                    **inst.build_params(ticker, market_data, context)
                ),
            ))

    if not requests:  # nothing to review (no tickers, every fetch failed, or all cooling down)
        results = [entries[t] for t in tickers]
        _sweep_excess_cash(portfolio, eur_price, results)
        _record_snapshot(portfolio)
        portfolio.set_last_run(today)
        return {"skipped": False, "date": today, "via": "batch", "results": results}

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

    results = [entries[t] for t in tickers]
    _sweep_excess_cash(portfolio, eur_price, results)
    _record_snapshot(portfolio)
    portfolio.set_last_run(today)
    return {"skipped": False, "date": today, "via": "batch", "results": results}
