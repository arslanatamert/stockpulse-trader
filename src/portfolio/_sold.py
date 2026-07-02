"""Shared realized-P&L derivation for the paper portfolios.

Both :class:`SandboxPortfolio` and :class:`ManagedPortfolio` store the same
``positions`` / ``transactions`` shape, so the logic that turns a trade history
into "sold" buckets (getquin-style Total sold / Partially sold) lives here as
pure functions and is reused by each portfolio's thin ``get_sold_positions``
wrapper.
"""

from datetime import datetime


def days_between(start_ts: str | None, end_ts: str | None) -> int:
    """Whole days between two ISO timestamps; 0 if either is missing/unparseable."""
    if not start_ts or not end_ts:
        return 0
    try:
        start = datetime.fromisoformat(start_ts)
        end = datetime.fromisoformat(end_ts)
    except ValueError:
        return 0
    return max((end - start).days, 0)


def derive_sold_positions(pos_rows, tx_rows) -> dict:
    """Turn a portfolio's positions + trade history into sold buckets.

    Replays each symbol's acquisitions (BUY) and disposals (SELL) in
    chronological order using running average cost — mirroring the avg-cost
    math in ``execute_trade`` — to compute realized price gains.

    Pre-owned holdings seeded without a transaction (``ManagedPortfolio``'s
    ``seed_holding``) leave a positions row but no BUY, so any shares present
    that recorded BUYs can't explain are inferred as a seed lot valued at the
    position's stored average cost.

    Parameters
    ----------
    pos_rows:
        Iterable of ``(symbol, shares, avg_cost)`` — the live positions table,
        including rows whose shares have fallen to 0 (fully-closed positions).
    tx_rows:
        Iterable of ``(symbol, action, shares, price, timestamp)`` restricted to
        BUY/SELL and ordered chronologically (oldest first).

    Returns
    -------
    dict
        ``{"total": [...], "partial": [...]}`` — each a list of entries sorted
        by realized gain (worst first, matching getquin's Price-gain default):

        * ``total``   — fully closed (0 shares held); carries ``holding_days``.
        * ``partial`` — still open but trimmed; carries ``sold_pct``.
    """
    held = {r[0]: r[1] for r in pos_rows}
    pos_avg = {r[0]: r[2] for r in pos_rows}

    by_symbol: dict[str, list] = {}
    for sym, action, shares, price, ts in tx_rows:
        by_symbol.setdefault(sym, []).append((action, shares, price, ts))

    total, partial = [], []
    for sym, txns in by_symbol.items():
        bought = sum(s for a, s, _, _ in txns if a == "BUY")
        sold = sum(s for a, s, _, _ in txns if a == "SELL")
        if sold == 0:
            continue  # never sold — still a plain holding

        cur_held = held.get(sym, 0)
        # Shares present that recorded BUYs can't explain came from a seed.
        seed_shares = max(cur_held + sold - bought, 0)

        shares = seed_shares
        avg_cost = pos_avg.get(sym, 0.0) if seed_shares else 0.0
        acquired = seed_shares
        first_ts = txns[0][3]
        last_sell_ts = None
        realized = 0.0
        sold_basis = 0.0

        for action, s, price, ts in txns:
            if action == "BUY":
                new_shares = shares + s
                avg_cost = (shares * avg_cost + s * price) / new_shares if new_shares else 0.0
                shares = new_shares
                acquired += s
            else:  # SELL
                realized += s * (price - avg_cost)
                sold_basis += s * avg_cost
                shares = max(shares - s, 0)
                last_sell_ts = ts

        realized_pct = (realized / sold_basis * 100) if sold_basis else 0.0
        entry = {
            "symbol": sym,
            "sold_shares": sold,
            "acquired_shares": acquired,
            "realized_gain": round(realized, 2),
            "realized_gain_pct": round(realized_pct, 2),
            "first_acquired": first_ts,
            "last_sold": last_sell_ts,
        }
        if cur_held > 0:
            entry["sold_pct"] = round(min(sold / acquired * 100, 100), 1) if acquired else 0.0
            partial.append(entry)
        else:
            entry["holding_days"] = days_between(first_ts, last_sell_ts)
            total.append(entry)

    total.sort(key=lambda e: e["realized_gain"])
    partial.sort(key=lambda e: e["realized_gain"])
    return {"total": total, "partial": partial}
