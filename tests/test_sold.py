"""Regression tests for the realized-P&L "Sold" derivation.

Exercises :func:`src.portfolio._sold.derive_sold_positions` end-to-end through
both portfolios (they share the helper), covering the avg-cost replay math that
powers the getquin-style Total sold / Partially sold widget.

Runs under pytest (``pytest tests``) or standalone (``python tests/test_sold.py``);
no network access is required — everything works against throwaway SQLite files.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio._sold import days_between
from src.portfolio.managed import ManagedPortfolio
from src.portfolio.sandbox import SandboxPortfolio

# execute_trade only reads these attrs off the decision; verdicts can be empty.
_DECISION = SimpleNamespace(action="SELL", confidence=80.0, reasoning="r", vote_summary="v")


def _fresh(Portfolio):
    """A portfolio on its own temp DB, topped up with plenty of cash for buys."""
    import sqlite3

    path = tempfile.mktemp(suffix=".db")
    p = Portfolio(db_path=path)
    conn = sqlite3.connect(path)
    conn.execute("UPDATE cash SET balance = 1000000 WHERE id = 1")
    conn.commit()
    conn.close()
    return p, path


def _buy(p, sym, sh, px):
    p.execute_trade(sym, "BUY", sh, px, _DECISION, [])


def _sell(p, sym, sh, px):
    p.execute_trade(sym, "SELL", sh, px, _DECISION, [])


def test_full_round_trips_gain_and_loss():
    p, path = _fresh(ManagedPortfolio)
    try:
        # avg cost blends to 110 across two buys, then a full exit at 150
        _buy(p, "NVDA", 10, 100.0)
        _buy(p, "NVDA", 10, 120.0)
        _sell(p, "NVDA", 20, 150.0)
        # a losing full exit
        _buy(p, "WCLD", 5, 200.0)
        _sell(p, "WCLD", 5, 180.0)

        out = p.get_sold_positions()
        total = {e["symbol"]: e for e in out["total"]}

        assert set(total) == {"NVDA", "WCLD"}
        assert not out["partial"]

        assert total["NVDA"]["realized_gain"] == 800.0            # 20 * (150 - 110)
        assert abs(total["NVDA"]["realized_gain_pct"] - 36.36) < 0.05
        assert total["WCLD"]["realized_gain"] == -100.0           # 5 * (180 - 200)
        assert total["WCLD"]["realized_gain_pct"] == -10.0

        # worst gain sorts first, matching the getquin Price-gain default
        assert [e["symbol"] for e in out["total"]][0] == "WCLD"
    finally:
        os.remove(path)


def test_partial_sell_reports_sold_pct():
    p, path = _fresh(ManagedPortfolio)
    try:
        _buy(p, "TRX", 100, 10.0)
        _sell(p, "TRX", 20, 15.0)  # trim 20%, still holding 80

        out = p.get_sold_positions()
        assert not out["total"]
        (entry,) = out["partial"]
        assert entry["symbol"] == "TRX"
        assert entry["sold_pct"] == 20.0
        assert entry["realized_gain"] == 100.0       # 20 * (15 - 10)
        assert entry["realized_gain_pct"] == 50.0
    finally:
        os.remove(path)


def test_seeded_holding_uses_stored_cost_basis():
    """Seeded lots leave no BUY row; basis must come from the positions row."""
    p, path = _fresh(ManagedPortfolio)
    try:
        # partially sold seed
        p.seed_holding("ASTS", 50, 40.0)
        _sell(p, "ASTS", 10, 60.0)      # hold 40 -> partial
        # fully sold seed
        p.seed_holding("KAP", 4, 10.0)
        _sell(p, "KAP", 4, 12.0)        # hold 0 -> total

        out = p.get_sold_positions()
        partial = {e["symbol"]: e for e in out["partial"]}
        total = {e["symbol"]: e for e in out["total"]}

        assert partial["ASTS"]["realized_gain"] == 200.0   # 10 * (60 - 40)
        assert partial["ASTS"]["sold_pct"] == 20.0         # 10 / 50
        assert total["KAP"]["realized_gain"] == 8.0        # 4 * (12 - 10)
    finally:
        os.remove(path)


def test_never_sold_holdings_are_excluded():
    p, path = _fresh(ManagedPortfolio)
    try:
        _buy(p, "HOLDME", 3, 50.0)
        out = p.get_sold_positions()
        assert out["total"] == [] and out["partial"] == []
    finally:
        os.remove(path)


def test_sandbox_shares_the_same_derivation():
    """SandboxPortfolio (no seeding) must produce identical results for BUY/SELL."""
    p, path = _fresh(SandboxPortfolio)
    try:
        _buy(p, "AAPL", 10, 100.0)
        _sell(p, "AAPL", 10, 90.0)     # full exit, loss
        _buy(p, "MSFT", 10, 50.0)
        _sell(p, "MSFT", 4, 80.0)      # partial

        out = p.get_sold_positions()
        total = {e["symbol"]: e for e in out["total"]}
        partial = {e["symbol"]: e for e in out["partial"]}

        assert total["AAPL"]["realized_gain"] == -100.0
        assert partial["MSFT"]["sold_pct"] == 40.0
        assert partial["MSFT"]["realized_gain"] == 120.0   # 4 * (80 - 50)
    finally:
        os.remove(path)


def test_days_between_handles_missing_and_bad_input():
    assert days_between("2021-02-09T13:18:00", "2026-05-04T18:02:00") > 1900
    assert days_between(None, "2026-01-01T00:00:00") == 0
    assert days_between("not-a-date", "2026-01-01T00:00:00") == 0
    # never negative even if the sale somehow predates acquisition
    assert days_between("2026-01-10T00:00:00", "2026-01-01T00:00:00") == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
