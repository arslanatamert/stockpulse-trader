"""EUR auto-managed portfolio.

A second, independent paper portfolio that the jury manages on its own.
Seeded with the user's real holdings (kept at their cost basis) plus €1000 of
fresh cash. Lives in its own SQLite file so it never collides with the USD
``SandboxPortfolio``. Mirrors that class's avg-cost math and transaction schema.
"""

import json
import os
import sqlite3
from datetime import datetime

from src.agents.base_agent import AgentVerdict
from src.jury.jury import JuryDecision
from src.portfolio._sold import derive_sold_positions

_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "managed_portfolio.db")
_INITIAL_CASH = 1000.0  # EUR — fresh spending cash on top of seeded holdings

_DDL = """
CREATE TABLE IF NOT EXISTS cash (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    balance REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol   TEXT PRIMARY KEY,
    shares   INTEGER NOT NULL DEFAULT 0,
    avg_cost REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS watchlist (
    symbol TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    total_value    REAL NOT NULL,
    cash           REAL NOT NULL,
    holdings_value REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS watchlist_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    basket_value REAL NOT NULL,
    item_count   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    action              TEXT NOT NULL,
    shares              INTEGER NOT NULL,
    price               REAL NOT NULL,
    total_value         REAL NOT NULL,
    jury_action         TEXT,
    jury_confidence     REAL,
    jury_reasoning      TEXT,
    vote_summary        TEXT,
    agent_names         TEXT,
    agent_actions       TEXT,
    agent_confidences   TEXT,
    agent_reasonings    TEXT,
    agent_key_factors   TEXT
);
"""


class ManagedPortfolio:
    def __init__(self, db_path: str = _DEFAULT_DB):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db = db_path
        conn = self._connect()
        conn.executescript(_DDL)
        row = conn.execute("SELECT COUNT(*) FROM cash").fetchone()
        if row[0] == 0:
            conn.execute("INSERT INTO cash (id, balance) VALUES (1, ?)", (_INITIAL_CASH,))
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('initial_capital', ?)",
                (str(_INITIAL_CASH),),
            )
            conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Cash / positions
    # ------------------------------------------------------------------

    def get_cash(self) -> float:
        conn = self._connect()
        val = conn.execute("SELECT balance FROM cash WHERE id = 1").fetchone()[0]
        conn.close()
        return val

    def get_positions(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT symbol, shares, avg_cost FROM positions WHERE shares > 0 ORDER BY symbol"
        ).fetchall()
        conn.close()
        return [{"symbol": r[0], "shares": r[1], "avg_cost": r[2]} for r in rows]

    def seed_holding(self, symbol: str, shares: int, avg_cost: float) -> None:
        """Add a pre-owned position WITHOUT spending cash.

        The position's cost basis (shares * avg_cost) is added to the P&L
        baseline (``initial_capital``) so unrealized gains are measured against
        what was actually paid. Re-seeding the same symbol averages it in.
        """
        symbol = symbol.strip().upper()
        if shares <= 0 or avg_cost <= 0:
            raise ValueError("Shares and average cost must be positive.")
        cost = shares * avg_cost
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT shares, avg_cost FROM positions WHERE symbol = ?", (symbol,)
            ).fetchone()
            if existing:
                new_shares = existing[0] + shares
                new_avg = (existing[0] * existing[1] + cost) / new_shares
                conn.execute(
                    "UPDATE positions SET shares = ?, avg_cost = ? WHERE symbol = ?",
                    (new_shares, new_avg, symbol),
                )
            else:
                conn.execute(
                    "INSERT INTO positions (symbol, shares, avg_cost) VALUES (?, ?, ?)",
                    (symbol, shares, avg_cost),
                )
            baseline = self._get_meta(conn, "initial_capital", _INITIAL_CASH)
            self._set_meta(conn, "initial_capital", baseline + cost)
            conn.commit()
        finally:
            conn.close()

    def execute_trade(
        self,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        decision: JuryDecision,
        verdicts: list[AgentVerdict],
    ) -> dict:
        total = shares * price
        conn = self._connect()
        try:
            if action == "BUY":
                cash = conn.execute("SELECT balance FROM cash WHERE id = 1").fetchone()[0]
                if total > cash:
                    raise ValueError(
                        f"Insufficient funds. Need €{total:,.2f}, available €{cash:,.2f}."
                    )
                conn.execute("UPDATE cash SET balance = balance - ? WHERE id = 1", (total,))
                existing = conn.execute(
                    "SELECT shares, avg_cost FROM positions WHERE symbol = ?", (symbol,)
                ).fetchone()
                if existing:
                    new_shares = existing[0] + shares
                    new_avg = (existing[0] * existing[1] + total) / new_shares
                    conn.execute(
                        "UPDATE positions SET shares = ?, avg_cost = ? WHERE symbol = ?",
                        (new_shares, new_avg, symbol),
                    )
                else:
                    conn.execute(
                        "INSERT INTO positions (symbol, shares, avg_cost) VALUES (?, ?, ?)",
                        (symbol, shares, price),
                    )

            elif action == "SELL":
                existing = conn.execute(
                    "SELECT shares FROM positions WHERE symbol = ?", (symbol,)
                ).fetchone()
                if not existing or existing[0] < shares:
                    held = existing[0] if existing else 0
                    raise ValueError(
                        f"Cannot sell {shares} shares of {symbol} — only {held} held."
                    )
                conn.execute("UPDATE cash SET balance = balance + ? WHERE id = 1", (total,))
                conn.execute(
                    "UPDATE positions SET shares = shares - ? WHERE symbol = ?",
                    (shares, symbol),
                )
            else:
                raise ValueError(f"Unknown action: {action}")

            conn.execute(
                """INSERT INTO transactions
                   (timestamp, symbol, action, shares, price, total_value,
                    jury_action, jury_confidence, jury_reasoning, vote_summary,
                    agent_names, agent_actions, agent_confidences,
                    agent_reasonings, agent_key_factors)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    symbol,
                    action,
                    shares,
                    price,
                    total,
                    decision.action,
                    decision.confidence,
                    decision.reasoning,
                    decision.vote_summary,
                    json.dumps([v.agent_name for v in verdicts]),
                    json.dumps([v.action for v in verdicts]),
                    json.dumps([v.confidence for v in verdicts]),
                    json.dumps([v.reasoning for v in verdicts]),
                    json.dumps([v.key_factors for v in verdicts]),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return {"symbol": symbol, "action": action, "shares": shares, "price": price, "total": total}

    def get_summary(self, current_prices: dict[str, float] | None = None) -> dict:
        cash = self.get_cash()
        positions = self.get_positions()
        cp = current_prices or {}
        baseline = self._read_meta("initial_capital", _INITIAL_CASH)

        enriched = []
        holdings_value = 0.0
        for pos in positions:
            cur = cp.get(pos["symbol"], pos["avg_cost"])
            val = pos["shares"] * cur
            holdings_value += val
            pnl_pct = (cur - pos["avg_cost"]) / pos["avg_cost"] * 100
            enriched.append({**pos, "current_price": cur, "value": val, "pnl_pct": round(pnl_pct, 2)})

        total = cash + holdings_value
        overall_pnl = total - baseline
        overall_pnl_pct = overall_pnl / baseline * 100 if baseline else 0.0

        return {
            "cash": cash,
            "holdings_value": holdings_value,
            "total_value": total,
            "pnl": overall_pnl,
            "pnl_pct": round(overall_pnl_pct, 2),
            "positions": enriched,
            "initial_cash": _INITIAL_CASH,
            "initial_capital": baseline,
        }

    def get_sold_positions(self) -> dict:
        """Derive fully-sold and partially-sold positions from trade history.

        Replays each symbol's acquisitions (BUY) and disposals (SELL) in
        chronological order using running average cost — mirroring
        ``execute_trade``'s avg-cost math — to compute realized price gains.

        Pre-owned holdings added via :meth:`seed_holding` never create a
        transaction, so their lot is inferred from the live positions row:
        any shares present that BUY transactions don't account for are treated
        as a seed lot valued at the position's stored average cost.

        Returns a dict with two buckets, each a list sorted by realized gain
        (worst first, matching the getquin "Price gain" default sort):

        * ``total``   — positions now fully closed (0 shares held). Each entry
          carries ``holding_days`` (first acquisition → last sale).
        * ``partial`` — positions still open but with some shares sold. Each
          entry carries ``sold_pct`` (share of the acquired lot disposed of).
        """
        conn = self._connect()
        pos_rows = conn.execute("SELECT symbol, shares, avg_cost FROM positions").fetchall()
        tx_rows = conn.execute(
            "SELECT symbol, action, shares, price, timestamp FROM transactions "
            "WHERE action IN ('BUY', 'SELL') ORDER BY id ASC"
        ).fetchall()
        conn.close()
        return derive_sold_positions(pos_rows, tx_rows)

    def get_transactions(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            """SELECT id, timestamp, symbol, action, shares, price, total_value,
                      jury_action, jury_confidence, jury_reasoning, vote_summary,
                      agent_names, agent_actions, agent_confidences,
                      agent_reasonings, agent_key_factors
               FROM transactions ORDER BY id DESC"""
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "timestamp": r[1],
                "symbol": r[2],
                "action": r[3],
                "shares": r[4],
                "price": r[5],
                "total_value": r[6],
                "jury_action": r[7],
                "jury_confidence": r[8],
                "jury_reasoning": r[9],
                "vote_summary": r[10],
                "agent_names": json.loads(r[11] or "[]"),
                "agent_actions": json.loads(r[12] or "[]"),
                "agent_confidences": json.loads(r[13] or "[]"),
                "agent_reasonings": json.loads(r[14] or "[]"),
                "agent_key_factors": json.loads(r[15] or "[]"),
            })
        return result

    # ------------------------------------------------------------------
    # Performance snapshots (equity curve)
    # ------------------------------------------------------------------

    def record_snapshot(self, current_prices: dict[str, float] | None = None) -> dict:
        """Capture total value / cash / holdings at this moment for the equity curve."""
        s = self.get_summary(current_prices)
        conn = self._connect()
        conn.execute(
            "INSERT INTO snapshots (timestamp, total_value, cash, holdings_value) VALUES (?,?,?,?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                s["total_value"], s["cash"], s["holdings_value"],
            ),
        )
        conn.commit()
        conn.close()
        return s

    def get_snapshots(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT timestamp, total_value, cash, holdings_value FROM snapshots ORDER BY id"
        ).fetchall()
        conn.close()
        return [
            {"timestamp": r[0], "total_value": r[1], "cash": r[2], "holdings_value": r[3]}
            for r in rows
        ]

    def record_watchlist_snapshot(self, basket_value: float, item_count: int) -> None:
        """Capture the combined value of the watchlist basket (separate from holdings)."""
        conn = self._connect()
        conn.execute(
            "INSERT INTO watchlist_snapshots (timestamp, basket_value, item_count) VALUES (?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), basket_value, item_count),
        )
        conn.commit()
        conn.close()

    def get_watchlist_snapshots(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT timestamp, basket_value, item_count FROM watchlist_snapshots ORDER BY id"
        ).fetchall()
        conn.close()
        return [{"timestamp": r[0], "basket_value": r[1], "item_count": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    def get_watchlist(self) -> list[str]:
        conn = self._connect()
        rows = conn.execute("SELECT symbol FROM watchlist ORDER BY symbol").fetchall()
        conn.close()
        return [r[0] for r in rows]

    def add_watch(self, symbol: str) -> None:
        symbol = symbol.strip().upper()
        if not symbol:
            return
        conn = self._connect()
        conn.execute("INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)", (symbol,))
        conn.commit()
        conn.close()

    def remove_watch(self, symbol: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.strip().upper(),))
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Daily-run bookkeeping
    # ------------------------------------------------------------------

    def get_last_run(self) -> str | None:
        return self._read_meta("last_run_date", None)

    def set_last_run(self, date_str: str) -> None:
        conn = self._connect()
        self._set_meta(conn, "last_run_date", date_str)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------

    def reset(self):
        conn = self._connect()
        conn.execute("UPDATE cash SET balance = ? WHERE id = 1", (_INITIAL_CASH,))
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM transactions")
        conn.execute("DELETE FROM watchlist")
        conn.execute("DELETE FROM snapshots")
        conn.execute("DELETE FROM watchlist_snapshots")
        self._set_meta(conn, "initial_capital", _INITIAL_CASH)
        conn.execute("DELETE FROM meta WHERE key = 'last_run_date'")
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db, check_same_thread=False)

    def _read_meta(self, key: str, default):
        conn = self._connect()
        val = self._get_meta(conn, key, default)
        conn.close()
        return val

    @staticmethod
    def _get_meta(conn: sqlite3.Connection, key: str, default):
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        if key in ("initial_capital",):
            return float(row[0])
        return row[0]

    @staticmethod
    def _set_meta(conn: sqlite3.Connection, key: str, value) -> None:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
