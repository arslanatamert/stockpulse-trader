import json
import os
import sqlite3
from datetime import datetime

from src.agents.base_agent import AgentVerdict
from src.jury.jury import JuryDecision
from src.portfolio._sold import derive_sold_positions

_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "portfolio.db")
_INITIAL_CASH = float(os.getenv("INITIAL_CASH", "100000"))

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


class SandboxPortfolio:
    def __init__(self, db_path: str = _DEFAULT_DB):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db = db_path
        conn = self._connect()
        conn.executescript(_DDL)
        row = conn.execute("SELECT COUNT(*) FROM cash").fetchone()
        if row[0] == 0:
            conn.execute("INSERT INTO cash (id, balance) VALUES (1, ?)", (_INITIAL_CASH,))
            conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Public API
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
                        f"Insufficient funds. Need ${total:,.2f}, available ${cash:,.2f}."
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

        enriched = []
        holdings_value = 0.0
        for pos in positions:
            cur = cp.get(pos["symbol"], pos["avg_cost"])
            val = pos["shares"] * cur
            holdings_value += val
            pnl_pct = (cur - pos["avg_cost"]) / pos["avg_cost"] * 100
            enriched.append({**pos, "current_price": cur, "value": val, "pnl_pct": round(pnl_pct, 2)})

        total = cash + holdings_value
        overall_pnl = total - _INITIAL_CASH
        overall_pnl_pct = overall_pnl / _INITIAL_CASH * 100

        return {
            "cash": cash,
            "holdings_value": holdings_value,
            "total_value": total,
            "pnl": overall_pnl,
            "pnl_pct": round(overall_pnl_pct, 2),
            "positions": enriched,
            "initial_cash": _INITIAL_CASH,
        }

    def get_sold_positions(self) -> dict:
        """Realized 'sold' buckets (Total sold / Partially sold) from trade history.

        Thin wrapper around :func:`derive_sold_positions`; see there for the
        avg-cost replay details. The sandbox has no seeding, so every acquired
        share comes from a recorded BUY.
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

    def reset(self):
        conn = self._connect()
        conn.execute("UPDATE cash SET balance = ? WHERE id = 1", (_INITIAL_CASH,))
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM transactions")
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db, check_same_thread=False)
