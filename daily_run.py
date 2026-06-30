#!/usr/bin/env python3
"""Standalone daily review for the auto-managed EUR portfolio.

Run this once a day to let the jury manage the portfolio even when the Streamlit
app is closed. It shares the same SQLite store as the app, so trades made here
show up in the UI on next load (and vice-versa).

    source .venv/bin/activate
    python daily_run.py            # Batch API (50% cheaper); no-op if today already ran
    python daily_run.py --force    # run again regardless
    python daily_run.py --sync     # use the live synchronous path instead of batch

Background scheduling on macOS (launchd) — save as
~/Library/LaunchAgents/com.stockpulse.daily.plist and `launchctl load` it:

    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
      "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0"><dict>
      <key>Label</key><string>com.stockpulse.daily</string>
      <key>ProgramArguments</key>
      <array>
        <string>/ABSOLUTE/PATH/stockpulse-trader/.venv/bin/python</string>
        <string>/ABSOLUTE/PATH/stockpulse-trader/daily_run.py</string>
      </array>
      <key>WorkingDirectory</key><string>/ABSOLUTE/PATH/stockpulse-trader</string>
      <key>StartCalendarInterval</key><dict><key>Hour</key><integer>18</integer>
        <key>Minute</key><integer>0</integer></dict>
      <key>StandardOutPath</key><string>/tmp/stockpulse-daily.log</string>
      <key>StandardErrorPath</key><string>/tmp/stockpulse-daily.err</string>
    </dict></plist>

Cron alternative (run at 18:00 daily):
    0 18 * * * cd /ABSOLUTE/PATH/stockpulse-trader && .venv/bin/python daily_run.py >> /tmp/stockpulse-daily.log 2>&1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.managed.cycle import AGENTS, run_daily_cycle, run_daily_cycle_batched
from src.portfolio.managed import ManagedPortfolio


def main() -> int:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or "your_anthropic" in api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set. Add it to your .env file.", file=sys.stderr)
        return 1

    args = sys.argv[1:]
    force = "--force" in args
    # Batch API by default (50% cheaper, async). --sync uses the live synchronous path.
    use_sync = "--sync" in args
    portfolio = ManagedPortfolio()

    if use_sync:
        outcome = run_daily_cycle(portfolio, AGENTS, force=force)
    else:
        outcome = run_daily_cycle_batched(portfolio, AGENTS, force=force)

    if outcome["skipped"]:
        print(f"[{outcome['date']}] Already ran today — nothing to do (use --force to override).")
        return 0

    via = "Batch API (50% cost)" if outcome.get("via") == "batch" else "synchronous"
    print(f"[{outcome['date']}] Daily review complete via {via}. Reviewed {len(outcome['results'])} ticker(s):")
    for r in outcome["results"]:
        flag = "✓" if r["executed"] else "·"
        conf = r.get("confidence", "—")
        print(f"  {flag} {r['ticker']:<10} {r['action']:<4} conf={conf}%  {r['note']}")

    summary = portfolio.get_summary()
    print(
        f"Cash €{summary['cash']:,.2f} | Holdings €{summary['holdings_value']:,.2f} | "
        f"Total €{summary['total_value']:,.2f} ({summary['pnl_pct']:+.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
