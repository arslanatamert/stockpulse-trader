import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import streamlit as st

from src.market.data import (
    get_stock_data, get_current_prices, get_eur_quotes,
    search_symbols, get_quote_preview,
)
from src.agents.base_agent import DEFAULT_JURY_MODEL
from src.managed.cycle import AGENTS, run_daily_cycle
from src.jury import jury as jury_module
from src.portfolio.sandbox import SandboxPortfolio
from src.portfolio.managed import ManagedPortfolio

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockPulse Trader",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.verdict-card  { border-radius:8px; padding:12px; margin:4px 0; border-left:4px solid #ccc; }
.verdict-buy   { border-left-color:#22c55e; background:#f0fdf4; }
.verdict-sell  { border-left-color:#ef4444; background:#fef2f2; }
.verdict-hold  { border-left-color:#f59e0b; background:#fffbeb; }
.badge-buy     { background:#22c55e; color:white; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:0.85em; }
.badge-sell    { background:#ef4444; color:white; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:0.85em; }
.badge-hold    { background:#f59e0b; color:white; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:0.85em; }
.jury-box      { border-radius:12px; padding:20px; margin:16px 0; }
</style>
""", unsafe_allow_html=True)

# ── Shared resources ───────────────────────────────────────────────────
@st.cache_resource
def get_portfolio() -> SandboxPortfolio:
    return SandboxPortfolio()

@st.cache_resource
def get_managed_portfolio() -> ManagedPortfolio:
    return ManagedPortfolio()

portfolio = get_portfolio()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_search(query: str) -> list[dict]:
    return search_symbols(query)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_quote(symbol: str) -> dict | None:
    return get_quote_preview(symbol)


def stock_search_picker(key_prefix: str, label: str = "Search company or ticker") -> dict | None:
    """Type a company name → pick from matching symbols → confirm with live price.

    Returns the chosen {symbol, name, exchange, price, currency} or None. Must be
    used OUTSIDE an st.form so the dropdown updates as you type.
    """
    query = st.text_input(label, key=f"{key_prefix}_query", placeholder="e.g. Allianz, NVIDIA, AAPL")
    if not query.strip():
        return None

    results = _cached_search(query)
    if results:
        labels = [
            f"{r['symbol']} — {r['name']} · {r['exchange']}" + (f" ({r['type']})" if r["type"] else "")
            for r in results
        ]
        chosen = st.selectbox("Matches", labels, key=f"{key_prefix}_sel")
        picked = results[labels.index(chosen)]
    else:
        # No name match — fall back to treating the input as a literal ticker.
        probe = _cached_quote(query.strip().upper())
        if not (probe and probe.get("price") is not None):
            st.caption("No matches found. Check the spelling or enter an exact ticker.")
            return None
        picked = {"symbol": probe["symbol"], "name": probe["symbol"], "exchange": probe.get("exchange", ""), "type": ""}

    quote = _cached_quote(picked["symbol"])
    if quote and quote.get("price") is not None:
        exch = picked["exchange"] or quote.get("exchange")  # prefer search's friendly name (XETRA vs GER)
        ccy  = quote.get("currency", "")
        price_str = f"{quote['price']:.2f} {ccy}"
        if ccy and ccy != "EUR" and quote.get("price_eur") is not None:
            price_str += f" (≈ €{quote['price_eur']:.2f})"
        st.success(f"✓ **{picked['symbol']}** · {picked['name']} · {exch} · **{price_str}**")
        return {**picked, "price": quote["price"], "currency": ccy,
                "price_eur": quote.get("price_eur"), "exchange": exch}

    st.warning(f"Selected **{picked['symbol']}** but couldn't fetch a live price — verify the symbol.")
    return {**picked, "price": None, "currency": ""}


def render_transactions(transactions: list[dict], cur: str = "$") -> None:
    """Shared transaction-history renderer used by both tabs."""
    if not transactions:
        st.info("No trades yet.")
        return
    for tx in transactions:
        direction_icon = {"BUY": "🟢", "SELL": "🔴"}.get(tx["action"], "⚪")
        header = (
            f"{tx['timestamp'][:10]}  ·  "
            f"{direction_icon} **{tx['action']}** {tx['shares']} × {tx['symbol']}  ·  "
            f"{cur}{tx['price']:.2f}/sh = **{cur}{tx['total_value']:,.0f}**  ·  "
            f"Jury: {tx['jury_action']} {tx['jury_confidence']:.0f}%"
        )
        with st.expander(header):
            st.markdown(f"**Jury reasoning:** {tx['jury_reasoning']}")
            st.markdown(f"**Vote breakdown:** `{tx['vote_summary']}`")
            st.markdown("**Agent votes at time of trade:**")
            vcols = st.columns(5)
            for j, col in enumerate(vcols):
                if j < len(tx["agent_names"]):
                    act  = tx["agent_actions"][j]
                    icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(act, "⚪")
                    col.markdown(f"**{tx['agent_names'][j]}**")
                    col.markdown(f"{icon} **{act}** ({tx['agent_confidences'][j]}%)")
                    col.caption(tx["agent_reasonings"][j])


# ── Sidebar — sandbox portfolio (USD) ──────────────────────────────────
with st.sidebar:
    st.title("📊 Sandbox Portfolio")
    st.caption("Manual trading sandbox (USD).")

    positions  = portfolio.get_positions()
    symbols    = [p["symbol"] for p in positions]
    cur_prices = get_current_prices(symbols) if symbols else {}
    summary    = portfolio.get_summary(cur_prices)

    c1, c2 = st.columns(2)
    c1.metric("Cash",        f"${summary['cash']:,.0f}")
    c2.metric("Total Value", f"${summary['total_value']:,.0f}",
              delta=f"{summary['pnl_pct']:+.2f}%")

    if summary["positions"]:
        st.subheader("Holdings")
        for pos in summary["positions"]:
            icon = "🟢" if pos["pnl_pct"] >= 0 else "🔴"
            with st.expander(f"{pos['symbol']}  ·  {pos['shares']} sh"):
                st.write(f"Avg cost: **${pos['avg_cost']:.2f}**")
                st.write(f"Current:  **${pos['current_price']:.2f}**")
                st.write(f"P&L: {icon} **{pos['pnl_pct']:+.2f}%**")
                st.write(f"Value:    **${pos['value']:,.0f}**")
    else:
        st.caption("No open positions yet.")

    st.divider()
    if st.button("🔄 Refresh Prices", use_container_width=True):
        st.rerun()

    with st.expander("⚠️ Reset Sandbox"):
        st.warning("Erases all positions, transactions, and restores $100,000 cash.")
        if st.button("Confirm Reset", type="secondary", use_container_width=True):
            portfolio.reset()
            st.session_state.pop("analysis", None)
            st.success("Portfolio reset to $100,000.")
            st.rerun()

    st.divider()
    st.subheader("⚙️ Jury Model")
    JURY_MODELS = {
        "Haiku 4.5 — fast & cheap":       "claude-haiku-4-5-20251001",
        "Sonnet 4.6 — deeper reasoning":  "claude-sonnet-4-6",
        "Opus 4.8 — deepest, costly":     "claude-opus-4-8",
    }
    _ids   = list(JURY_MODELS.values())
    _cur   = os.getenv("JURY_MODEL", DEFAULT_JURY_MODEL)
    _idx   = _ids.index(_cur) if _cur in _ids else 0
    _label = st.selectbox("Model for all 5 agents", list(JURY_MODELS.keys()), index=_idx)
    os.environ["JURY_MODEL"] = JURY_MODELS[_label]
    st.caption("Applies to this app session (both tabs). Background daily runs use `JURY_MODEL` in `.env`.")

# ── Header + shared API-key check ──────────────────────────────────────
st.title("🏛️ StockPulse Trader")
st.caption("Five legendary investors form a jury — and they decide whether to trade.")

api_key = os.getenv("ANTHROPIC_API_KEY", "")
if not api_key or "your_anthropic" in api_key:
    st.error("**ANTHROPIC_API_KEY** is not set. Create a `.env` file from `.env.example` and add your key.")
    st.stop()

tab_jury, tab_managed = st.tabs(["⚖️ Convene Jury", "🤖 Managed Portfolio"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — Manual "convene the jury" on a single ticker
# ═══════════════════════════════════════════════════════════════════════
with tab_jury:
    # ── Stock picker ───────────────────────────────────────────────────
    jury_pick = stock_search_picker("jury", label="Search a company or ticker to analyze")
    submitted = st.button("⚖️ Convene Jury", type="primary",
                          disabled=jury_pick is None, key="jury_btn")

    # ── Run analysis (only when a stock is picked and button clicked) ──
    if submitted and jury_pick:
        ticker = jury_pick["symbol"]

        with st.spinner(f"Fetching market data for **{ticker}**…"):
            try:
                market_data = get_stock_data(ticker)
            except Exception as exc:
                st.error(f"Could not load data for **{ticker}**: {exc}")
                st.stop()

        agent_cols   = st.columns(5)
        placeholders = [col.empty() for col in agent_cols]

        verdicts = []
        for i, (name, AgentClass) in enumerate(AGENTS):
            placeholders[i].info(f"**{name}**\n\n⏳ Thinking…")
            try:
                verdict = AgentClass().analyze(ticker, market_data)
            except Exception as exc:
                placeholders[i].error(f"**{name}**\n\nError: {exc}")
                continue
            verdicts.append(verdict)

            action    = verdict.action
            card_cls  = f"verdict-{action.lower()}"
            badge_cls = f"badge-{action.lower()}"
            factors_li = "".join(f"<li>{f}</li>" for f in verdict.key_factors)
            placeholders[i].markdown(
                f"""<div class="verdict-card {card_cls}">
                <b>{name}</b><br/>
                <span class="{badge_cls}">{action}</span>&nbsp;
                <small><b>{verdict.confidence}%</b> confidence</small>
                <hr style="margin:6px 0"/>
                <small>{verdict.reasoning}</small>
                <ul style="margin:4px 0 4px 0;padding-left:16px;font-size:0.8em">{factors_li}</ul>
                <small style="color:#666"><i>Risk: {verdict.risk_assessment}</i></small>
                </div>""",
                unsafe_allow_html=True,
            )

        if not verdicts:
            st.error("All agents failed. Verify your **ANTHROPIC_API_KEY** and try again.")
            st.stop()

        decision = jury_module.deliberate(verdicts)

        # Persist to session state so results survive the next rerun
        st.session_state["analysis"] = {
            "ticker":      ticker,
            "market_data": market_data,
            "verdicts":    verdicts,
            "decision":    decision,
        }
        st.rerun()   # clean rerun so results render from session state (no double-render)

    # ── Render results from session state (survives any rerun) ─────────
    if "analysis" not in st.session_state:
        st.info("Enter a stock ticker and click **Convene Jury** to start analysis.")
    else:
        analysis    = st.session_state["analysis"]
        ticker      = analysis["ticker"]
        market_data = analysis["market_data"]
        verdicts    = analysis["verdicts"]
        decision    = analysis["decision"]

        # ── Stock overview ─────────────────────────────────────────────
        st.subheader(f"{market_data['name']}  ·  {ticker}")
        st.caption(f"{market_data['sector']}  /  {market_data['industry']}")

        row1 = st.columns(6)
        row1[0].metric("Price",       f"${market_data['price']:.2f}")
        row1[1].metric("P/E (TTM)",   market_data["pe_ratio"]  or "—")
        row1[2].metric("PEG",         market_data["peg_ratio"] or "—")
        row1[3].metric("Market Cap",  market_data["market_cap_fmt"])
        row1[4].metric("1Y Momentum", f"{market_data['momentum_1y_pct']:+.1f}%" if market_data["momentum_1y_pct"] is not None else "—")
        row1[5].metric("Beta",        market_data["beta"] or "—")

        row2 = st.columns(6)
        row2[0].metric("ROE",          f"{market_data['roe']:.1f}%"           if market_data["roe"]           is not None else "—")
        row2[1].metric("D/E Ratio",    market_data["debt_to_equity"]          or "—")
        row2[2].metric("Profit Margin",f"{market_data['profit_margin']:.1f}%" if market_data["profit_margin"] is not None else "—")
        row2[3].metric("FCF",          market_data["free_cashflow_fmt"])
        row2[4].metric("Rev Growth",   f"{market_data['revenue_growth_yoy']:+.1f}%" if market_data["revenue_growth_yoy"] is not None else "—")
        row2[5].metric("Short % Float",f"{market_data['short_pct_float']:.1f}%"    if market_data["short_pct_float"]    is not None else "—")

        if market_data.get("business_summary"):
            with st.expander("Business summary"):
                st.write(market_data["business_summary"])

        # ── Agent cards ────────────────────────────────────────────────
        st.divider()
        st.subheader("🧠 Jury Deliberation")
        agent_cols = st.columns(5)
        for i, (verdict, col) in enumerate(zip(verdicts, agent_cols)):
            action    = verdict.action
            card_cls  = f"verdict-{action.lower()}"
            badge_cls = f"badge-{action.lower()}"
            factors_li = "".join(f"<li>{f}</li>" for f in verdict.key_factors)
            col.markdown(
                f"""<div class="verdict-card {card_cls}">
                <b>{verdict.agent_name}</b><br/>
                <span class="{badge_cls}">{action}</span>&nbsp;
                <small><b>{verdict.confidence}%</b> confidence</small>
                <hr style="margin:6px 0"/>
                <small>{verdict.reasoning}</small>
                <ul style="margin:4px 0 4px 0;padding-left:16px;font-size:0.8em">{factors_li}</ul>
                <small style="color:#666"><i>Risk: {verdict.risk_assessment}</i></small>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── Jury verdict ───────────────────────────────────────────────
        st.divider()
        st.subheader("⚖️ Jury Verdict")

        _color = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#f59e0b"}.get(decision.action, "#888")
        _icon  = {"BUY": "🟢",      "SELL": "🔴",      "HOLD": "🟡"     }.get(decision.action, "⚪")

        verdict_col, trade_col = st.columns([3, 1])

        with verdict_col:
            st.markdown(
                f"""<div class="jury-box" style="background:{_color}18;border:2px solid {_color};">
                <h2 style="color:{_color};margin:0">{_icon} {decision.action}</h2>
                <p style="margin:10px 0 4px 0">{decision.reasoning}</p>
                <small style="color:#555">{decision.vote_summary}</small>
                </div>""",
                unsafe_allow_html=True,
            )
            st.progress(int(decision.confidence) / 100,
                        text=f"Jury conviction: {decision.confidence:.0f}%")
            if decision.dissenting_views:
                with st.expander("Dissenting views"):
                    for dv in decision.dissenting_views:
                        st.markdown(f"- {dv}")

        with trade_col:
            st.markdown("**Execute in Sandbox**")
            if decision.action in ("BUY", "SELL"):
                shares = st.number_input("Shares", min_value=1, value=10, step=1, key="shares_input")
                est    = shares * market_data["price"]
                st.caption(f"Est. total: **${est:,.2f}**")
                if decision.action == "BUY":
                    st.caption(f"Cash available: **${summary['cash']:,.0f}**")

                if st.button(f"✅ {decision.action} {shares} shares",
                             type="primary", use_container_width=True, key="trade_btn"):
                    try:
                        portfolio.execute_trade(
                            symbol=ticker,
                            action=decision.action,
                            shares=shares,
                            price=market_data["price"],
                            decision=decision,
                            verdicts=verdicts,
                        )
                        st.success(
                            f"**{decision.action}** {shares} × {ticker} "
                            f"@ ${market_data['price']:.2f} = **${est:,.2f}**"
                        )
                        st.rerun()  # safe: results stay in session_state
                    except ValueError as exc:
                        st.error(str(exc))
            else:
                st.info("Jury recommends **HOLD** — no trade executed.")

    # ── Transaction history ────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Transaction History")
    render_transactions(portfolio.get_transactions(), cur="$")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — Auto-managed EUR portfolio (jury runs it daily)
# ═══════════════════════════════════════════════════════════════════════
with tab_managed:
    mp = get_managed_portfolio()
    today = date.today().isoformat()

    st.subheader("🤖 Jury-Managed Portfolio")
    st.caption(
        "Seed your real holdings + €1000 cash. Each day the jury reviews every holding and your "
        "watchlist, then buys / holds / sells with auto-sized quantities. "
        "Non-EUR stocks are converted to EUR at the current FX rate."
    )

    # ── On-start catch-up: run once per session if today hasn't run yet ─
    if not st.session_state.get("managed_catchup_done"):
        st.session_state["managed_catchup_done"] = True
        if mp.get_last_run() != today and (mp.get_positions() or mp.get_watchlist()):
            with st.spinner("Running today's jury review (catch-up)…"):
                outcome = run_daily_cycle(mp, AGENTS, force=False)
            if not outcome.get("skipped"):
                st.session_state["managed_last_outcome"] = outcome

    # ── Summary metrics ────────────────────────────────────────────────
    m_positions = mp.get_positions()
    m_symbols   = [p["symbol"] for p in m_positions]
    m_quotes    = get_eur_quotes(m_symbols) if m_symbols else {}
    m_prices    = {s: q["price_eur"] for s, q in m_quotes.items() if q["price_eur"] is not None}
    m_summary   = mp.get_summary(m_prices)

    cols = st.columns(4)
    cols[0].metric("Cash",          f"€{m_summary['cash']:,.2f}")
    cols[1].metric("Holdings",      f"€{m_summary['holdings_value']:,.2f}")
    cols[2].metric("Total Value",   f"€{m_summary['total_value']:,.2f}",
                   delta=f"{m_summary['pnl_pct']:+.2f}%")
    last_run = mp.get_last_run() or "never"
    cols[3].metric("Last Review",   last_run)

    st.divider()

    # ── Seed holdings + watchlist editors ──────────────────────────────
    edit_l, edit_r = st.columns(2)

    with edit_l:
        st.markdown("**➕ Seed / add a holding**")
        st.caption("Pre-owned shares. Cost basis seeds your P&L baseline; cash is untouched.")
        seed_pick = stock_search_picker("seed")
        sc1, sc2 = st.columns(2)
        s_shares = sc1.number_input("Shares", min_value=1, value=1, step=1, key="seed_shares")
        s_cost   = sc2.number_input("Avg cost (€)", min_value=0.01, value=100.0, step=1.0, key="seed_cost",
                                    help="Your historical purchase price — not necessarily today's price shown above.")
        if st.button("Add holding", use_container_width=True, disabled=seed_pick is None, key="seed_add"):
            try:
                mp.seed_holding(seed_pick["symbol"], int(s_shares), float(s_cost))
                st.success(f"Seeded {int(s_shares)} × {seed_pick['symbol']} @ €{s_cost:.2f}")
                for k in ("seed_query", "seed_sel"):
                    st.session_state.pop(k, None)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        if m_summary["positions"]:
            st.markdown("**Current holdings**")
            for pos in m_summary["positions"]:
                icon = "🟢" if pos["pnl_pct"] >= 0 else "🔴"
                q   = m_quotes.get(pos["symbol"], {})
                ccy = q.get("currency", "EUR")
                with st.expander(f"{pos['symbol']}  ·  {pos['shares']} sh  ·  {ccy}  ·  {icon} {pos['pnl_pct']:+.2f}%"):
                    st.write(f"Avg cost: **€{pos['avg_cost']:.2f}**")
                    if ccy and ccy != "EUR" and q.get("native_price") is not None:
                        st.write(f"Current:  **€{pos['current_price']:.2f}**  ({q['native_price']:.2f} {ccy})")
                    else:
                        st.write(f"Current:  **€{pos['current_price']:.2f}**")
                    st.write(f"Value:    **€{pos['value']:,.2f}**")
        else:
            st.caption("No holdings yet — seed one above.")

        if st.button("📸 Snapshot portfolio value", use_container_width=True, key="snap_holdings"):
            snap = mp.record_snapshot(m_prices)
            st.success(f"Recorded portfolio value: €{snap['total_value']:,.2f}")
            st.rerun()

    with edit_r:
        st.markdown("**👀 Watchlist**")
        st.caption("Tickers you don't own yet. The jury may open new positions here with spare cash.")
        watch_pick = stock_search_picker("watch")
        if st.button("Add to watchlist", use_container_width=True, disabled=watch_pick is None, key="watch_add"):
            mp.add_watch(watch_pick["symbol"])
            for k in ("watch_query", "watch_sel"):
                st.session_state.pop(k, None)
            st.rerun()

        watchlist = mp.get_watchlist()
        if watchlist:
            for sym in watchlist:
                wc1, wc2 = st.columns([4, 1])
                wc1.write(f"• **{sym}**")
                if wc2.button("✕", key=f"rmwatch_{sym}"):
                    mp.remove_watch(sym)
                    st.rerun()
        else:
            st.caption("Watchlist is empty.")

        if st.button("📸 Snapshot watchlist value", use_container_width=True,
                     disabled=not watchlist, key="snap_watch"):
            wq = get_eur_quotes(watchlist)
            basket = round(sum(q["price_eur"] for q in wq.values() if q["price_eur"] is not None), 2)
            mp.record_watchlist_snapshot(basket, len(wq))
            st.success(f"Recorded watchlist basket: €{basket:,.2f} across {len(wq)} item(s)")
            st.rerun()

    st.divider()

    # ── Daily review controls ──────────────────────────────────────────
    run_col, info_col = st.columns([1, 3])
    with run_col:
        if st.button("▶️ Run today's review", type="primary", use_container_width=True):
            if not (mp.get_positions() or mp.get_watchlist()):
                st.warning("Add a holding or watchlist ticker first.")
            else:
                with st.spinner("Convening the jury for each ticker…"):
                    outcome = run_daily_cycle(mp, AGENTS, force=True)
                st.session_state["managed_last_outcome"] = outcome
                st.rerun()
    with info_col:
        st.caption(
            "Auto-runs once when you open the app on a new day. For true background runs even when "
            "the app is closed, schedule `daily_run.py` via cron/launchd (see the file header)."
        )

    last_outcome = st.session_state.get("managed_last_outcome")
    if last_outcome and not last_outcome.get("skipped"):
        st.markdown(f"**Latest review — {last_outcome['date']}**")
        for r in last_outcome["results"]:
            badge = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(r["action"], "⚪")
            flag  = "✅ executed" if r.get("executed") else "—"
            conf  = r.get("confidence", "—")
            st.markdown(
                f"{badge} **{r['ticker']}** · {r['action']} ({conf}%) · {flag} · "
                f"<small style='color:#666'>{r['note']}</small>",
                unsafe_allow_html=True,
            )

    # ── Performance over time ──────────────────────────────────────────
    st.divider()
    st.subheader("📈 Performance")
    snaps = mp.get_snapshots()
    if not snaps:
        st.caption("No history yet — each daily review records a point. Run a review to start the curve.")
    else:
        chart_df = pd.DataFrame(snaps)
        chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
        chart_df = chart_df.set_index("timestamp").rename(columns={"total_value": "Total value"})
        chart_df["Invested (baseline)"] = m_summary["initial_capital"]
        st.line_chart(chart_df[["Total value", "Invested (baseline)"]], color=["#22c55e", "#9ca3af"])
        if len(snaps) == 1:
            st.caption("Just one data point so far — more appear with each daily review.")

    w_snaps = mp.get_watchlist_snapshots()
    if w_snaps:
        st.markdown("**👀 Watchlist basket value**")
        st.caption("Combined current value of watchlist tickers (1 unit each), captured on demand.")
        w_df = pd.DataFrame(w_snaps)
        w_df["timestamp"] = pd.to_datetime(w_df["timestamp"])
        w_df = w_df.set_index("timestamp").rename(columns={"basket_value": "Watchlist basket"})
        st.line_chart(w_df[["Watchlist basket"]], color=["#3b82f6"])

    # ── Managed transaction history ────────────────────────────────────
    st.divider()
    st.subheader("📋 Managed Transaction History")
    render_transactions(mp.get_transactions(), cur="€")

    # ── Reset ──────────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚠️ Reset Managed Portfolio"):
        st.warning("Erases all holdings, watchlist, transactions, and restores €1000 cash.")
        if st.button("Confirm Managed Reset", type="secondary"):
            mp.reset()
            st.session_state.pop("managed_last_outcome", None)
            st.session_state.pop("managed_catchup_done", None)
            st.success("Managed portfolio reset to €1000.")
            st.rerun()
