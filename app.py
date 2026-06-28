import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.market.data import get_stock_data, get_current_prices
from src.agents.buffett import BuffettAgent
from src.agents.dalio import DalioAgent
from src.agents.soros import SorosAgent
from src.agents.lynch import LynchAgent
from src.agents.simons import SimonsAgent
from src.jury import jury as jury_module
from src.portfolio.sandbox import SandboxPortfolio

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

portfolio = get_portfolio()

AGENTS = [
    ("Warren Buffett", BuffettAgent),
    ("Ray Dalio",      DalioAgent),
    ("George Soros",   SorosAgent),
    ("Peter Lynch",    LynchAgent),
    ("Jim Simons",     SimonsAgent),
]

# ── Sidebar — portfolio ────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Portfolio")

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

    with st.expander("⚠️ Reset Portfolio"):
        st.warning("Erases all positions, transactions, and restores $100,000 cash.")
        if st.button("Confirm Reset", type="secondary", use_container_width=True):
            portfolio.reset()
            st.session_state.pop("analysis", None)
            st.success("Portfolio reset to $100,000.")
            st.rerun()

# ── Main ───────────────────────────────────────────────────────────────
st.title("🏛️ StockPulse Trader")
st.caption("Five legendary investors form a jury — and they decide whether to trade.")

api_key = os.getenv("ANTHROPIC_API_KEY", "")
if not api_key or "your_anthropic" in api_key:
    st.error("**ANTHROPIC_API_KEY** is not set. Create a `.env` file from `.env.example` and add your key.")
    st.stop()

# ── Ticker form ────────────────────────────────────────────────────────
with st.form("analyze_form"):
    col_input, col_btn = st.columns([5, 1])
    ticker_raw = col_input.text_input(
        "Stock ticker", placeholder="e.g. AAPL  TSLA  MSFT  NVDA  META  ALV.DE",
        label_visibility="collapsed",
    )
    submitted = col_btn.form_submit_button("⚖️ Convene Jury", type="primary", use_container_width=True)

# ── Run analysis (only when form submitted) ────────────────────────────
if submitted and ticker_raw.strip():
    ticker = ticker_raw.strip().upper()

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

# ── Render results from session state (survives any rerun) ─────────────
if "analysis" not in st.session_state:
    st.info("Enter a stock ticker and click **Convene Jury** to start analysis.")
else:
    analysis    = st.session_state["analysis"]
    ticker      = analysis["ticker"]
    market_data = analysis["market_data"]
    verdicts    = analysis["verdicts"]
    decision    = analysis["decision"]

    # ── Stock overview ─────────────────────────────────────────────────
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

    # ── Agent cards ────────────────────────────────────────────────────
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

    # ── Jury verdict ───────────────────────────────────────────────────
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

# ── Transaction history ────────────────────────────────────────────────
st.divider()
st.subheader("📋 Transaction History")

transactions = portfolio.get_transactions()
if not transactions:
    st.info("No trades yet. Analyze a stock and execute a trade to see history here.")
else:
    for tx in transactions:
        direction_icon = {"BUY": "🟢", "SELL": "🔴"}.get(tx["action"], "⚪")
        header = (
            f"{tx['timestamp'][:10]}  ·  "
            f"{direction_icon} **{tx['action']}** {tx['shares']} × {tx['symbol']}  ·  "
            f"${tx['price']:.2f}/sh = **${tx['total_value']:,.0f}**  ·  "
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
