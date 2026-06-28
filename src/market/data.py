import yfinance as yf


def get_stock_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info

    hist = stock.history(period="1y")
    if hist.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol and try again.")

    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or float(hist["Close"].iloc[-1])
    )

    momentum_1y = None
    if len(hist) >= 50:
        price_start = float(hist["Close"].iloc[0])
        momentum_1y = round((current_price - price_start) / price_start * 100, 2)

    momentum_3m = None
    if len(hist) >= 63:
        price_3m = float(hist["Close"].iloc[-63])
        momentum_3m = round((current_price - price_3m) / price_3m * 100, 2)

    momentum_1m = None
    if len(hist) >= 21:
        price_1m = float(hist["Close"].iloc[-21])
        momentum_1m = round((current_price - price_1m) / price_1m * 100, 2)

    avg_volume_90d = None
    vol_ratio = None
    if len(hist) >= 20:
        avg_volume_90d = int(hist["Volume"].tail(90).mean())
        recent_volume = int(hist["Volume"].iloc[-1])
        if avg_volume_90d > 0:
            vol_ratio = round(recent_volume / avg_volume_90d, 2)

    market_cap = info.get("marketCap")
    market_cap_fmt = _fmt_large_number(market_cap)

    free_cashflow = info.get("freeCashflow")
    revenue = info.get("totalRevenue")

    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "price": round(current_price, 2),
        "market_cap": market_cap,
        "market_cap_fmt": market_cap_fmt,
        "pe_ratio": _safe_round(info.get("trailingPE")),
        "forward_pe": _safe_round(info.get("forwardPE")),
        "peg_ratio": _safe_round(info.get("pegRatio")),
        "price_to_book": _safe_round(info.get("priceToBook")),
        "eps_ttm": _safe_round(info.get("trailingEps")),
        "eps_growth_yoy": _safe_pct(info.get("earningsGrowth")),
        "revenue_growth_yoy": _safe_pct(info.get("revenueGrowth")),
        "gross_margin": _safe_pct(info.get("grossMargins")),
        "profit_margin": _safe_pct(info.get("profitMargins")),
        "roe": _safe_pct(info.get("returnOnEquity")),
        "roa": _safe_pct(info.get("returnOnAssets")),
        "debt_to_equity": _safe_round(info.get("debtToEquity")),
        "current_ratio": _safe_round(info.get("currentRatio")),
        "free_cashflow": free_cashflow,
        "free_cashflow_fmt": _fmt_large_number(free_cashflow),
        "revenue": revenue,
        "revenue_fmt": _fmt_large_number(revenue),
        "dividend_yield": _safe_pct(info.get("dividendYield")),
        "beta": _safe_round(info.get("beta")),
        "short_ratio": _safe_round(info.get("shortRatio")),
        "short_pct_float": _safe_pct(info.get("shortPercentOfFloat")),
        "52w_high": _safe_round(info.get("fiftyTwoWeekHigh")),
        "52w_low": _safe_round(info.get("fiftyTwoWeekLow")),
        "momentum_1y_pct": momentum_1y,
        "momentum_3m_pct": momentum_3m,
        "momentum_1m_pct": momentum_1m,
        "avg_volume_90d": avg_volume_90d,
        "volume_ratio": vol_ratio,
        "analyst_target_price": _safe_round(info.get("targetMeanPrice")),
        "analyst_recommendation": info.get("recommendationKey", "N/A"),
        "business_summary": (info.get("longBusinessSummary") or "")[:400],
    }


def get_current_prices(symbols: list[str]) -> dict[str, float]:
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                prices[symbol] = round(float(hist["Close"].iloc[-1]), 2)
        except Exception:
            pass
    return prices


def _safe_round(val, digits: int = 2):
    if val is None:
        return None
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return None


def _safe_pct(val):
    if val is None:
        return None
    try:
        return round(float(val) * 100, 2)
    except (TypeError, ValueError):
        return None


def _fmt_large_number(val) -> str:
    if val is None:
        return "N/A"
    try:
        val = float(val)
        if val >= 1e12:
            return f"${val / 1e12:.1f}T"
        if val >= 1e9:
            return f"${val / 1e9:.1f}B"
        if val >= 1e6:
            return f"${val / 1e6:.1f}M"
        return f"${val:,.0f}"
    except (TypeError, ValueError):
        return "N/A"
