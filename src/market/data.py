import csv
import io

import requests
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
        "currency": info.get("currency", "EUR"),
        "financial_currency": info.get("financialCurrency"),
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
        "dividend_yield": _dividend_yield_pct(info, current_price),
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


_FX_CACHE: dict[str, float] = {}  # process-lifetime memo; FX barely moves intraday


def get_fx_rate(from_ccy: str, to_ccy: str = "EUR") -> float:
    """Units of ``to_ccy`` per 1 ``from_ccy`` (e.g. USD→EUR ≈ 0.88). Falls back to 1.0."""
    from_ccy = (from_ccy or "EUR").upper()
    to_ccy = (to_ccy or "EUR").upper()
    if from_ccy == to_ccy:
        return 1.0
    key = f"{from_ccy}{to_ccy}"
    if key in _FX_CACHE:
        return _FX_CACHE[key]

    rate = None
    try:  # Yahoo "{FROM}{TO}=X" quotes TO per 1 FROM
        rate = _fi_get(yf.Ticker(f"{from_ccy}{to_ccy}=X").fast_info, "last_price")
    except Exception:
        rate = None
    if not rate:  # try the inverse pair
        try:
            inv = _fi_get(yf.Ticker(f"{to_ccy}{from_ccy}=X").fast_info, "last_price")
            rate = 1.0 / float(inv) if inv else None
        except Exception:
            rate = None

    rate = float(rate) if rate else 1.0
    _FX_CACHE[key] = rate
    return rate


def to_eur(amount: float | None, from_ccy: str) -> float | None:
    """Convert an amount in ``from_ccy`` to EUR. Handles GBp (pence) and unknowns."""
    if amount is None:
        return None
    ccy = from_ccy or "EUR"
    factor = 1.0
    if ccy == "GBp":  # London pence = GBP / 100
        ccy, factor = "GBP", 0.01
    return round(amount * factor * get_fx_rate(ccy, "EUR"), 2)


def search_symbols(query: str, max_results: int = 8) -> list[dict]:
    """Resolve a company name (or partial ticker) to candidate symbols.

    Returns a list of {symbol, name, exchange, type}, equities/ETFs first so the
    most likely match for a company name sits at the top.
    """
    query = (query or "").strip()
    if not query:
        return []
    try:
        quotes = yf.Search(query, max_results=max_results).quotes or []
    except Exception:
        return []

    results = []
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol:
            continue
        results.append({
            "symbol": symbol,
            "name": q.get("longname") or q.get("shortname") or symbol,
            "exchange": q.get("exchDisp") or q.get("exchange") or "",
            "type": q.get("typeDisp") or q.get("quoteType") or "",
        })

    priority = {"Equity": 0, "ETF": 1}
    results.sort(key=lambda r: priority.get(r["type"], 2))  # stable: keeps Yahoo relevance within a tier
    return results


_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# iShares publishes a fund's *full* constituent list as a "Detailed Holdings"
# CSV — unlike Yahoo, which only exposes the top ~10. Keyed by the tickers a
# user is likely to search; aliases map onto a canonical fund key.
_ISHARES_HOLDINGS_CSV = {
    # iShares MSCI World Small Cap UCITS ETF (LSE: WLDS/WSML · Xetra: IUSN)
    "WLDS": "https://www.ishares.com/uk/individual/en/products/296576/"
            "ishares-msci-world-small-cap-ucits-etf/1506575576011.ajax"
            "?fileType=csv&fileName=WLDS_holdings&dataType=fund",
}
_FUND_ALIASES = {
    "WLDS": "WLDS", "WLDS.L": "WLDS", "WSML": "WLDS", "WSML.L": "WLDS",
    "IUSN": "WLDS", "IUSN.DE": "WLDS", "IUSN.SG": "WLDS",
}

# iShares exchange name → Yahoo Finance ticker suffix. "" means US (no suffix).
# Exchanges absent here are skipped rather than guessed at.
_EXCH_SUFFIX = {
    "New York Stock Exchange Inc.": "", "NASDAQ": "", "Nyse Mkt Llc": "",
    "Tokyo Stock Exchange": ".T", "London Stock Exchange": ".L",
    "Toronto Stock Exchange": ".TO", "Asx - All Markets": ".AX",
    "Nasdaq Omx Nordic": ".ST", "Xetra": ".DE",
    "Nyse Euronext - Euronext Paris": ".PA", "Tel Aviv Stock Exchange": ".TA",
    "SIX Swiss Exchange": ".SW", "Hong Kong Exchanges And Clearing Ltd": ".HK",
    "Borsa Italiana": ".MI", "Singapore Exchange": ".SI", "Oslo Bors Asa": ".OL",
    "Bolsa De Madrid": ".MC", "Euronext Amsterdam": ".AS",
    "Nyse Euronext - Euronext Brussels": ".BR",
    "Omx Nordic Exchange Copenhagen A/S": ".CO", "Nasdaq Omx Helsinki Ltd.": ".HE",
    "Wiener Boerse Ag": ".VI", "Nyse Euronext - Euronext Lisbon": ".LS",
    "New Zealand Exchange Ltd": ".NZ", "Irish Stock Exchange - All Market": ".IR",
}


def _to_yahoo_symbol(ticker: str, exchange: str) -> str | None:
    """Map an iShares (ticker, exchange) to a Yahoo symbol, or None if unmappable."""
    if exchange not in _EXCH_SUFFIX:
        return None
    suffix = _EXCH_SUFFIX[exchange]
    sym = ticker.strip().replace(" ", "-")  # share classes: "VOLV B" → "VOLV-B"
    if not sym:
        return None
    if suffix == ".HK":  # Yahoo zero-pads HK codes to 4 digits (e.g. 522 → 0522)
        sym = sym.zfill(4)
    return sym + suffix


def _parse_ishares_csv(text: str) -> list[dict]:
    lines = text.splitlines()
    try:
        hdr = next(i for i, ln in enumerate(lines) if ln.startswith("Ticker,"))
    except StopIteration:
        return []

    holdings = []
    for row in csv.DictReader(io.StringIO("\n".join(lines[hdr:]))):
        if (row.get("Asset Class") or "").strip() != "Equity":
            continue  # skip cash, derivatives, FX lines
        symbol = _to_yahoo_symbol((row.get("Ticker") or ""), (row.get("Exchange") or "").strip())
        if not symbol:
            continue
        try:
            weight = round(float((row.get("Weight (%)") or "0").replace(",", "")), 2)
        except (TypeError, ValueError):
            weight = None
        holdings.append({
            "symbol": symbol,
            "name": (row.get("Name") or symbol).title(),
            "weight_pct": weight,
        })
    return holdings


def get_full_fund_holdings(symbol: str) -> list[dict]:
    """Full constituent list from the fund issuer (iShares) as ``[{symbol, name,
    weight_pct}]``, with symbols already resolved to Yahoo tickers.

    Returns ``[]`` when the symbol isn't a known issuer fund or the download
    fails — callers fall back to Yahoo's top-holdings via ``get_fund_holdings``.
    """
    key = _FUND_ALIASES.get((symbol or "").strip().upper())
    if not key:
        return []
    try:
        resp = requests.get(_ISHARES_HOLDINGS_CSV[key], headers={"User-Agent": _UA}, timeout=20)
        resp.raise_for_status()
    except Exception:
        return []
    return _parse_ishares_csv(resp.content.decode("utf-8-sig"))


def get_fund_holdings(symbol: str) -> list[dict]:
    """Top constituents of an ETF / index fund as ``[{symbol, name, weight_pct}]``.

    Yahoo only publishes a fund's *top* holdings (typically ~10), so this returns
    at most that many — not the full index. Returns ``[]`` for a non-fund symbol
    or when Yahoo has no holdings data, so callers can use a truthy check to tell
    whether the picked symbol is a screenable fund.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return []
    try:
        th = yf.Ticker(symbol).funds_data.top_holdings
    except Exception:
        return []
    if th is None or len(th) == 0:
        return []

    holdings = []
    for sym, row in th.iterrows():
        try:
            weight = round(float(row.get("Holding Percent", 0)) * 100, 2)
        except (TypeError, ValueError):
            weight = None
        holdings.append({
            "symbol": str(sym),
            "name": row.get("Name") or str(sym),
            "weight_pct": weight,
        })
    return holdings


def get_quote_preview(symbol: str) -> dict | None:
    """Lightweight live quote for confirming a pick: price, currency, exchange."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None
    try:
        fi = yf.Ticker(symbol).fast_info
        price = _fi_get(fi, "last_price")
        price = round(float(price), 2) if price is not None else None
        currency = _fi_get(fi, "currency") or ""
        return {
            "symbol": symbol,
            "price": price,
            "currency": currency,
            "price_eur": to_eur(price, currency),
            "exchange": _fi_get(fi, "exchange") or "",
        }
    except Exception:
        return None


def _fi_get(fast_info, key):
    """fast_info raises KeyError for absent keys instead of returning None."""
    try:
        return fast_info[key]
    except Exception:
        return None


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


def get_eur_quotes(symbols: list[str]) -> dict[str, dict]:
    """Per-symbol live quote with native price, currency, and the EUR-converted price.

    Returns {symbol: {"native_price", "currency", "price_eur"}}.
    """
    quotes = {}
    for symbol in symbols:
        try:
            fi = yf.Ticker(symbol).fast_info
            price = _fi_get(fi, "last_price")
            if price is None:
                continue
            price = round(float(price), 2)
            ccy = _fi_get(fi, "currency") or "EUR"
            quotes[symbol] = {
                "native_price": price,
                "currency": ccy,
                "price_eur": to_eur(price, ccy),
            }
        except Exception:
            pass
    return quotes


def _dividend_yield_pct(info: dict, price: float | None):
    """Dividend yield as a percentage, robust to yfinance's unit changes.

    yfinance flipped `dividendYield` from a fraction to a percent in recent
    versions (so `_safe_pct` would 100× it → e.g. 167%). `dividendRate` (€/share)
    and `trailingAnnualDividendYield` (a fraction) are stable, so derive from the
    per-share rate when possible and fall back to the fractional yield.
    """
    rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    if rate and price:
        return round(float(rate) / price * 100, 2)
    ty = info.get("trailingAnnualDividendYield")
    if ty is not None:
        return _safe_pct(ty)
    return None


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
