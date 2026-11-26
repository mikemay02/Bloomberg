"""Marktdaten ueber yfinance: Kurse, Stammdaten, Marktmonitor."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

# Marktmonitor (WEI-artig): Symbol -> Anzeigename
MONITOR = {
    "^GSPC": "S&P 500",
    "^NDX": "NASDAQ 100",
    "^DJI": "DOW JONES",
    "^GDAXI": "DAX",
    "^STOXX50E": "EURO STOXX 50",
    "^FTSE": "FTSE 100",
    "^N225": "NIKKEI 225",
    "^HSI": "HANG SENG",
    "^VIX": "VIX",
    "^TNX": "UST 10Y",
    "EURUSD=X": "EUR/USD",
    "USDJPY=X": "USD/JPY",
    "GC=F": "GOLD",
    "CL=F": "WTI CRUDE",
    "BTC-USD": "BITCOIN",
}

INFO_FIELDS = {
    "longName": "Name",
    "exchange": "Boerse",
    "currency": "Waehrung",
    "sector": "Sektor",
    "industry": "Branche",
    "country": "Land",
    "fullTimeEmployees": "Mitarbeiter",
    "marketCap": "Marktkap.",
    "enterpriseValue": "Enterprise Value",
    "trailingPE": "KGV (ttm)",
    "forwardPE": "KGV (fwd)",
    "priceToBook": "KBV",
    "enterpriseToEbitda": "EV/EBITDA",
    "trailingEps": "EPS (ttm)",
    "dividendYield": "Div.-Rendite %",
    "beta": "Beta",
    "fiftyTwoWeekHigh": "52W Hoch",
    "fiftyTwoWeekLow": "52W Tief",
    "averageVolume": "Ø Volumen",
    "profitMargins": "Nettomarge",
    "returnOnEquity": "ROE",
    "debtToEquity": "Debt/Equity",
    "targetMeanPrice": "Kursziel Ø",
    "recommendationKey": "Empfehlung",
    "numberOfAnalystOpinions": "Analysten",
}


def get_info(ticker: str) -> dict:
    """Stammdaten & Kennzahlen. Leeres dict, wenn Yahoo nichts liefert."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    # yfinance liefert fuer unbekannte Symbole ein fast leeres dict
    if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None
                    and info.get("previousClose") is None):
        return {}
    return info


def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    for n in (20, 50, 200):
        df[f"SMA{n}"] = df["Close"].rolling(n).mean()
    # RSI(14) nach Wilder
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / loss)
    return df


def get_monitor() -> pd.DataFrame:
    """Letzter Kurs, Veraenderung und %-Veraenderung fuer den Marktmonitor."""
    raw = yf.download(list(MONITOR), period="10d", interval="1d", progress=False,
                      auto_adjust=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    rows = []
    for sym, name in MONITOR.items():
        if sym not in close:
            continue
        s = close[sym].dropna()
        if len(s) < 2:
            continue
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        rows.append({"Symbol": sym, "Name": name, "Last": last,
                     "Chg": last - prev, "Pct": (last / prev - 1) * 100,
                     "Spark": s.tolist()})
    return pd.DataFrame(rows)


def stats_from_history(df: pd.DataFrame) -> dict:
    """Rendite-/Risikokennzahlen aus einer Tageskurs-Historie."""
    if df.empty or len(df) < 2:
        return {}
    ret = df["Close"].pct_change().dropna()
    peak = df["Close"].cummax()
    return {
        "Periodenrendite %": (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100,
        "Vola p.a. %": ret.std() * (252 ** 0.5) * 100,
        "Max Drawdown %": ((df["Close"] / peak - 1).min()) * 100,
        "RSI(14)": df["RSI14"].iloc[-1] if "RSI14" in df else float("nan"),
    }
