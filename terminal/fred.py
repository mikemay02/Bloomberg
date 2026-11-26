"""Makrodaten aus FRED (Federal Reserve Bank of St. Louis).

Ohne API-Key wird der oeffentliche CSV-Export (fredgraph.csv) genutzt.
Ist FRED_API_KEY gesetzt, wird die offizielle API verwendet.
"""
from __future__ import annotations

import io
import os

import pandas as pd
import requests

# id -> (Anzeigename, Transformation, Einheit)
#   "level": Wert wie geliefert, "yoy": Veraenderung ggue. Vorjahr in %
SERIES = {
    "A191RL1Q225SBEA": ("US BIP-Wachstum (q/q ann.)", "level", "%"),
    "CPIAUCSL": ("US Inflation CPI", "yoy", "% y/y"),
    "PCEPILFE": ("US Kern-PCE", "yoy", "% y/y"),
    "UNRATE": ("US Arbeitslosenquote", "level", "%"),
    "PAYEMS": ("US Beschaeftigte (Nonfarm)", "level", "Tsd."),
    "ICSA": ("Erstantraege Arbeitslosenhilfe", "level", "Anz."),
    "FEDFUNDS": ("Fed Funds Rate", "level", "%"),
    "DGS2": ("US Treasury 2J", "level", "%"),
    "DGS10": ("US Treasury 10J", "level", "%"),
    "T10Y2Y": ("Zinskurve 10J-2J", "level", "%-Pkt."),
    "BAMLH0A0HYM2": ("High-Yield-Spread", "level", "%"),
    "VIXCLS": ("VIX", "level", "Pkt."),
    "DCOILWTICO": ("Oel WTI", "level", "USD"),
    "DEXUSEU": ("USD je EUR", "level", "USD"),
    "M2SL": ("Geldmenge M2", "yoy", "% y/y"),
    "UMCSENT": ("Verbrauchervertrauen (UMich)", "level", "Idx"),
}

# Laufzeiten fuer die Zinsstrukturkurve
CURVE = {"DGS1MO": "1M", "DGS3MO": "3M", "DGS6MO": "6M", "DGS1": "1J", "DGS2": "2J",
         "DGS3": "3J", "DGS5": "5J", "DGS7": "7J", "DGS10": "10J", "DGS20": "20J",
         "DGS30": "30J"}


def fetch(series_id: str, start: str = "2000-01-01") -> pd.Series:
    key = os.environ.get("FRED_API_KEY")
    if key:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": series_id, "api_key": key,
                                 "file_type": "json", "observation_start": start},
                         timeout=30)
        r.raise_for_status()
        obs = r.json()["observations"]
        s = pd.Series({o["date"]: o["value"] for o in obs})
    else:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": series_id, "cosd": start}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        s = df.set_index(df.columns[0])[df.columns[1]]
    s = pd.to_numeric(s, errors="coerce").dropna()  # "." = fehlender Wert
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s


def transform(s: pd.Series, how: str) -> pd.Series:
    if how != "yoy" or s.empty:
        return s
    freq = pd.infer_freq(s.index[-24:]) or ""
    lag = 12 if freq.startswith("M") else 4 if freq.startswith("Q") else 52 \
        if freq.startswith("W") else 12
    return (s / s.shift(lag) - 1).dropna() * 100


def snapshot(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Tabelle mit letztem Wert, Vorwert und Veraenderung je Reihe."""
    rows = []
    for sid, s in series.items():
        if s.empty:
            continue
        name, _, unit = SERIES.get(sid, (sid, "level", ""))
        last, prev = s.iloc[-1], s.iloc[-2] if len(s) > 1 else float("nan")
        rows.append({"Indikator": name, "FRED-ID": sid, "Stand": s.index[-1].date(),
                     "Wert": last, "Vorwert": prev, "Δ": last - prev, "Einheit": unit})
    return pd.DataFrame(rows)


def yield_curve() -> pd.DataFrame:
    """Aktuelle Kurve sowie Kurven vor 1 Monat und 1 Jahr."""
    data = {lbl: fetch(sid, start=str(pd.Timestamp.today().year - 2) + "-01-01")
            for sid, lbl in CURVE.items()}
    df = pd.DataFrame(data).ffill().dropna()
    if df.empty:
        return df
    t = df.index[-1]
    pick = {"Aktuell": t, "vor 1 Monat": t - pd.DateOffset(months=1),
            "vor 1 Jahr": t - pd.DateOffset(years=1)}
    out = {k: df.loc[:v].iloc[-1] for k, v in pick.items() if not df.loc[:v].empty}
    return pd.DataFrame(out)
