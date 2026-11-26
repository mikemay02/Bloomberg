"""Bilanz-, GuV- und Cashflow-Daten aus SEC EDGAR (XBRL companyfacts API).

Die SEC verlangt einen User-Agent mit Kontaktangabe. Setze dafuer die
Umgebungsvariable SEC_USER_AGENT, z. B. "Max Muster max@example.com".
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests

UA = os.environ.get("SEC_USER_AGENT", "BBTerminal research-app contact@example.com")
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}

# Anzeigename -> Liste moeglicher US-GAAP-Tags (erster Treffer mit Daten gewinnt)
INCOME = {
    "Umsatz": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
               "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "Herstellkosten": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "Bruttogewinn": ["GrossProfit"],
    "F&E": ["ResearchAndDevelopmentExpense"],
    "Operatives Ergebnis": ["OperatingIncomeLoss"],
    "Nettogewinn": ["NetIncomeLoss", "ProfitLoss"],
    "EPS verwaessert": ["EarningsPerShareDiluted"],
}
BALANCE = {
    "Liquide Mittel": ["CashAndCashEquivalentsAtCarryingValue",
                       "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "Umlaufvermoegen": ["AssetsCurrent"],
    "Bilanzsumme": ["Assets"],
    "Kurzfr. Verbindlichk.": ["LiabilitiesCurrent"],
    "Langfr. Schulden": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "Verbindlichkeiten ges.": ["Liabilities"],
    "Eigenkapital": ["StockholdersEquity",
                     "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}
CASHFLOW = {
    "Operativer Cashflow": ["NetCashProvidedByUsedInOperatingActivities"],
    "Capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "Dividenden": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "Aktienrueckkaeufe": ["PaymentsForRepurchaseOfCommonStock"],
}

ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F"}
QUARTER_FORMS = {"10-Q", "10-Q/A"}


def _get(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def ticker_map() -> dict[str, dict]:
    """Ticker -> {cik, title} fuer alle bei der SEC registrierten Emittenten."""
    data = _get("https://www.sec.gov/files/company_tickers.json")
    return {v["ticker"].upper(): {"cik": int(v["cik_str"]), "title": v["title"]}
            for v in data.values()}


def lookup_cik(ticker: str, mapping: dict[str, dict] | None = None) -> int | None:
    mapping = mapping or ticker_map()
    t = ticker.upper().replace(".", "-")
    hit = mapping.get(t) or mapping.get(ticker.upper())
    return hit["cik"] if hit else None


def company_facts(cik: int) -> dict:
    return _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")


def submissions(cik: int) -> dict:
    return _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _series(facts: dict, tags: list[str], annual: bool, flow: bool) -> pd.Series:
    """Zeitreihe fuer das erste Tag, das Werte im gewuenschten Rhythmus hat.

    flow=True: Zeitraumgroessen (GuV, Cashflow) – Dauer muss ~1 Jahr bzw. ~1 Quartal sein.
    flow=False: Stichtagsgroessen (Bilanz).
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    ifrs = facts.get("facts", {}).get("ifrs-full", {})
    forms = ANNUAL_FORMS if annual else (QUARTER_FORMS | ANNUAL_FORMS)
    for tag in tags:
        node = gaap.get(tag) or ifrs.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        unit = next((u for u in ("USD", "USD/shares", "EUR", "EUR/shares") if u in units),
                    next(iter(units), None))
        if unit is None:
            continue
        rows = []
        for f in units[unit]:
            if f.get("form") not in forms or "end" not in f:
                continue
            if flow:
                if "start" not in f:
                    continue
                d = _days(f["start"], f["end"])
                if annual and not 350 <= d <= 380:
                    continue
                if not annual and not 80 <= d <= 100:
                    continue
            rows.append((f["end"], f.get("filed", ""), f["val"]))
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["end", "filed", "val"])
        # pro Stichtag die zuletzt eingereichte Angabe (beruecksichtigt Restatements)
        df = df.sort_values("filed").groupby("end").last()
        s = df["val"].astype(float)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        if flow and not annual:
            s = _fill_q4(s, _series(facts, [tag], annual=True, flow=True))
        return s
    return pd.Series(dtype=float)


def _fill_q4(quarters: pd.Series, years: pd.Series) -> pd.Series:
    """Q4 steht nur im 10-K (als Jahreswert). Q4 = Geschaeftsjahr - Q1..Q3."""
    extra = {}
    for end, total in years.items():
        if end in quarters.index:
            continue
        q = quarters[(quarters.index > end - pd.Timedelta(days=370)) & (quarters.index < end)]
        if len(q) == 3:
            extra[end] = total - q.sum()
    if not extra:
        return quarters
    return pd.concat([quarters, pd.Series(extra)]).sort_index()


def statement(facts: dict, items: dict[str, list[str]], annual: bool = True,
              flow: bool = True, periods: int = 8) -> pd.DataFrame:
    """Tabelle: Zeilen = Positionen, Spalten = Stichtage (neueste rechts)."""
    data = {name: _series(facts, tags, annual, flow) for name, tags in items.items()}
    df = pd.DataFrame(data).T
    if df.empty:
        return df
    df = df[sorted(df.columns)[-periods:]]
    df.columns = [c.strftime("%Y-%m-%d") for c in df.columns]
    return df.dropna(how="all")


def financials(ticker: str, annual: bool = True, periods: int = 8) -> dict:
    """Komplettpaket fuer die FA-Funktion."""
    mapping = ticker_map()
    cik = lookup_cik(ticker, mapping)
    if cik is None:
        raise LookupError(f"{ticker} ist nicht bei der SEC registriert (kein CIK gefunden).")
    facts = company_facts(cik)
    inc = statement(facts, INCOME, annual, flow=True, periods=periods)
    bal = statement(facts, BALANCE, annual, flow=False, periods=periods)
    cfs = statement(facts, CASHFLOW, annual, flow=True, periods=periods)
    if "Operativer Cashflow" in cfs.index and "Capex" in cfs.index:
        cfs.loc["Free Cashflow"] = cfs.loc["Operativer Cashflow"] - cfs.loc["Capex"]
    return {"cik": cik, "name": facts.get("entityName", ticker), "income": inc,
            "balance": bal, "cashflow": cfs, "ratios": ratios(inc, bal)}


def ratios(inc: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    """Einfache Kennzahlen aus GuV und Bilanz (nur gemeinsame Stichtage)."""
    cols = [c for c in inc.columns if c in bal.columns]
    out = {}

    def row(df, name):
        return df.loc[name, cols] if name in df.index else None

    rev, gp, op, ni = (row(inc, n) for n in ("Umsatz", "Bruttogewinn",
                                              "Operatives Ergebnis", "Nettogewinn"))
    ca, cl, eq, ta, tl = (row(bal, n) for n in ("Umlaufvermoegen", "Kurzfr. Verbindlichk.",
                                                 "Eigenkapital", "Bilanzsumme",
                                                 "Verbindlichkeiten ges."))
    if rev is not None:
        if gp is not None:
            out["Bruttomarge %"] = gp / rev * 100
        if op is not None:
            out["Operative Marge %"] = op / rev * 100
        if ni is not None:
            out["Nettomarge %"] = ni / rev * 100
    if ni is not None and eq is not None:
        out["ROE %"] = ni / eq * 100
    if ni is not None and ta is not None:
        out["ROA %"] = ni / ta * 100
    if ca is not None and cl is not None:
        out["Current Ratio"] = ca / cl
    if eq is not None and ta is not None:
        out["EK-Quote %"] = eq / ta * 100
    if tl is not None and eq is not None:
        out["Verschuldungsgrad"] = tl / eq
    return pd.DataFrame(out).T if out else pd.DataFrame()


def recent_filings(ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K", "20-F", "6-K",
                                                          "DEF 14A", "S-1", "4"),
                   limit: int = 25) -> pd.DataFrame:
    cik = lookup_cik(ticker)
    if cik is None:
        return pd.DataFrame()
    rec = submissions(cik)["filings"]["recent"]
    df = pd.DataFrame({k: rec[k] for k in ("form", "filingDate", "reportDate",
                                           "accessionNumber", "primaryDocument",
                                           "primaryDocDescription")})
    df = df[df["form"].isin(forms)].head(limit).copy()
    df["URL"] = [
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{a.replace('-', '')}/{d}"
        for a, d in zip(df["accessionNumber"], df["primaryDocument"])
    ]
    return df.rename(columns={"form": "Form", "filingDate": "Eingereicht",
                              "reportDate": "Periode", "primaryDocDescription": "Beschreibung"}
                     )[["Form", "Eingereicht", "Periode", "Beschreibung", "URL"]]
