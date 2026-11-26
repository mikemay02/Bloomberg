"""Schlagzeilen aus Yahoo Finance, Google News und Reuters (RSS).

Reuters bietet seit 2020 keine oeffentlichen RSS-Feeds mehr an. Reuters-
Schlagzeilen werden deshalb ueber die Google-News-Suche mit
``site:reuters.com`` bezogen.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) BBTerminal/1.0"}


def _gnews(q: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"


_LEGAL = re.compile(r"[,.]?\s+(Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|Limited|plc|PLC|"
                    r"AG|SE|N\.V|NV|S\.A|SA|Holdings?|Group|Class [A-C])\.?$", re.I)


def short_name(company: str) -> str:
    """'NVIDIA Corporation' -> 'NVIDIA', 'Alphabet Inc.' -> 'Alphabet'."""
    name = company.strip()
    while True:
        new = _LEGAL.sub("", name).strip(" ,")
        if new == name or not new:
            return name
        name = new


def feeds_for(ticker: str, company: str | None = None) -> dict[str, str]:
    q = f'"{short_name(company)}"' if company else ticker
    return {
        "Yahoo": f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}"
                 "&region=US&lang=en-US",
        "Google": _gnews(f"{q} OR {ticker} stock when:7d"),
        "Reuters": _gnews(f"{q} site:reuters.com when:30d"),
    }


MARKET_FEEDS = {
    "Yahoo": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^DJI,^IXIC"
             "&region=US&lang=en-US",
    "Google": _gnews("stock market OR wall street OR federal reserve when:2d"),
    "Reuters": _gnews("markets site:reuters.com when:3d"),
}

_SUFFIX = re.compile(r"\s+-\s+[^-]{2,60}$")


def _clean_title(title: str, source: str) -> tuple[str, str]:
    """Google-News-Titel enden mit ' - Publisher'. Abtrennen und als Quelle liefern."""
    publisher = source
    if source in ("Google", "Reuters"):
        m = _SUFFIX.search(title)
        if m:
            publisher = m.group(0).strip(" -")
            title = title[: m.start()]
    return title.strip(), publisher


def _parse(source: str, url: str, limit: int) -> list[dict]:
    feed = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(0.8 * (attempt + 1))
            continue
        feed = feedparser.parse(r.content)
        if feed.entries:
            break
        time.sleep(0.8 * (attempt + 1))
    if feed is None:
        return []
    out = []
    for e in feed.entries[:limit]:
        title, publisher = _clean_title(e.get("title", ""), source)
        if not title:
            continue
        if source == "Reuters" and "reuters" not in publisher.lower():
            continue
        ts = e.get("published_parsed") or e.get("updated_parsed")
        when = datetime(*ts[:6], tzinfo=timezone.utc) if ts else None
        out.append({"Zeit": when, "Feed": source, "Quelle": publisher,
                    "Schlagzeile": title, "Link": e.get("link", "")})
    return out


def headlines(feeds: dict[str, str], limit_per_feed: int = 30) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=len(feeds)) as ex:
        results = ex.map(lambda kv: _parse(kv[0], kv[1], limit_per_feed), feeds.items())
    rows = [r for chunk in results for r in chunk]
    df = pd.DataFrame(rows, columns=["Zeit", "Feed", "Quelle", "Schlagzeile", "Link"])
    if df.empty:
        return df
    key = df["Schlagzeile"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    df = df.loc[~key.duplicated()]
    return df.sort_values("Zeit", ascending=False, na_position="last").reset_index(drop=True)
