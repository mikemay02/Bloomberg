"""Terminal im Bloomberg-Stil.

Start:  streamlit run streamlit_app.py

Befehlszeile wie im Original:  <TICKER> [<FUNKTION>] <GO>
    AAPL GP      Kurschart          AAPL FA     SEC-Finanzdaten
    AAPL DES     Beschreibung       AAPL CN     News + FinBERT-Stimmung
    ECO          Makro (FRED)       WEI         Weltmaerkte / Marktmonitor
    TOP          Marktnachrichten mit Stimmung
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from terminal import fred, market, news, sec, sentiment

st.set_page_config(page_title="TERMINAL", page_icon="▮", layout="wide",
                   initial_sidebar_state="collapsed")

AMBER, GREEN, RED, BLUE, GREY = "#FB8B1E", "#00D26A", "#FF433D", "#4AA8FF", "#8A8A8A"
FUNCTIONS = {
    "DES": "Profil",
    "GP": "Chart",
    "FA": "SEC-Finanzen",
    "CN": "News+Stimmung",
    "ECO": "Makro FRED",
    "WEI": "Weltmaerkte",
    "TOP": "Top News",
}

# --------------------------------------------------------------------------- Stil
st.markdown(f"""
<style>
  html, body, [class*="css"], .stMarkdown, .stDataFrame, input, button, label {{
      font-family: "IBM Plex Mono", "Consolas", "Menlo", monospace !important; }}
  .block-container {{ padding-top: 0.6rem; padding-bottom: 1rem; max-width: 100%; }}
  header[data-testid="stHeader"], div[data-testid="stToolbar"], div[data-testid="stDecoration"] {{ display:none; }}
  .bb-bar {{ background:{AMBER}; color:#000; font-weight:700; padding:3px 10px;
            display:flex; justify-content:space-between; letter-spacing:.05em; }}
  .bb-tape {{ white-space:nowrap; overflow:hidden; border-bottom:1px solid #333;
             padding:4px 0; font-size:.82rem; }}
  .bb-tape span {{ margin-right:22px; }}
  .bb-h {{ color:{AMBER}; font-weight:700; border-bottom:1px solid {AMBER};
          margin:10px 0 6px 0; padding-bottom:2px; letter-spacing:.06em; }}
  .bb-kv {{ display:grid; grid-template-columns: 1fr auto; gap:1px 12px; font-size:.85rem; }}
  .bb-kv .k {{ color:{AMBER}; }}  .bb-kv .v {{ color:#fff; text-align:right; }}
  .up {{ color:{GREEN}; }} .dn {{ color:{RED}; }} .am {{ color:{AMBER}; }} .mut {{ color:{GREY}; }}
  .bb-big {{ font-size:2.1rem; font-weight:700; color:#fff; line-height:1.1; }}
  .bb-news a {{ color:#fff; text-decoration:none; }} .bb-news a:hover {{ color:{AMBER}; }}
  .bb-news div {{ padding:3px 0; border-bottom:1px solid #1d1d1d; font-size:.86rem; }}
  .bb-pill {{ display:inline-block; min-width:62px; text-align:center; font-weight:700;
             padding:0 6px; margin-right:8px; color:#000; }}
  div[data-testid="stTextInput"] input {{ background:#000; color:{AMBER}; font-weight:700;
             border:1px solid {AMBER}; font-size:1.05rem; text-transform:uppercase; }}
  div[role="radiogroup"] label p {{ font-size:.8rem; }}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- Cache
@st.cache_data(ttl=60, show_spinner=False)
def c_monitor():
    return market.get_monitor()


@st.cache_data(ttl=600, show_spinner=False)
def c_info(t):
    return market.get_info(t)


@st.cache_data(ttl=300, show_spinner=False)
def c_hist(t, period, interval):
    return market.get_history(t, period, interval)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def c_fin(t, annual):
    return sec.financials(t, annual=annual, periods=8 if annual else 9)


@st.cache_data(ttl=3600, show_spinner=False)
def c_filings(t):
    return sec.recent_filings(t)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def c_fred(sid, start):
    return fred.fetch(sid, start)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def c_curve():
    return fred.yield_curve()


@st.cache_data(ttl=600, show_spinner=False)
def c_news(feeds: tuple):
    return news.headlines(dict(feeds))


@st.cache_resource(show_spinner="Lade FinBERT (ProsusAI/finbert) …")
def c_finbert():
    return sentiment.load_model()


@st.cache_data(ttl=3600, show_spinner=False)
def c_score(titles: tuple):
    return sentiment.score(c_finbert(), list(titles))


# --------------------------------------------------------------------------- Helfer
def fmt_num(v, dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, str):
        return v.upper()
    a = abs(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= div:
            return f"{v / div:,.{dec}f}{suf}"
    return f"{v:,.{dec}f}"


def chg_cls(v):
    return "up" if v > 0 else "dn" if v < 0 else "mut"


def header(text):
    st.markdown(f'<div class="bb-h">{text}</div>', unsafe_allow_html=True)


def kv_grid(pairs):
    html = "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in pairs)
    st.markdown(f'<div class="bb-kv">{html}</div>', unsafe_allow_html=True)


def style_fig(fig, h=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#000",
                      height=h, margin=dict(l=8, r=8, t=28, b=8),
                      font=dict(family="monospace", color="#ddd", size=11),
                      legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)"),
                      hovermode="x unified")
    fig.update_xaxes(gridcolor="#1c1c1c", zeroline=False, showspikes=True, spikecolor="#555")
    fig.update_yaxes(gridcolor="#1c1c1c", zeroline=False, side="right")
    return fig


def sent_color(score):
    return GREEN if score > 0.05 else RED if score < -0.05 else GREY


def parse_command(cmd: str):
    toks = [t for t in cmd.upper().replace("<GO>", "").split() if t not in ("US", "EQUITY")]
    fn = next((t for t in toks if t in FUNCTIONS), None)
    ticks = [t for t in toks if t not in FUNCTIONS]
    return (ticks[0] if ticks else None), fn


# --------------------------------------------------------------------------- Zustand
ss = st.session_state
ss.setdefault("ticker", "AAPL")
ss.setdefault("fn", "DES")


def on_cmd():
    t, fn = parse_command(ss.cmd)
    if t:
        ss.ticker = t
    if fn:
        ss.fn = fn
    elif t:
        ss.fn = "DES"
    ss.cmd = ""


# --------------------------------------------------------------------------- Kopf
now = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC").upper()
st.markdown(f'<div class="bb-bar"><span>▮ TERMINAL</span><span>{ss.ticker} US EQUITY · '
            f'{ss.fn} {FUNCTIONS[ss.fn].upper()}</span><span>{now}</span></div>',
            unsafe_allow_html=True)

try:
    mon = c_monitor()
    tape = "".join(
        f'<span><b class="am">{r.Name}</b> {r.Last:,.2f} '
        f'<span class="{chg_cls(r.Pct)}">{"▲" if r.Pct >= 0 else "▼"} {r.Pct:+.2f}%</span></span>'
        for r in mon.itertuples())
    st.markdown(f'<div class="bb-tape">{tape}</div>', unsafe_allow_html=True)
except Exception as e:  # Marktmonitor ist optional
    mon = pd.DataFrame()
    st.caption(f"Marktmonitor nicht verfuegbar: {e}")

c1, c2 = st.columns([2, 6])
with c1:
    st.text_input("Befehl", key="cmd", on_change=on_cmd, label_visibility="collapsed",
                  placeholder="z.B. MSFT FA  ·  NVDA CN  ·  ECO  <ENTER>")
with c2:
    st.radio("Funktion", list(FUNCTIONS), key="fn", horizontal=True,
             format_func=lambda f: f"{f} {FUNCTIONS[f]}", label_visibility="collapsed")

T = ss.ticker
FN = ss.fn


# =========================================================================== DES
def page_des():
    info = c_info(T)
    if not info:
        st.error(f"Keine Kursdaten fuer '{T}' bei Yahoo Finance gefunden.")
        return
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    prev = info.get("previousClose") or price
    chg = price - prev
    pct = chg / prev * 100 if prev else 0
    left, mid, right = st.columns([3, 4, 3])
    with left:
        st.markdown(f'<div class="am"><b>{info.get("longName", T)}</b></div>'
                    f'<div class="bb-big">{price:,.2f} <span class="mut" style="font-size:1rem">'
                    f'{info.get("currency", "")}</span></div>'
                    f'<div class="{chg_cls(chg)}" style="font-size:1.1rem">{chg:+,.2f} '
                    f'({pct:+.2f}%)</div>', unsafe_allow_html=True)
        header("KENNZAHLEN")
        pairs = []
        for k, lbl in market.INFO_FIELDS.items():
            if k in ("longName",) or info.get(k) is None:
                continue
            v = info[k]
            if k in ("profitMargins", "returnOnEquity"):
                v = f"{v * 100:.1f}%"
            elif k == "dividendYield":
                v = f"{v:.2f}%"
            elif k == "fullTimeEmployees":
                v = f"{v:,}"
            else:
                v = fmt_num(v)
            pairs.append((lbl, v))
        kv_grid(pairs)
    with mid:
        h = c_hist(T, "1y", "1d")
        if not h.empty:
            col = GREEN if h["Close"].iloc[-1] >= h["Close"].iloc[0] else RED
            fig = go.Figure(go.Scatter(x=h.index, y=h["Close"], line=dict(color=col, width=1.6),
                                       fill="tozeroy", fillcolor="rgba(251,139,30,0.06)",
                                       name="Close"))
            fig.update_yaxes(range=[h["Close"].min() * 0.95, h["Close"].max() * 1.02])
            header("1 JAHR")
            st.plotly_chart(style_fig(fig, 330), use_container_width=True)
            stats = market.stats_from_history(h)
            kv_grid([(k, f"{v:,.2f}") for k, v in stats.items()])
    with right:
        header("BESCHREIBUNG")
        st.markdown(f'<div style="font-size:.82rem;color:#ccc;max-height:520px;overflow:auto">'
                    f'{info.get("longBusinessSummary", "—")}</div>', unsafe_allow_html=True)
        if info.get("website"):
            st.markdown(f'<div class="mut" style="font-size:.8rem">{info["website"]}</div>',
                        unsafe_allow_html=True)


# =========================================================================== GP
def page_gp():
    a, b, c, d = st.columns([3, 2, 3, 3])
    period = a.radio("Zeitraum", ["1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "max"], index=4,
                     horizontal=True)
    interval = b.selectbox("Intervall", ["1d", "1wk", "1mo"], index=0)
    style = c.radio("Darstellung", ["Kerzen", "Linie"], horizontal=True)
    overlays = d.multiselect("Overlays", ["SMA20", "SMA50", "SMA200"], default=["SMA50", "SMA200"])
    h = c_hist(T, period, interval)
    if h.empty:
        st.error(f"Keine Kurshistorie fuer {T}.")
        return
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.62, 0.18, 0.20])
    if style == "Kerzen":
        fig.add_trace(go.Candlestick(x=h.index, open=h.Open, high=h.High, low=h.Low,
                                     close=h.Close, name=T, increasing_line_color=GREEN,
                                     decreasing_line_color=RED), 1, 1)
    else:
        fig.add_trace(go.Scatter(x=h.index, y=h.Close, name=T, line=dict(color=AMBER)), 1, 1)
    for o, col in zip(("SMA20", "SMA50", "SMA200"), (BLUE, AMBER, "#C77DFF")):
        if o in overlays:
            fig.add_trace(go.Scatter(x=h.index, y=h[o], name=o, line=dict(color=col, width=1)),
                          1, 1)
    vol_col = np.where(h.Close >= h.Open, GREEN, RED)
    fig.add_trace(go.Bar(x=h.index, y=h.Volume, marker_color=vol_col, name="Volumen",
                         opacity=0.6), 2, 1)
    fig.add_trace(go.Scatter(x=h.index, y=h.RSI14, name="RSI14", line=dict(color=BLUE, width=1)),
                  3, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(color="#444", dash="dot"), row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(style_fig(fig, 640), use_container_width=True)
    stats = market.stats_from_history(h)
    cols = st.columns(len(stats) + 3)
    cols[0].metric("Letzter", f"{h.Close.iloc[-1]:,.2f}")
    cols[1].metric("Hoch", f"{h.High.max():,.2f}")
    cols[2].metric("Tief", f"{h.Low.min():,.2f}")
    for col, (k, v) in zip(cols[3:], stats.items()):
        col.metric(k, f"{v:,.2f}")


# =========================================================================== FA
def fin_table(df: pd.DataFrame, scale=1e6, pct=False):
    if df.empty:
        st.caption("Keine Daten.")
        return
    shown = df.copy()
    if not pct:
        per_share = [i for i in shown.index if "EPS" in i]
        rest = [i for i in shown.index if i not in per_share]
        shown.loc[rest] = shown.loc[rest] / scale
    shown = shown.iloc[:, ::-1]  # neueste zuerst wie im Terminal
    styler = shown.style.format("{:,.2f}" if pct else "{:,.0f}", na_rep="—")
    eps = [i for i in shown.index if "EPS" in i]
    if eps and not pct:
        styler = styler.format("{:,.2f}", subset=pd.IndexSlice[eps, :], na_rep="—")
    st.dataframe(styler.map(lambda v: f"color:{RED}" if isinstance(v, (int, float)) and v < 0
                            else ""),
                 use_container_width=True, height=(len(shown) + 1) * 35 + 3)


def page_fa():
    a, b = st.columns([2, 6])
    annual = a.radio("Periode", ["Jaehrlich", "Quartal"], horizontal=True) == "Jaehrlich"
    try:
        with st.spinner("Lade XBRL-Daten von SEC EDGAR …"):
            f = c_fin(T, annual)
    except LookupError as e:
        st.error(str(e) + " FA funktioniert nur fuer US-gelistete SEC-Filer.")
        return
    except Exception as e:
        st.error(f"SEC EDGAR nicht erreichbar: {e}")
        return
    b.markdown(f'<div class="am" style="padding-top:8px">{f["name"]} · CIK {f["cik"]:010d} · '
               f'Quelle: SEC EDGAR XBRL · Werte in Mio. USD (EPS in USD)</div>',
               unsafe_allow_html=True)

    inc, bal, cfs = f["income"], f["balance"], f["cashflow"]
    t1, t2, t3, t4, t5 = st.tabs(["GuV", "Bilanz", "Cashflow", "Kennzahlen", "Filings"])
    with t1:
        fin_table(inc)
        if {"Umsatz", "Nettogewinn"} <= set(inc.index):
            fig = go.Figure()
            fig.add_bar(x=inc.columns, y=inc.loc["Umsatz"] / 1e9, name="Umsatz (Mrd)",
                        marker_color=AMBER)
            if "Operatives Ergebnis" in inc.index:
                fig.add_bar(x=inc.columns, y=inc.loc["Operatives Ergebnis"] / 1e9,
                            name="Op. Ergebnis (Mrd)", marker_color=BLUE)
            fig.add_bar(x=inc.columns, y=inc.loc["Nettogewinn"] / 1e9, name="Nettogewinn (Mrd)",
                        marker_color=GREEN)
            fig.update_layout(barmode="group")
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with t2:
        fin_table(bal)
        if {"Verbindlichkeiten ges.", "Eigenkapital"} <= set(bal.index):
            fig = go.Figure()
            fig.add_bar(x=bal.columns, y=bal.loc["Eigenkapital"] / 1e9, name="Eigenkapital",
                        marker_color=GREEN)
            fig.add_bar(x=bal.columns, y=bal.loc["Verbindlichkeiten ges."] / 1e9,
                        name="Verbindlichkeiten", marker_color=RED)
            fig.update_layout(barmode="stack", title="Passivseite (Mrd USD)")
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with t3:
        fin_table(cfs)
        if "Free Cashflow" in cfs.index:
            fig = go.Figure(go.Bar(x=cfs.columns, y=cfs.loc["Free Cashflow"] / 1e9,
                                   marker_color=[GREEN if v >= 0 else RED
                                                 for v in cfs.loc["Free Cashflow"]],
                                   name="FCF"))
            fig.update_layout(title="Free Cashflow (Mrd USD)")
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    with t4:
        fin_table(f["ratios"], pct=True)
    with t5:
        try:
            fl = c_filings(T)
            st.dataframe(fl, use_container_width=True, hide_index=True,
                         column_config={"URL": st.column_config.LinkColumn("Dokument",
                                                                           display_text="oeffnen")})
        except Exception as e:
            st.error(f"Filings nicht verfuegbar: {e}")
    if not annual:
        st.caption("Q4-Werte von GuV/Cashflow sind abgeleitet: Geschaeftsjahr (10-K) minus Q1–Q3.")


# =========================================================================== ECO
def page_eco():
    a, b = st.columns([3, 5])
    start = a.radio("Ab", ["2000", "2010", "2015", "2020", "2023"], index=3, horizontal=True)
    chosen = b.multiselect("Indikatoren", list(fred.SERIES), default=list(fred.SERIES),
                           format_func=lambda s: fred.SERIES[s][0])
    # YoY-Reihen brauchen ein Jahr Vorlauf
    fetch_start = f"{int(start) - 1}-01-01"
    data, errors = {}, []
    with st.spinner("Lade FRED-Daten …"):
        for sid in chosen:
            try:
                s = fred.transform(c_fred(sid, fetch_start), fred.SERIES[sid][1])
                data[sid] = s[s.index >= f"{start}-01-01"]
            except Exception as e:
                errors.append(f"{sid}: {e}")
    if errors:
        st.warning("Nicht geladen: " + "; ".join(errors))

    snap = fred.snapshot(data)
    if not snap.empty:
        header("AKTUELLE WERTE")
        st.dataframe(snap.style.format({"Wert": "{:,.2f}", "Vorwert": "{:,.2f}", "Δ": "{:+,.2f}"})
                     .map(lambda v: f"color:{GREEN}" if isinstance(v, float) and v > 0 else
                          f"color:{RED}" if isinstance(v, float) and v < 0 else "", subset=["Δ"]),
                     use_container_width=True, hide_index=True, height=(len(snap) + 1) * 35 + 3)

    header("ZEITREIHEN")
    ids = list(data)
    for i in range(0, len(ids), 4):
        cols = st.columns(4)
        for col, sid in zip(cols, ids[i:i + 4]):
            s = data[sid]
            name, _, unit = fred.SERIES[sid]
            fig = go.Figure(go.Scatter(x=s.index, y=s.values, line=dict(color=AMBER, width=1.3),
                                       name=sid))
            if s.min() < 0 < s.max():
                fig.add_hline(y=0, line=dict(color="#555", dash="dot"))
            fig.update_layout(title=dict(text=f"{name} [{unit}]", font=dict(size=11)),
                              showlegend=False)
            col.plotly_chart(style_fig(fig, 220), use_container_width=True)

    header("US-ZINSSTRUKTURKURVE")
    try:
        curve = c_curve()
        fig = go.Figure()
        for name, col in zip(curve.columns, (AMBER, BLUE, GREY)):
            fig.add_scatter(x=curve.index, y=curve[name], name=name, mode="lines+markers",
                            line=dict(color=col, width=2 if name == "Aktuell" else 1))
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    except Exception as e:
        st.warning(f"Zinskurve nicht verfuegbar: {e}")
    st.caption("Quelle: Federal Reserve Bank of St. Louis (FRED).")


# =========================================================================== CN / TOP
def news_page(feeds: dict, title: str):
    a, b = st.columns([3, 5])
    use_fb = a.toggle("FinBERT-Stimmung", value=True)
    srcs = b.multiselect("Feeds", list(feeds), default=list(feeds))
    with st.spinner("Lade Schlagzeilen …"):
        df = c_news(tuple((k, v) for k, v in feeds.items() if k in srcs))
    if df.empty:
        st.warning("Keine Schlagzeilen gefunden.")
        return
    if use_fb:
        try:
            with st.spinner("FinBERT bewertet Schlagzeilen …"):
                sc = c_score(tuple(df["Schlagzeile"]))
            df = pd.concat([df, sc], axis=1)
        except Exception as e:
            st.error(f"FinBERT nicht verfuegbar ({e}). Zeige Schlagzeilen ohne Stimmung.")
            use_fb = False

    if use_fb:
        summ = sentiment.summarize(df)
        header(f"STIMMUNG · {title}")
        g, s1, s2 = st.columns([2, 3, 3])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=summ["score"], number=dict(valueformat="+.2f"),
            title=dict(text=summ["verdict"], font=dict(color=sent_color(summ["score"]))),
            gauge=dict(axis=dict(range=[-1, 1]), bar=dict(color=sent_color(summ["score"])),
                       steps=[dict(range=[-1, -0.05], color="#2a0d0c"),
                              dict(range=[-0.05, 0.05], color="#1a1a1a"),
                              dict(range=[0.05, 1], color="#0b2a17")])))
        gauge = style_fig(gauge, 250)
        gauge.update_layout(margin=dict(l=20, r=20, t=60, b=0))
        g.plotly_chart(gauge, use_container_width=True)

        by_feed = df.groupby("Feed").agg(Score=("Score", "mean"), N=("Score", "size")).reset_index()
        fig = go.Figure(go.Bar(x=by_feed.Feed, y=by_feed.Score, text=by_feed.N,
                               texttemplate="n=%{text}", textposition="outside",
                               marker_color=[sent_color(v) for v in by_feed.Score]))
        fig.update_layout(title="Ø Score je Quelle")
        fig.update_yaxes(range=[-1, 1])
        s1.plotly_chart(style_fig(fig, 230), use_container_width=True)

        daily = (df.dropna(subset=["Zeit"]).set_index("Zeit")["Score"]
                 .resample("D").agg(["mean", "size"]))
        daily = daily[daily["size"] > 0]
        fig = go.Figure(go.Bar(x=daily.index, y=daily["mean"],
                               marker_color=[sent_color(v) for v in daily["mean"]]))
        fig.update_layout(title="Ø Score je Tag")
        fig.update_yaxes(range=[-1, 1])
        s2.plotly_chart(style_fig(fig, 230), use_container_width=True)
        kv_grid([("Schlagzeilen", summ["n"]), ("Positiv", f'{summ["pos"]:.0f}%'),
                 ("Neutral", f'{summ["neu"]:.0f}%'), ("Negativ", f'{summ["neg"]:.0f}%')])

    header(f"SCHLAGZEILEN ({len(df)})")
    lines = []
    for r in df.itertuples():
        when = r.Zeit.strftime("%d.%m %H:%M") if pd.notna(r.Zeit) else "—"
        pill = ""
        if use_fb:
            pill = (f'<span class="bb-pill" style="background:{sent_color(r.Score)}">'
                    f'{r.Score:+.2f}</span>')
        lines.append(f'<div>{pill}<span class="mut">{when}</span> '
                     f'<span class="am">[{r.Feed if r.Feed == r.Quelle else r.Feed + "·" + r.Quelle}]'
                     f'</span> '
                     f'<a href="{r.Link}" target="_blank">{r.Schlagzeile}</a></div>')
    st.markdown(f'<div class="bb-news">{"".join(lines)}</div>', unsafe_allow_html=True)
    if use_fb:
        st.caption("Score = P(positiv) − P(negativ) laut FinBERT (ProsusAI/finbert), "
                   "Bereich −1 … +1. Reuters-Meldungen via Google News (site:reuters.com).")


def page_cn():
    info = c_info(T)
    news_page(news.feeds_for(T, info.get("longName") if info else None), T)


def page_top():
    news_page(news.MARKET_FEEDS, "MARKT")


# =========================================================================== WEI
def page_wei():
    if mon.empty:
        st.error("Marktmonitor nicht verfuegbar.")
        return
    header("WELTMAERKTE · MONITOR")
    cols = st.columns(5)
    for i, r in enumerate(mon.itertuples()):
        with cols[i % 5]:
            st.metric(r.Name, f"{r.Last:,.2f}", f"{r.Pct:+.2f}%")
            spark = go.Figure(go.Scatter(y=r.Spark, line=dict(
                color=GREEN if r.Spark[-1] >= r.Spark[0] else RED, width=1.4)))
            spark.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=50, showlegend=False,
                                paper_bgcolor="#000", plot_bgcolor="#000")
            spark.update_xaxes(visible=False)
            spark.update_yaxes(visible=False)
            st.plotly_chart(spark, use_container_width=True, config={"displayModeBar": False},
                            key=f"spark{i}")
    header("TAGESVERAENDERUNG %")
    m = mon.sort_values("Pct")
    fig = go.Figure(go.Bar(x=m.Pct, y=m.Name, orientation="h",
                           marker_color=[GREEN if v >= 0 else RED for v in m.Pct]))
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)


PAGES = {"DES": page_des, "GP": page_gp, "FA": page_fa, "CN": page_cn, "ECO": page_eco,
         "WEI": page_wei, "TOP": page_top}
PAGES[FN]()

st.markdown('<div class="mut" style="font-size:.72rem;margin-top:14px;border-top:1px solid #222;'
            'padding-top:4px">Daten: Yahoo Finance (yfinance) · SEC EDGAR · FRED · Yahoo/Google '
            'News · Reuters via Google News · Stimmung: FinBERT. Kein Anlagerat. Kurse ggf. '
            'verzoegert.</div>', unsafe_allow_html=True)
