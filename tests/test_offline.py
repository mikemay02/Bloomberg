"""Offline-Tests der Datenlogik (ohne Netzwerk). Ausfuehren: python -m pytest tests"""
import pandas as pd

from terminal import fred, news, sec, sentiment


def _facts(tag, rows):
    return {"facts": {"us-gaap": {tag: {"units": {"USD": rows}}}}}


def test_annual_flow_uses_latest_filing_and_full_year_only():
    rows = [
        {"start": "2024-01-01", "end": "2024-12-31", "val": 100, "form": "10-K", "filed": "2025-02-01"},
        {"start": "2024-01-01", "end": "2024-12-31", "val": 110, "form": "10-K", "filed": "2026-02-01"},
        {"start": "2024-10-01", "end": "2024-12-31", "val": 30, "form": "10-K", "filed": "2025-02-01"},
    ]
    s = sec._series(_facts("Revenues", rows), ["Revenues"], annual=True, flow=True)
    assert list(s) == [110.0]  # Restatement gewinnt, Quartalsangabe im 10-K ignoriert


def test_quarterly_q4_is_derived_from_annual():
    q = [("2024-01-01", "2024-03-31", 10), ("2024-04-01", "2024-06-30", 20),
         ("2024-07-01", "2024-09-30", 30)]
    rows = [{"start": a, "end": b, "val": v, "form": "10-Q", "filed": b} for a, b, v in q]
    rows.append({"start": "2024-01-01", "end": "2024-12-31", "val": 100, "form": "10-K",
                 "filed": "2025-02-01"})
    s = sec._series(_facts("Revenues", rows), ["Revenues"], annual=False, flow=True)
    assert s[pd.Timestamp("2024-12-31")] == 40


def test_balance_instants():
    rows = [{"end": "2024-12-31", "val": 5, "form": "10-K", "filed": "2025-02-01"}]
    s = sec._series(_facts("Assets", rows), ["Assets"], annual=True, flow=False)
    assert list(s) == [5.0]


def test_google_title_split_and_short_name():
    assert news._clean_title("Nvidia beats estimates - Reuters", "Reuters") == \
        ("Nvidia beats estimates", "Reuters")
    assert news._clean_title("A - B", "Yahoo") == ("A - B", "Yahoo")
    assert news.short_name("NVIDIA Corporation") == "NVIDIA"
    assert news.short_name("Alphabet Inc.") == "Alphabet"


def test_yoy_transform_monthly():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    s = pd.Series([100.0] * 12 + [110.0] * 12, index=idx)
    assert round(fred.transform(s, "yoy").iloc[-1], 6) == 10.0


def test_sentiment_score_with_fake_pipeline():
    def fake(texts, batch_size):
        return [[{"label": "positive", "score": .8}, {"label": "negative", "score": .1},
                 {"label": "neutral", "score": .1}] for _ in texts]
    df = sentiment.score(fake, ["a", "b"])
    assert list(df["Label"]) == ["positive", "positive"]
    assert sentiment.summarize(df)["verdict"] == "BULLISH"
