"""Stimmungsbewertung von Schlagzeilen mit FinBERT (ProsusAI/finbert)."""
from __future__ import annotations

import pandas as pd

MODEL = "ProsusAI/finbert"


def load_model():
    """Laedt die FinBERT-Pipeline (beim ersten Mal ~440 MB Download von Hugging Face)."""
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL, tokenizer=MODEL, top_k=None,
                    truncation=True, max_length=128, device=-1)


def score(pipe, texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    """Je Text: P(positiv), P(negativ), P(neutral), Label und Score = P(pos) - P(neg)."""
    if not texts:
        return pd.DataFrame(columns=["positive", "negative", "neutral", "Label", "Score"])
    raw = pipe(list(texts), batch_size=batch_size)
    rows = [{d["label"].lower(): d["score"] for d in r} for r in raw]
    df = pd.DataFrame(rows, columns=["positive", "negative", "neutral"]).fillna(0.0)
    df["Label"] = df[["positive", "negative", "neutral"]].idxmax(axis=1)
    df["Score"] = df["positive"] - df["negative"]
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Aggregat: Durchschnittsscore, Anteile, Fazit."""
    if df.empty:
        return {"n": 0, "score": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0, "verdict": "—"}
    counts = df["Label"].value_counts(normalize=True)
    s = float(df["Score"].mean())
    verdict = ("BULLISH" if s > 0.15 else "LEICHT POSITIV" if s > 0.05 else
               "BEARISH" if s < -0.15 else "LEICHT NEGATIV" if s < -0.05 else "NEUTRAL")
    return {"n": len(df), "score": s, "pos": counts.get("positive", 0.0) * 100,
            "neg": counts.get("negative", 0.0) * 100,
            "neu": counts.get("neutral", 0.0) * 100, "verdict": verdict}
