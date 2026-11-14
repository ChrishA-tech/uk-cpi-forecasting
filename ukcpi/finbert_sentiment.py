import math
from typing import Dict
from pydantic import BaseModel, Field
from .exceptions import CPIForecastValidationError


class FinBERTOutput(BaseModel):
    positive_prob: float = Field(..., ge=0.0, le=1.0, description="Inflationary / hawkish expansion probability")
    negative_prob: float = Field(..., ge=0.0, le=1.0, description="Disinflationary / contractionary probability")
    neutral_prob: float = Field(..., ge=0.0, le=1.0, description="Neutral monetary stance probability")
    inflation_sentiment_index: float = Field(..., ge=-1.0, le=1.0, description="Net inflation pressure score (pos - neg)")
    backend: str = Field(default="lexicon fallback (not FinBERT)", description="Which scorer produced this output")


class FinBERTSentimentAnalyzer:
    """
    Wraps ProsusAI/finbert for MPC statement sentiment extraction. If the model cannot be
    loaded (no `transformers`, no network), a keyword-lexicon fallback is used instead; it is
    NOT FinBERT, and every output reports which backend produced it (`backend`).
    Note: FinBERT's positive/negative labels are about financial sentiment, which is only a
    proxy for hawkish/dovish inflation pressure.
    """
    HAWKISH_INFLATIONARY_TERMS = [
        "persistent inflation", "upward pressure", "tightening", "elevated wage growth",
        "supply bottleneck", "second-round effects", "energy shock", "above target", "surge", "hawkish"
    ]
    DOVISH_DISINFLATIONARY_TERMS = [
        "cooling", "disinflation", "decelerating", "easing", "below target",
        "demand weakness", "falling energy prices", "rate cut", "dovish", "slack"
    ]

    def __init__(self, use_pretrained_hf: bool = False, model_name: str = "ProsusAI/finbert"):
        self.use_pretrained_hf = use_pretrained_hf
        self.model_name = model_name
        self._pipeline = None
        if self.use_pretrained_hf:
            try:
                from transformers import pipeline
                self._pipeline = pipeline("text-classification", model=self.model_name, top_k=None)
            except Exception:
                self._pipeline = None

    def score_text(self, text: str) -> FinBERTOutput:
        if not isinstance(text, str) or len(text.strip()) < 5:
            raise CPIForecastValidationError("Monetary policy text must be a string of at least 5 characters.")

        clean = text.strip()
        if self._pipeline is not None:
            preds = self._pipeline(clean[:512])
            if preds and isinstance(preds[0], list):      # older transformers: [[{label, score}, ...]]
                preds = preds[0]
            scores: Dict[str, float] = {item["label"].lower(): float(item["score"]) for item in preds}
            pos = scores.get("positive", 0.33)
            neg = scores.get("negative", 0.33)
            neu = scores.get("neutral", max(0.0, 1.0 - pos - neg))
        else:
            lower = clean.lower()
            hawk_hits = sum(1.2 for term in self.HAWKISH_INFLATIONARY_TERMS if term in lower)
            dove_hits = sum(1.2 for term in self.DOVISH_DISINFLATIONARY_TERMS if term in lower)

            logits = [hawk_hits, dove_hits, 0.8]
            exp_l = [math.exp(x) for x in logits]
            total = sum(exp_l)
            pos, neg, neu = exp_l[0] / total, exp_l[1] / total, exp_l[2] / total

        net_index = max(-1.0, min(1.0, round(pos - neg, 4)))
        return FinBERTOutput(
            backend=self.model_name if self._pipeline is not None else "lexicon fallback (not FinBERT)",
            positive_prob=round(pos, 4),
            negative_prob=round(neg, 4),
            neutral_prob=round(neu, 4),
            inflation_sentiment_index=net_index,
        )
