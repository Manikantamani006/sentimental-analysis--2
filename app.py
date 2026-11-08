#!/usr/bin/env python3
"""
Sentima AI — FastAPI Backend Server for Fine-Tuned DistilBERT & Stitch UI (`page.html`)
======================================================================================
Serves `page.html` at `http://127.0.0.1:8000` and powers all interactive features:
  - `POST /api/analyze`: 6-class DistilBERT Softmax emotion decomposition, sentence
    polarity flow, aspect sentiment extraction, subjectivity index, lexical drivers,
    and tone telemetry.
  - `POST /api/batch`: Batch analysis for CSV / semicolon-delimited (`train.txt`/`test.txt`)
    files uploaded in the UI.
  - `GET /api/history`: Returns recent analysis runs stored in SQLite (`artifacts/analysis_history.db`).
  - `GET /api/health`: Health & loaded model status check.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from transformer_emotion_classifier import (
    DEFAULT_EMOTION_MAPPING,
    clean_text_for_transformer,
)


EMOTION_POLARITY_WEIGHT: Dict[str, float] = {
    "joy": 0.92,
    "love": 0.88,
    "surprise": 0.25,
    "sadness": -0.78,
    "fear": -0.72,
    "anger": -0.86,
}


def split_into_sentences(text: str) -> List[str]:
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    sentences = [s.strip() for s in raw_sentences if s and len(s.strip()) > 1]
    return sentences if sentences else [text.strip()]


class DistilBertEmotionService:
    """
    Loads your locally fine-tuned DistilBERT checkpoint from
    `artifacts/pretrained_transformer_emotion` (Option B) or falls back to
    `bhadresh-savani/distilbert-base-uncased-emotion` if training has not been run yet.
    """

    def __init__(
        self,
        local_model_dir: str = os.path.join("artifacts", "pretrained_transformer_emotion"),
        fallback_hub_model: str = "bhadresh-savani/distilbert-base-uncased-emotion",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model_dir = local_model_dir
        self.fallback_hub_model = fallback_hub_model
        self.model_name_loaded = "Uninitialized"
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if os.path.isdir(self.local_model_dir) and os.path.exists(
            os.path.join(self.local_model_dir, "config.json")
        ):
            target = self.local_model_dir
            self.model_name_loaded = "DistilBERT (Local Fine-Tuned)"
        else:
            target = self.fallback_hub_model
            self.model_name_loaded = "DistilBERT v2.0 (6-Emotion)"

        print(f"[Sentima AI] Loading Transformer model from '{target}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(target)
        self.model = AutoModelForSequenceClassification.from_pretrained(target).to(
            self.device
        )
        self.model.eval()

    def predict_probs(self, text: str) -> Dict[str, float]:
        cleaned = clean_text_for_transformer(text) or text
        inputs = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        id2label = getattr(self.model.config, "id2label", DEFAULT_EMOTION_MAPPING)
        result: Dict[str, float] = {
            emo: 0.0 for emo in DEFAULT_EMOTION_MAPPING.values()
        }
        for idx, p in enumerate(probs):
            raw_label = str(id2label.get(idx, DEFAULT_EMOTION_MAPPING.get(idx, "joy"))).lower()
            if raw_label in result:
                result[raw_label] = float(p)
        return result

    def analyze_corpus(self, text: str) -> Dict[str, Any]:
        sentences = split_into_sentences(text)
        sentence_results = []
        agg_emotions: Dict[str, float] = {
            emo: 0.0 for emo in DEFAULT_EMOTION_MAPPING.values()
        }

        pos_count = 0
        neg_count = 0
        neu_count = 0

        for sent in sentences:
            probs = self.predict_probs(sent)
            top_emotion = max(probs, key=probs.get)
            top_conf = probs[top_emotion]

            for emo, val in probs.items():
                agg_emotions[emo] += val / len(sentences)

            polarity = sum(
                probs[emo] * EMOTION_POLARITY_WEIGHT.get(emo, 0.0) for emo in probs
            )
            if polarity > 0.16:
                category = "pos"
                pos_count += 1
            elif polarity < -0.16:
                category = "neg"
                neg_count += 1
            else:
                category = "neu"
                neu_count += 1

            sentence_results.append(
                {
                    "text": sent,
                    "emotion": top_emotion,
                    "confidence": round(top_conf, 4),
                    "polarity": round(polarity, 2),
                    "category": category,
                    "probabilities": {k: round(v, 4) for k, v in probs.items()},
                }
            )

        overall_polarity = round(
            sum(s["polarity"] for s in sentence_results) / max(len(sentence_results), 1),
            2,
        )
        dominant_emotion = max(agg_emotions, key=agg_emotions.get)
        overall_confidence = round(agg_emotions[dominant_emotion] * 100, 1)

        # Extract top positive & critical lexical drivers
        words = [
            w
            for w in re.findall(r"[a-zA-Z']{4,}", text.lower())
            if w
            not in {
                "that",
                "this",
                "with",
                "from",
                "have",
                "were",
                "been",
                "their",
                "within",
                "across",
                "when",
                "once",
                "about",
                "there",
            }
        ]
        unique_words = list(dict.fromkeys(words))[:16]
        pos_drivers = []
        neg_drivers = []
        for token in unique_words:
            tok_probs = self.predict_probs(token)
            tok_pol = sum(
                tok_probs[e] * EMOTION_POLARITY_WEIGHT.get(e, 0.0) for e in tok_probs
            )
            if tok_pol >= 0.15:
                pos_drivers.append({"term": token, "weight": round(tok_pol, 2)})
            elif tok_pol <= -0.15:
                neg_drivers.append({"term": token, "weight": round(tok_pol, 2)})

        pos_drivers = sorted(pos_drivers, key=lambda x: x["weight"], reverse=True)[:3]
        neg_drivers = sorted(neg_drivers, key=lambda x: x["weight"])[:3]

        # Extract Dynamic Aspect Sentiment from noun/topic phrases in sentences
        aspects = []
        for s in sentence_results[:4]:
            tokens = [
                w.capitalize()
                for w in re.findall(r"[a-zA-Z]{4,}", s["text"])
                if w.lower() not in {"this", "that", "with", "from", "were", "have", "been", "very", "feel", "felt"}
            ]
            aspect_label = " ".join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else "Clause Focus")
            aspects.append(
                {
                    "name": aspect_label,
                    "score": s["polarity"],
                    "category": s["category"],
                }
            )

        total_sents = max(len(sentence_results), 1)
        pos_pct = round((pos_count / total_sents) * 100)
        neg_pct = round((neg_count / total_sents) * 100)
        neu_pct = max(0, 100 - pos_pct - neg_pct)

        subjectivity_pct = min(
            96,
            max(18, round((pos_pct + neg_pct) * 0.72 + abs(overall_polarity) * 25)),
        )
        has_negation = bool(
            re.search(r"\b(not|never|no|without|neither|nor|can't|won't|isn't|didn't)\b", text.lower())
        )
        avg_word_len = sum(len(w) for w in text.split()) / max(len(text.split()), 1)
        formality_pct = min(98, max(35, round(avg_word_len * 14.5)))
        urgency_label = (
            "High"
            if ("!" in text or agg_emotions["anger"] > 0.28 or agg_emotions["fear"] > 0.28)
            else "Norm."
        )

        word_count = len(text.split())
        char_count = len(text)

        return {
            "model_name": self.model_name_loaded,
            "dominant_emotion": dominant_emotion,
            "confidence_pct": overall_confidence,
            "aggregate_polarity": overall_polarity,
            "subjectivity_pct": subjectivity_pct,
            "composition": {"pos": pos_pct, "neu": neu_pct, "neg": neg_pct},
            "emotions_pct": {
                k: round(v * 100, 1) for k, v in agg_emotions.items()
            },
            "sentences": sentence_results,
            "positive_drivers": pos_drivers,
            "negative_drivers": neg_drivers,
            "aspects": aspects,
            "telemetry": {
                "negation": "Detected" if has_negation else "Minimal",
                "negation_sub": "Self-Attended" if has_negation else "Direct Polarity",
                "formality_pct": formality_pct,
                "formality_sub": "Scholarly / High" if formality_pct >= 70 else "Conversational",
                "urgency": urgency_label,
                "urgency_sub": "Elevated" if urgency_label == "High" else "Balanced",
            },
            "stats": {
                "words": word_count,
                "chars": char_count,
                "read_time_sec": max(5, round((word_count / 220) * 60)),
            },
        }


DB_PATH = os.path.join("artifacts", "analysis_history.db")


def init_db() -> None:
    os.makedirs("artifacts", exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                snippet TEXT,
                dominant_emotion TEXT,
                polarity REAL,
                confidence REAL
            )
            """
        )


def record_history(
    snippet: str, dominant_emotion: str, polarity: float, confidence: float
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO history (timestamp, snippet, dominant_emotion, polarity, confidence) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                snippet[:180],
                dominant_emotion,
                polarity,
                confidence,
            ),
        )


class AnalyzeRequest(BaseModel):
    text: str


app = FastAPI(
    title="Sentima AI — DistilBERT Emotion & Sentiment Engine",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service: DistilBertEmotionService | None = None


@app.on_event("startup")
def on_startup() -> None:
    global service
    init_db()
    service = DistilBertEmotionService()


@app.get("/", response_class=HTMLResponse)
def serve_stitch_ui() -> Any:
    for candidate in ["page.html", os.path.join("frontend", "page.html")]:
        if os.path.exists(candidate):
            return FileResponse(candidate)
    raise HTTPException(status_code=404, detail="page.html not found in project root.")


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": service.model_name_loaded if service else "loading",
    }


@app.post("/api/analyze")
def analyze_text(req: AnalyzeRequest) -> Dict[str, Any]:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is still loading.")

    result = service.analyze_corpus(req.text)
    record_history(
        req.text.strip(),
        result["dominant_emotion"],
        result["aggregate_polarity"],
        result["confidence_pct"],
    )
    return result


@app.get("/api/history")
def get_history() -> List[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, timestamp, snippet, dominant_emotion, polarity, confidence FROM history ORDER BY id DESC LIMIT 25"
        ).fetchall()
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "snippet": r[2],
            "dominant_emotion": r[3],
            "polarity": r[4],
            "confidence": r[5],
        }
        for r in rows
    ]


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)