"""
intent_classifier.py
--------------------
Classifies incoming customer messages into one of 8 Spotify-specific intents.

Two modes:
  - "llm"      : Few-shot classification via Gemini 1.5 Flash (default)
  - "baseline" : TF-IDF + Logistic Regression (no API key needed)

The classifier returns:
    {
        "intent": str,          # predicted intent label
        "confidence": float,    # 0.0 – 1.0
        "all_scores": dict      # {intent: score, ...}
    }
"""

from __future__ import annotations

import json
import os
import re
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ── Intent taxonomy ──────────────────────────────────────────────────────────

INTENTS = [
    "playback_issue",
    "account_login",
    "subscription_billing",
    "app_crash_bug",
    "download_offline",
    "content_missing",
    "device_compatibility",
    "other_general",
]

INTENT_DESCRIPTIONS = {
    "playback_issue": "Music stops, pauses, skips, audio quality problems, crossfade, shuffle not working.",
    "account_login": "Can't log in, forgot password, account hacked, 2FA problems, wrong email.",
    "subscription_billing": "Charged incorrectly, can't cancel, refund requests, student/family plan, price issues.",
    "app_crash_bug": "App crashes, freezes, black screen, won't open, not responding after update.",
    "download_offline": "Downloaded songs disappeared, can't download, offline mode broken, download limit.",
    "content_missing": "Album/song/podcast removed, not available in country, playlist gone.",
    "device_compatibility": "Not working on PS5, Alexa, Apple Watch, TV, Bluetooth, car.",
    "other_general": "General questions, sharing playlists, account information, Wrapped, other topics.",
}

# Few-shot examples per intent (used in the LLM prompt)
FEW_SHOT_EXAMPLES = {
    "playback_issue": [
        "songs keep pausing randomly",
        "music stops every 30 seconds",
        "audio quality sounds terrible on bluetooth",
    ],
    "account_login": [
        "can't log into my account",
        "forgot my spotify password",
        "account was hacked someone changed my email",
    ],
    "subscription_billing": [
        "charged twice this month",
        "how do I cancel my premium",
        "student discount not working",
    ],
    "app_crash_bug": [
        "app keeps crashing on my iphone",
        "spotify shows black screen on android",
        "app freezes on home screen",
    ],
    "download_offline": [
        "my downloaded songs disappeared",
        "can't download songs for offline listening",
        "offline mode is not working",
    ],
    "content_missing": [
        "my favorite album was removed",
        "song not available in my country",
        "podcast episode disappeared",
    ],
    "device_compatibility": [
        "spotify not working on ps5",
        "alexa won't play spotify",
        "doesn't work on my smart TV",
    ],
    "other_general": [
        "how do I share a playlist",
        "can I have two accounts",
        "what is spotify wrapped",
    ],
}

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_CACHE = DATA_DIR / "baseline_classifier.pkl"


# ── LLM Classifier ───────────────────────────────────────────────────────────

def _build_llm_prompt(message: str) -> str:
    examples_block = ""
    for intent, examples in FEW_SHOT_EXAMPLES.items():
        for ex in examples[:2]:
            examples_block += f'  Message: "{ex}"\n  Intent: {intent}\n\n'

    intents_block = "\n".join(
        f"  - {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items()
    )

    return f"""You are an intent classifier for Spotify customer support messages.

Intent categories:
{intents_block}

Examples:
{examples_block}
Now classify this message. Respond with ONLY a JSON object like:
{{"intent": "<one of the intent labels>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}

Message: "{message}"
"""


def classify_with_llm(message: str, model) -> dict:
    """Classify using Gemini. `model` is a google.generativeai GenerativeModel."""
    prompt = _build_llm_prompt(message)
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        result = json.loads(text)
        intent = result.get("intent", "other_general")
        if intent not in INTENTS:
            intent = "other_general"
        confidence = float(result.get("confidence", 0.7))
        # Build pseudo all_scores
        all_scores = {i: 0.05 for i in INTENTS}
        all_scores[intent] = confidence
        return {
            "intent": intent,
            "confidence": confidence,
            "all_scores": all_scores,
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        # Graceful fallback to baseline
        return {"intent": "other_general", "confidence": 0.4,
                "all_scores": {i: 0.05 for i in INTENTS},
                "reasoning": f"LLM parse error: {e}"}


# ── Baseline TF-IDF Classifier ───────────────────────────────────────────────

def _build_training_data() -> tuple[list[str], list[str]]:
    """Build training examples from few-shot examples + synthetic expansions."""
    texts, labels = [], []

    extra = {
        "playback_issue": [
            "music keeps buffering", "crossfade not working", "shuffle repeats same songs",
            "spotify keeps stopping", "songs won't play", "bad audio quality streaming",
            "playback pauses randomly", "skip to next song automatically",
            "can't play the current song", "music cuts out every few minutes",
        ],
        "account_login": [
            "login not working", "reset my password", "email not recognized",
            "two factor authentication problem", "signed out of all devices",
            "can't access my account", "forgot email for spotify", "account security issue",
        ],
        "subscription_billing": [
            "unexpected charge on my card", "refund request", "cancel premium subscription",
            "family plan not working", "price increased", "billing problem",
            "premium trial ended unexpectedly", "student plan verification failed",
            "double charged", "cancel my account",
        ],
        "app_crash_bug": [
            "app not working", "spotify crashes", "black screen issue",
            "app freezing", "won't open", "loading forever", "not responding",
            "bug after update", "app glitch", "white screen",
        ],
        "download_offline": [
            "offline songs gone", "download feature broken", "can't save offline",
            "offline listening not working", "download limit reached", "songs not available offline",
        ],
        "content_missing": [
            "album removed from spotify", "can't find song", "not available in region",
            "podcast gone", "playlist deleted", "music not in my country",
            "content not available", "missing songs",
        ],
        "device_compatibility": [
            "not working on playstation", "amazon echo spotify problem",
            "apple watch spotify sync", "car bluetooth spotify issue",
            "roku spotify app", "xbox spotify", "TV app broken",
            "garmin spotify", "fitbit spotify",
        ],
        "other_general": [
            "how to create playlist", "share music with friend", "spotify stats",
            "wrapped not showing", "how to follow artist", "lyrics feature",
            "collaborative playlist", "radio station", "spotify connect",
        ],
    }

    for intent, examples in FEW_SHOT_EXAMPLES.items():
        for ex in examples:
            texts.append(ex)
            labels.append(intent)

    for intent, examples in extra.items():
        for ex in examples:
            texts.append(ex)
            labels.append(intent)

    return texts, labels


def train_baseline_classifier():
    """Train a TF-IDF + Logistic Regression classifier and cache it."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts, labels = _build_training_data()

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000,
                                   sublinear_tf=True)),
        ("lr", LogisticRegression(C=5.0, max_iter=1000, multi_class="ovr")),
    ])
    clf.fit(texts, labels)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(clf, f)

    return clf


def load_baseline_classifier():
    """Load cached baseline classifier or retrain."""
    if MODEL_CACHE.exists():
        with open(MODEL_CACHE, "rb") as f:
            return pickle.load(f)
    return train_baseline_classifier()


def classify_with_baseline(message: str, clf) -> dict:
    """Classify using TF-IDF + LR baseline."""
    proba = clf.predict_proba([message])[0]
    classes = clf.classes_
    all_scores = dict(zip(classes, proba.tolist()))
    best_idx = np.argmax(proba)
    return {
        "intent": classes[best_idx],
        "confidence": float(proba[best_idx]),
        "all_scores": all_scores,
        "reasoning": "TF-IDF baseline",
    }


# ── Public API ────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Unified intent classifier supporting both LLM and baseline modes.

    Usage:
        clf = IntentClassifier(mode="llm")          # needs GEMINI_API_KEY
        clf = IntentClassifier(mode="baseline")      # no API key needed
        result = clf.classify("songs keep stopping")
    """

    def __init__(self, mode: Literal["llm", "baseline"] = "llm", gemini_model=None):
        self.mode = mode
        self._llm_model = gemini_model
        self._baseline_clf = None

        if mode == "baseline" or gemini_model is None:
            self.mode = "baseline"
            print("Intent classifier: loading TF-IDF baseline...")
            self._baseline_clf = load_baseline_classifier()
            print("  Baseline classifier ready.")

    def classify(self, message: str) -> dict:
        """
        Returns:
            {
                "intent": str,
                "confidence": float,
                "all_scores": {intent: float},
                "reasoning": str,
            }
        """
        message = message.strip()
        if self.mode == "llm" and self._llm_model is not None:
            return classify_with_llm(message, self._llm_model)
        return classify_with_baseline(message, self._baseline_clf)


# ── Trivial baseline (for eval comparison) ───────────────────────────────────

class TrivialClassifier:
    """Always predicts 'playback_issue' (most common Spotify intent)."""

    def classify(self, message: str) -> dict:
        return {
            "intent": "playback_issue",
            "confidence": 1.0,
            "all_scores": {"playback_issue": 1.0},
            "reasoning": "Trivial baseline: always predicts most common class",
        }


if __name__ == "__main__":
    clf = IntentClassifier(mode="baseline")
    tests = [
        "my songs keep pausing randomly",
        "I can't log into my account",
        "charged twice this month please help",
        "app crashes every time i open it",
    ]
    for t in tests:
        result = clf.classify(t)
        print(f"[{result['intent']:25s} {result['confidence']:.2f}] {t}")
