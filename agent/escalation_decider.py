"""
escalation_decider.py
---------------------
Decides whether a customer issue should be auto-handled or escalated to a
human agent — always with a stated reason.

Decision logic (hybrid approach):

  LAYER 1 – Hard rules (high-precision triggers that always escalate):
    • Intent is subscription_billing AND message contains keywords like
      "charged", "fraud", "refund", "chargeback", "dispute"
    • Message mentions account compromise ("hacked", "unauthorized",
      "someone changed my password/email")
    • Customer expresses legal threats ("sue", "lawyer", "FTC", "GDPR")
    • Repeated escalation signal: customer says "still not working" after
      mentioning a prior contact

  LAYER 2 – Confidence-based escalation:
    • Intent classifier confidence < 0.40 (ambiguous case)
    • Retriever top-1 similarity score < 0.20 (no good historical match)

  LAYER 3 – LLM judgment (optional, used when mode="llm"):
    • Gemini evaluates ambiguous edge cases with a structured prompt

  If none of the above triggers fire → AUTO_HANDLE.

Returns:
    {
        "decision": "AUTO_HANDLE" | "ESCALATE",
        "reason": str,        # human-readable explanation
        "triggered_by": str,  # which layer fired ("rule", "confidence", "llm")
    }
"""

from __future__ import annotations

import json
import re
from typing import Literal

# ── Hard-rule keyword sets ───────────────────────────────────────────────────

BILLING_ESCALATION_KEYWORDS = {
    "fraud", "chargeback", "dispute", "unauthorized charge", "refund",
    "charged multiple times", "double charged", "triple charged",
    "didn't authorize", "report to bank",
}

SECURITY_ESCALATION_KEYWORDS = {
    "hacked", "hack", "compromised", "someone changed my",
    "unauthorized access", "stolen account", "account takeover",
    "someone logged in", "wasn't me",
}

LEGAL_ESCALATION_KEYWORDS = {
    "sue", "lawsuit", "lawyer", "attorney", "ftc complaint",
    "gdpr", "data breach", "legal action", "consumer protection",
    "file a complaint",
}

REPEAT_CONTACT_PATTERNS = [
    r"still not (working|fixed|resolved)",
    r"(already|previously) contacted",
    r"been waiting (for )?(days|weeks|hours)",
    r"this is the (second|third|\d+)(st|nd|rd|th)? time",
    r"no one has helped",
]

ESCALATION_INTENTS = {"subscription_billing", "account_login"}


def _matches_any(text: str, keywords: set) -> str | None:
    """Return the matched keyword, or None."""
    tl = text.lower()
    for kw in keywords:
        if kw in tl:
            return kw
    return None


def _matches_pattern(text: str, patterns: list[str]) -> str | None:
    """Return matched pattern, or None."""
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            return m.group(0)
    return None


# ── Escalation decision function ─────────────────────────────────────────────

def decide_escalation(
    message: str,
    intent: str,
    intent_confidence: float,
    retriever_top_score: float,
    llm_model=None,
) -> dict:
    """
    Main escalation decision logic.

    Args:
        message:              The raw customer message.
        intent:               Predicted intent label.
        intent_confidence:    Classifier confidence (0.0–1.0).
        retriever_top_score:  Similarity score of best retrieved example (0.0–1.0).
        llm_model:            Optional Gemini model for LLM-layer judgment.

    Returns:
        {"decision": str, "reason": str, "triggered_by": str}
    """

    # ── Layer 1a: Security / account compromise ──────────────────────────────
    kw = _matches_any(message, SECURITY_ESCALATION_KEYWORDS)
    if kw:
        return {
            "decision": "ESCALATE",
            "reason": f"Account security concern detected ('{kw}'). Requires immediate human verification.",
            "triggered_by": "rule:security",
        }

    # ── Layer 1b: Legal threats ──────────────────────────────────────────────
    kw = _matches_any(message, LEGAL_ESCALATION_KEYWORDS)
    if kw:
        return {
            "decision": "ESCALATE",
            "reason": f"Legal language detected ('{kw}'). Must be handled by a senior support agent.",
            "triggered_by": "rule:legal",
        }

    # ── Layer 1c: Billing disputes (high-severity subset) ───────────────────
    if intent == "subscription_billing":
        kw = _matches_any(message, BILLING_ESCALATION_KEYWORDS)
        if kw:
            return {
                "decision": "ESCALATE",
                "reason": f"Billing dispute or refund request detected ('{kw}'). Requires human review of payment records.",
                "triggered_by": "rule:billing",
            }

    # ── Layer 1d: Repeat contact / frustrated returning customer ─────────────
    pat = _matches_pattern(message, REPEAT_CONTACT_PATTERNS)
    if pat:
        return {
            "decision": "ESCALATE",
            "reason": f"Customer indicates a repeated or unresolved contact ('{pat}'). Escalating for continuity.",
            "triggered_by": "rule:repeat_contact",
        }

    # ── Layer 2: Confidence-based ────────────────────────────────────────────
    if intent_confidence < 0.40:
        return {
            "decision": "ESCALATE",
            "reason": f"Intent unclear (confidence={intent_confidence:.2f}). Routing to human to avoid a wrong auto-reply.",
            "triggered_by": "confidence:low_intent",
        }

    if retriever_top_score < 0.20:
        return {
            "decision": "ESCALATE",
            "reason": f"No sufficiently similar historical case found (score={retriever_top_score:.2f}). Human needed for novel issue.",
            "triggered_by": "confidence:low_retrieval",
        }

    # ── Layer 3: LLM judgment for edge cases ─────────────────────────────────
    if llm_model is not None:
        llm_decision = _llm_escalation_check(message, intent, llm_model)
        if llm_decision["decision"] == "ESCALATE":
            return llm_decision

    # ── Default: auto-handle ─────────────────────────────────────────────────
    intent_display = intent.replace("_", " ")
    if not intent_display.endswith("issue"):
        intent_display += " inquiry"
    return {
        "decision": "AUTO_HANDLE",
        "reason": f"Standard {intent_display} with good classifier confidence and historical match. Safe to auto-reply.",
        "triggered_by": "default",
    }


def _llm_escalation_prompt(message: str, intent: str) -> str:
    return f"""You are a triage agent for Spotify customer support on Twitter.

Decide whether this customer message should be handled automatically or escalated to a human agent.

Escalate if the issue involves:
- Account security (hacking, unauthorized access)
- Billing disputes, fraud, or refund demands
- Legal threats
- Complex emotional distress (threatening to cancel after long unresolved issue)
- Anything ambiguous where a wrong automated reply would cause real harm

Message: "{message}"
Detected intent: {intent}

Respond ONLY with a JSON object:
{{"decision": "AUTO_HANDLE" | "ESCALATE", "reason": "<one concise sentence explaining why>"}}
"""


def _llm_escalation_check(message: str, intent: str, llm_model) -> dict:
    """Ask the LLM to make the escalation call for edge cases."""
    try:
        response = llm_model.generate_content(
            _llm_escalation_prompt(message, intent)
        )
        text = response.text.strip()
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        result = json.loads(text)
        return {
            "decision": result.get("decision", "AUTO_HANDLE"),
            "reason": result.get("reason", "LLM judgment."),
            "triggered_by": "llm",
        }
    except Exception:
        return {"decision": "AUTO_HANDLE", "reason": "LLM unavailable.", "triggered_by": "llm_error"}


# ── Trivial baseline ──────────────────────────────────────────────────────────

def trivial_escalation_decider(message: str, intent: str, **kwargs) -> dict:
    """Always auto-handles — simplest possible baseline."""
    return {
        "decision": "AUTO_HANDLE",
        "reason": "Trivial baseline: always auto-handle.",
        "triggered_by": "trivial",
    }


if __name__ == "__main__":
    test_cases = [
        ("my songs keep pausing", "playback_issue", 0.92, 0.85),
        ("I was charged twice and I want a refund", "subscription_billing", 0.88, 0.70),
        ("someone hacked my account and changed my email", "account_login", 0.95, 0.55),
        ("I'm going to sue Spotify", "other_general", 0.60, 0.30),
        ("still not working, I already contacted you twice", "app_crash_bug", 0.75, 0.65),
        ("blah blah something weird I've never seen", "other_general", 0.22, 0.10),
    ]
    for msg, intent, conf, score in test_cases:
        result = decide_escalation(msg, intent, conf, score)
        print(f"[{result['decision']:11s}] ({result['triggered_by']:25s}) {msg[:55]}")
        print(f"             → {result['reason']}\n")
