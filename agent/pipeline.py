"""
pipeline.py
-----------
Main entry point for the Spotify AI Support Agent.

Usage:
    # Interactive demo
    python agent/pipeline.py

    # Single message
    python agent/pipeline.py --message "my songs keep stopping"

    # Baseline mode (no API key needed)
    python agent/pipeline.py --mode baseline --message "can't log in"

    # Run on a CSV file of messages
    python agent/pipeline.py --input eval/golden_set.csv --output results.jsonl

The pipeline:
    1. Classifies the customer message into an intent
    2. Retrieves top-3 most similar historical Spotify conversations
    3. Drafts a reply in Spotify's brand voice
    4. Decides: auto-handle or escalate (with stated reason)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from agent.intent_classifier import IntentClassifier, TrivialClassifier
from agent.retriever import Retriever, TrivialRetriever
from agent.reply_drafter import ReplyDrafter, TrivialDrafter
from agent.escalation_decider import decide_escalation, trivial_escalation_decider


# ── Gemini setup ──────────────────────────────────────────────────────────────

def _init_gemini():
    """Initialize Gemini model if API key is available."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model
    except Exception as e:
        print(f"Warning: Could not initialize Gemini ({e}). Falling back to baseline.")
        return None


# ── Agent ─────────────────────────────────────────────────────────────────────

class SpotifySupportAgent:
    """
    Full pipeline for Spotify customer support.

    Modes:
        "llm"      — Uses Gemini for classification + drafting (needs API key)
        "baseline" — TF-IDF classifier + nearest-neighbor reply (no API key)
        "trivial"  — Always mode-intent + canned reply (for eval comparison)
    """

    def __init__(self, mode: Literal["llm", "baseline", "trivial"] = "llm"):
        self.mode = mode
        gemini = None

        if mode == "llm":
            gemini = _init_gemini()
            if gemini is None:
                print("No Gemini API key found — switching to baseline mode.")
                self.mode = "baseline"

        print(f"\n[INFO] Initializing Spotify Support Agent ({self.mode} mode)...")

        if self.mode == "trivial":
            self.classifier = TrivialClassifier()
            self.retriever = TrivialRetriever()
            self.drafter = TrivialDrafter()
        elif self.mode == "baseline":
            self.classifier = IntentClassifier(mode="baseline")
            self.retriever = Retriever()
            self.drafter = ReplyDrafter(mode="baseline")
        else:  # llm
            self.classifier = IntentClassifier(mode="llm", gemini_model=gemini)
            self.retriever = Retriever()
            self.drafter = ReplyDrafter(mode="llm", gemini_model=gemini)

        self._gemini = gemini
        print("[INFO] Agent ready.\n")

    def run(self, message: str) -> dict:
        """
        Process one customer message through the full pipeline.
        """
        t0 = time.perf_counter()

        # Step 1: Classify
        intent_result = self.classifier.classify(message)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # Step 2: Retrieve
        retrieved = self.retriever.retrieve(message, k=3)
        top_score = retrieved[0].score if retrieved else 0.0

        # Step 3: Draft reply
        draft = self.drafter.draft(message, intent, retrieved)

        # Step 4: Escalation decision
        if self.mode == "trivial":
            esc = trivial_escalation_decider(message, intent)
        else:
            esc = decide_escalation(
                message=message,
                intent=intent,
                intent_confidence=confidence,
                retriever_top_score=top_score,
                llm_model=self._gemini if self.mode == "llm" else None,
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "message": message,
            "intent": intent,
            "intent_confidence": round(confidence, 4),
            "retrieved": [
                {"customer_text": r.customer_text[:100],
                 "brand_reply": r.brand_reply[:120],
                 "score": round(r.score, 4)}
                for r in retrieved
            ],
            "draft_reply": draft,
            "escalation_decision": esc["decision"],
            "escalation_reason": esc["reason"],
            "escalation_triggered_by": esc["triggered_by"],
            "latency_ms": round(latency_ms, 1),
        }


def _pretty_print(result: dict):
    print("-----------------------------------------------------------------")
    print(f"[Message]      : {result['message']}")
    print(f"[Intent]       : {result['intent']} (confidence: {result['intent_confidence']:.2f})")
    print(f"[Top Retrieve] : {result['retrieved'][0]['customer_text'][:60] if result['retrieved'] else 'n/a'}")
    print(f"[Draft Reply]  : {result['draft_reply']}")
    print(f"[Decision]     : {result['escalation_decision']}")
    print(f"[Reason]       : {result['escalation_reason']}")
    print(f"[Latency]      : {result['latency_ms']} ms")
    print("-----------------------------------------------------------------")


def _run_on_csv(agent: SpotifySupportAgent, input_path: str, output_path: str):
    import pandas as pd
    df = pd.read_csv(input_path)
    if "customer_text" not in df.columns and "message" not in df.columns:
        raise ValueError("CSV must have a 'customer_text' or 'message' column.")
    msg_col = "customer_text" if "customer_text" in df.columns else "message"

    results = []
    for _, row in df.iterrows():
        result = agent.run(str(row[msg_col]))
        results.append(result)

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[INFO] Wrote {len(results)} results to {output_path}")


def _interactive_loop(agent: SpotifySupportAgent):
    print("Interactive mode. Type a customer message (or 'quit' to exit).\n")
    while True:
        try:
            msg = input("Customer: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if msg.lower() in {"quit", "exit", "q"}:
            break
        if not msg:
            continue
        result = agent.run(msg)
        _pretty_print(result)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spotify AI Customer Support Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["llm", "baseline", "trivial"],
        default="llm",
        help="Agent mode: 'llm' (Gemini), 'baseline' (TF-IDF), 'trivial' (canned).",
    )
    parser.add_argument(
        "--message", "-m",
        type=str,
        default=None,
        help="Single customer message to process.",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to CSV file with customer messages (column: customer_text).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="eval/results/pipeline_output.jsonl",
        help="Output path for batch results (JSONL).",
    )
    args = parser.parse_args()

    agent = SpotifySupportAgent(mode=args.mode)

    if args.input:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        _run_on_csv(agent, args.input, args.output)
    elif args.message:
        result = agent.run(args.message)
        _pretty_print(result)
    else:
        _interactive_loop(agent)


if __name__ == "__main__":
    main()
