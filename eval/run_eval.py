"""
run_eval.py
-----------
Full evaluation harness for the Spotify AI Support Agent.

Evaluates three systems on the golden set:
    1. Trivial baseline  — mode-intent + canned reply
    2. Simple baseline   — TF-IDF classifier + nearest-neighbor reply
    3. Our agent         — full RAG + LLM pipeline

Automated metrics:
    - Intent accuracy (vs. golden labels)
    - Escalation precision, recall, F1 (vs. golden should_escalate)
    - ROUGE-L (vs. ideal_reply_sketch — acknowledged as weak signal)
    - Response latency (p50, p95)

LLM-as-judge metrics (if Gemini API key available):
    - Average scores for helpfulness, brand_voice, accuracy, groundedness, conciseness
    - Average overall weighted score
    - Escalation appropriateness rate
    - Human agreement (Cohen's κ) on a 30-example spot-check

Usage:
    python eval/run_eval.py                   # Full eval with LLM judge
    python eval/run_eval.py --no-llm-judge    # Automated metrics only
    python eval/run_eval.py --limit 20        # Quick run on first 20 examples
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

load_dotenv()

from agent.pipeline import SpotifySupportAgent
from agent.intent_classifier import INTENTS

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GOLDEN_SET = Path(__file__).parent / "golden_set.csv"


# ── Metric helpers ────────────────────────────────────────────────────────────

def rouge_l(hypothesis: str, reference: str) -> float:
    """Sentence-level ROUGE-L (LCS-based F1)."""
    if not hypothesis or not reference:
        return 0.0
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    m, n = len(hyp_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / m if m > 0 else 0
    recall = lcs / n if n > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's κ for two binary label lists."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    agree = sum(a == b for a, b in zip(labels_a, labels_b))
    po = agree / n
    # Expected agreement
    p_pos_a = sum(labels_a) / n
    p_pos_b = sum(labels_b) / n
    pe = p_pos_a * p_pos_b + (1 - p_pos_a) * (1 - p_pos_b)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ── LLM Judge ─────────────────────────────────────────────────────────────────

def run_llm_judge(rows: list[dict], gemini_model) -> list[dict]:
    """Run LLM-as-judge on a list of result rows. Returns updated rows."""
    from eval.judge_prompt import build_judge_prompt, parse_judge_response

    print(f"\n🧑‍⚖️  Running LLM judge on {len(rows)} examples...")
    for row in tqdm(rows, desc="LLM judge"):
        prompt = build_judge_prompt(
            customer_message=row["message"],
            intent=row["intent"],
            generated_reply=row["draft_reply"],
            ideal_reply_sketch=row.get("ideal_reply_sketch", ""),
            escalation_decision=row["escalation_decision"],
            escalation_reason=row["escalation_reason"],
        )
        try:
            response = gemini_model.generate_content(prompt)
            scores = parse_judge_response(response.text)
        except Exception as e:
            scores = {
                "helpfulness": 0, "brand_voice": 0, "accuracy": 0,
                "groundedness": 0, "conciseness": 0, "overall": 0.0,
                "escalation_appropriate": None, "escalation_comment": str(e),
                "critique": f"Judge error: {e}",
            }
        row["judge"] = scores
        time.sleep(0.5)  # Rate limiting

    return rows


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_system(
    agent: SpotifySupportAgent,
    df: pd.DataFrame,
    system_name: str,
) -> dict:
    """Run a system on all golden examples and collect raw results."""
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {system_name}"):
        result = agent.run(str(row["message"]))
        result["golden_intent"] = str(row["intent_label"])
        result["golden_escalate"] = bool(row["should_escalate"])
        result["ideal_reply_sketch"] = str(row.get("ideal_reply_sketch", ""))
        result["system"] = system_name
        results.append(result)
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute all automated metrics from a results list."""
    metrics = {}

    # Intent accuracy
    correct = sum(r["intent"] == r["golden_intent"] for r in results)
    metrics["intent_accuracy"] = correct / len(results)

    # Per-class accuracy
    per_class = {}
    for intent in INTENTS:
        subset = [r for r in results if r["golden_intent"] == intent]
        if subset:
            c = sum(r["intent"] == r["golden_intent"] for r in subset)
            per_class[intent] = c / len(subset)
    metrics["per_class_accuracy"] = per_class

    # Escalation precision, recall, F1
    tp = sum(
        1 for r in results
        if r["escalation_decision"] == "ESCALATE" and r["golden_escalate"]
    )
    fp = sum(
        1 for r in results
        if r["escalation_decision"] == "ESCALATE" and not r["golden_escalate"]
    )
    fn = sum(
        1 for r in results
        if r["escalation_decision"] == "AUTO_HANDLE" and r["golden_escalate"]
    )
    p, rec, f1 = precision_recall_f1(tp, fp, fn)
    metrics["escalation_precision"] = p
    metrics["escalation_recall"] = rec
    metrics["escalation_f1"] = f1
    metrics["escalation_tp"] = tp
    metrics["escalation_fp"] = fp
    metrics["escalation_fn"] = fn

    # ROUGE-L
    rouge_scores = [
        rouge_l(r["draft_reply"], r["ideal_reply_sketch"])
        for r in results
    ]
    metrics["rouge_l_mean"] = float(np.mean(rouge_scores))
    metrics["rouge_l_std"] = float(np.std(rouge_scores))

    # Latency
    latencies = [r["latency_ms"] for r in results]
    metrics["latency_p50"] = float(np.percentile(latencies, 50))
    metrics["latency_p95"] = float(np.percentile(latencies, 95))
    metrics["latency_mean"] = float(np.mean(latencies))

    # LLM judge (if available)
    judged = [r for r in results if "judge" in r and not r["judge"].get("parse_error")]
    if judged:
        dims = ["helpfulness", "brand_voice", "accuracy", "groundedness", "conciseness", "overall"]
        for dim in dims:
            vals = [r["judge"][dim] for r in judged if dim in r["judge"]]
            if vals:
                metrics[f"judge_{dim}_mean"] = float(np.mean(vals))

        esc_correct = [
            r["judge"].get("escalation_appropriate")
            for r in judged
            if r["judge"].get("escalation_appropriate") is not None
        ]
        if esc_correct:
            metrics["judge_escalation_appropriate_rate"] = float(
                np.mean([1 if x else 0 for x in esc_correct])
            )
        metrics["judge_n_evaluated"] = len(judged)

    return metrics


def print_results_table(all_metrics: dict[str, dict]):
    """Print a formatted comparison table."""
    systems = list(all_metrics.keys())
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS — Spotify AI Support Agent")
    print("=" * 80)

    key_metrics = [
        ("Intent Accuracy", "intent_accuracy", "{:.1%}"),
        ("Escalation Precision", "escalation_precision", "{:.1%}"),
        ("Escalation Recall", "escalation_recall", "{:.1%}"),
        ("Escalation F1", "escalation_f1", "{:.1%}"),
        ("ROUGE-L (vs sketch)", "rouge_l_mean", "{:.3f}"),
        ("Latency p50 (ms)", "latency_p50", "{:.0f}"),
        ("Latency p95 (ms)", "latency_p95", "{:.0f}"),
        ("Judge: Helpfulness", "judge_helpfulness_mean", "{:.2f}/5"),
        ("Judge: Brand Voice", "judge_brand_voice_mean", "{:.2f}/5"),
        ("Judge: Accuracy", "judge_accuracy_mean", "{:.2f}/5"),
        ("Judge: Overall", "judge_overall_mean", "{:.2f}/5"),
        ("Judge: Escalation OK", "judge_escalation_appropriate_rate", "{:.1%}"),
    ]

    # Header
    col_w = 20
    print(f"\n{'Metric':<35}", end="")
    for s in systems:
        print(f"{s:>{col_w}}", end="")
    print()
    print("-" * (35 + col_w * len(systems)))

    for label, key, fmt in key_metrics:
        print(f"{label:<35}", end="")
        for s in systems:
            val = all_metrics[s].get(key)
            if val is None:
                print(f"{'—':>{col_w}}", end="")
            else:
                print(f"{fmt.format(val):>{col_w}}", end="")
        print()

    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spotify Agent Evaluation Harness")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Skip LLM-as-judge evaluation (automated metrics only).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples (for quick testing).")
    parser.add_argument("--judge-sample", type=int, default=30,
                        help="Number of examples to run through LLM judge (default: 30).")
    args = parser.parse_args()

    # Load golden set
    df = pd.read_csv(GOLDEN_SET)
    df["should_escalate"] = df["should_escalate"].astype(str).str.lower() == "true"
    if args.limit:
        df = df.head(args.limit)
    print(f"Golden set loaded: {len(df)} examples.")

    # Init Gemini for judge
    gemini_model = None
    if not args.no_llm_judge:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key and api_key != "your_gemini_api_key_here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                print("LLM judge: Gemini 1.5 Flash ready.")
            except Exception as e:
                print(f"Warning: Gemini unavailable ({e}). Skipping LLM judge.")
        else:
            print("No GEMINI_API_KEY — LLM judge skipped. Automated metrics only.")

    all_results = {}
    all_metrics = {}

    # ── System 1: Trivial baseline ───────────────────────────────────────────
    print("\n[1/3] Trivial baseline...")
    trivial_agent = SpotifySupportAgent(mode="trivial")
    trivial_results = evaluate_system(trivial_agent, df, "trivial")
    all_results["trivial"] = trivial_results
    all_metrics["trivial"] = compute_metrics(trivial_results)

    # ── System 2: Simple baseline (TF-IDF) ──────────────────────────────────
    print("\n[2/3] Simple baseline (TF-IDF)...")
    baseline_agent = SpotifySupportAgent(mode="baseline")
    baseline_results = evaluate_system(baseline_agent, df, "baseline")
    all_results["baseline"] = baseline_results
    all_metrics["baseline"] = compute_metrics(baseline_results)

    # ── System 3: Our agent (LLM if available, else baseline) ───────────────
    print("\n[3/3] Our agent...")
    agent = SpotifySupportAgent(mode="llm")
    agent_results = evaluate_system(agent, df, "agent")

    # LLM judge on agent results
    if gemini_model is not None and not args.no_llm_judge:
        sample_size = min(args.judge_sample, len(agent_results))
        judge_sample = agent_results[:sample_size]
        judge_sample = run_llm_judge(judge_sample, gemini_model)
        agent_results[:sample_size] = judge_sample

    all_results["agent"] = agent_results
    all_metrics["agent"] = compute_metrics(agent_results)

    # ── Human agreement κ ────────────────────────────────────────────────────
    judged_agent = [r for r in agent_results if "judge" in r]
    if judged_agent:
        # Compare: does judge agree with golden should_escalate?
        golden_esc = [r["golden_escalate"] for r in judged_agent]
        agent_esc = [r["escalation_decision"] == "ESCALATE" for r in judged_agent]
        judge_esc = [
            bool(r["judge"].get("escalation_appropriate"))
            for r in judged_agent
        ]
        # κ between agent escalation vs golden escalation
        kappa_agent = cohen_kappa(
            [int(x) for x in golden_esc],
            [int(x) for x in agent_esc],
        )
        all_metrics["agent"]["escalation_cohen_kappa"] = kappa_agent
        print(f"\nCohen's κ (agent escalation vs golden): {kappa_agent:.3f}")

    # ── Save results ─────────────────────────────────────────────────────────
    timestamp = int(time.time())
    for system, results in all_results.items():
        out_path = RESULTS_DIR / f"{system}_results_{timestamp}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")

    metrics_path = RESULTS_DIR / f"metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n[INFO] Metrics saved to {metrics_path}")

    # ── Print table ───────────────────────────────────────────────────────────
    print_results_table(all_metrics)

    # ── Failure analysis: top misclassified intents ──────────────────────────
    print("\n--- TOP FAILURE MODES (agent) ---")
    agent_errors = [
        r for r in agent_results
        if r["intent"] != r["golden_intent"]
    ]
    print(f"Intent misclassifications: {len(agent_errors)}/{len(agent_results)}")
    confusion = {}
    for r in agent_errors:
        key = (r["golden_intent"], r["intent"])
        confusion[key] = confusion.get(key, 0) + 1
    for (true_l, pred_l), cnt in sorted(confusion.items(), key=lambda x: -x[1])[:5]:
        print(f"  {true_l:25s} -> {pred_l:25s}  ({cnt}x)")

    esc_errors = [
        r for r in agent_results
        if (r["escalation_decision"] == "ESCALATE") != r["golden_escalate"]
    ]
    fn_esc = [r for r in esc_errors if r["golden_escalate"]]
    fp_esc = [r for r in esc_errors if not r["golden_escalate"]]
    print(f"\nEscalation errors: {len(esc_errors)}")
    print(f"  False negatives (missed escalations): {len(fn_esc)}")
    for r in fn_esc[:3]:
        print(f"    -> '{r['message'][:60]}...' [triggered_by={r['escalation_triggered_by']}]")
    print(f"  False positives (unnecessary escalations): {len(fp_esc)}")
    for r in fp_esc[:3]:
        print(f"    -> '{r['message'][:60]}...' [reason={r['escalation_reason'][:50]}]")

    print("\n[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
