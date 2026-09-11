# Spotify AI Customer Support Agent

An end-to-end AI support agent that classifies incoming Spotify customer queries, retrieves contextually similar historical resolutions via RAG, drafts brand-aligned replies, and intelligently routes high-risk cases to human agents.

---

## Overview

Given any incoming customer tweet directed at `@SpotifyCares`:

1. **Classify**: Assign one of 8 Spotify-specific intents (e.g., `playback_issue`, `subscription_billing`)
2. **Retrieve**: Pull top historical conversation pairs from a FAISS vector store
3. **Draft**: Generate a concise, helpful response adhering to Spotify's brand guidelines
4. **Decide**: Auto-handle routine inquiries or escalate to a human agent with a transparent reason

---

## Quickstart (Reproduce in < 15 Minutes)

### Prerequisites
- Python 3.10+
- pip
- (Optional) Gemini API key for LLM mode (the system runs completely offline in baseline mode without an API key)

### Setup

```bash
# 1. Navigate to repository root
cd Assignment_Hiver

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: configure API key
cp .env.example .env
# Set GEMINI_API_KEY in .env if testing Gemini mode
# Without an API key, the system runs with local TF-IDF + FAISS baseline

# 5. Extract Spotify data and build vector index
python data/download_data.py
python data/build_vectorstore.py
```

### Run Inference

```bash
# Interactive CLI
python agent/pipeline.py

# Single message
python agent/pipeline.py --message "my songs keep stopping randomly"

# Baseline mode (zero API key needed)
python agent/pipeline.py --mode baseline --message "can't log into my account"
```

**Example output:**
```
-----------------------------------------------------------------
[Message]      : my songs keep stopping randomly
[Intent]       : playback_issue (confidence: 0.94)
[Top Retrieve] : songs keep pausing randomly
[Draft Reply]  : Sounds frustrating! Try clearing your app cache first (Settings > Storage > Clear Cache). Still happening? DM us
[Decision]     : AUTO_HANDLE
[Reason]       : Standard playback issue with high confidence and good match.
[Latency]      : 487.3 ms
-----------------------------------------------------------------
```

### Run Full Evaluation

```bash
# Full eval (automated metrics + LLM judge if API key configured)
python eval/run_eval.py

# Automated metrics only (no API key needed)
python eval/run_eval.py --no-llm-judge

# Quick run on 20 examples
python eval/run_eval.py --no-llm-judge --limit 20
```

---

## Project Structure

```
Assignment_Hiver/
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── download_data.py        # Downloads Spotify subset (HF/Kaggle/synthetic)
│   ├── build_vectorstore.py    # Builds FAISS index from conversation pairs
│   ├── spotify_subset.csv      # Filtered Spotify conversations
│   └── vectorstore/            # Generated: FAISS index + metadata
│
├── agent/
│   ├── intent_classifier.py    # LLM few-shot + TF-IDF baseline classifier
│   ├── retriever.py            # FAISS semantic retriever (TF-IDF fallback)
│   ├── reply_drafter.py        # LLM drafter with brand-voice prompt + RAG
│   ├── escalation_decider.py   # Rule + confidence + LLM hybrid escalation
│   └── pipeline.py             # Main orchestrator + CLI entry point
│
├── eval/
│   ├── golden_set.csv          # 183 hand-labelled examples
│   ├── golden_set_notes.md     # Sampling + labelling methodology
│   ├── judge_prompt.py         # LLM-as-judge 5-dimension rubric
│   ├── run_eval.py             # Full evaluation harness
│   └── results/                # Generated: JSONL results + metrics JSON
│
└── report/
    └── report.md               # Full report (problem framing, results, failures)
```

---

## Intent Taxonomy

| Intent | Description |
|--------|-------------|
| `playback_issue` | Songs pausing, buffering, wrong quality, shuffle bugs |
| `account_login` | Login failures, password reset, 2FA issues |
| `subscription_billing` | Charges, cancellations, refunds, plan management |
| `app_crash_bug` | Crashes, freezes, black screen, post-update bugs |
| `download_offline` | Missing downloads, offline mode failures |
| `content_missing` | Removed albums/songs, regional unavailability |
| `device_compatibility` | PS5, Alexa, Apple Watch, smart TV, car Bluetooth |
| `other_general` | FAQs, privacy, sharing, Wrapped, general questions |

---

## Escalation Logic

The escalation decider uses a 3-layer hybrid approach:

1. **Hard rules** (always fires first, zero latency):
   - Account security keywords (hacked, unauthorized access)
   - Legal threats (sue, GDPR complaint, FTC)
   - Billing disputes (fraud, chargeback, unauthorized charge)
   - Repeat contact patterns ("still not working", "third time contacting")

2. **Confidence thresholds**:
   - Intent confidence < 0.40 → escalate (ambiguous case)
   - Retriever top-score < 0.20 → escalate (novel issue, no good match)

3. **LLM judgment** (edge cases only, if API key configured)

Every escalation decision includes a human-readable `reason`.

---

## Modes

| Mode | When to use | LLM needed? |
|------|------------|-------------|
| `llm` | Full pipeline (default) | Yes (Gemini API key) |
| `baseline` | No API key, or fast testing | No |
| `trivial` | Evaluation comparison floor | No |

---

## Key Results (from `eval/run_eval.py`)

| Metric | Trivial | Baseline | Agent |
|--------|---------|----------|-------|
| Intent Accuracy | ~12% | ~68% | **~84%** |
| Escalation F1 | 0% | ~63% | **~90%** |
| Judge Overall | 1.4/5 | 2.8/5 | **4.1/5** |

See [report/report.md](report/report.md) for full analysis, failure modes, and decision log.

---

## Design Decisions (Summary)

- **Brand: Spotify** — Rich data, finite intent space, distinctive voice, clear escalation signals
- **Gemini 1.5 Flash** — Free tier, fast, handles tweet-length text well
- **FAISS flat index** — Sub-millisecond retrieval at this scale, no server needed
- **Hybrid escalation** — Rules = zero cost + 100% recall on critical cases; LLM = edge cases only
- **Not fine-tuned** — Data is noisy/anonymized; few-shot prompting outperforms fine-tuning here

See [Decision Log in report.md](report/report.md#7-decision-log) for all 15 decisions with rationale.

---

## Requirements

```
google-generativeai>=0.7.0
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
datasets>=2.20.0
scikit-learn>=1.5.0
numpy>=1.26.0
pandas>=2.2.0
rouge-score>=0.1.2
python-dotenv>=1.0.0
tqdm>=4.66.0
```
