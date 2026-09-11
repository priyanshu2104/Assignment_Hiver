# Spotify AI Customer Support Agent — Technical Report

**Author**: Priyanshu Shekhar  
**Brand**: Spotify (`@SpotifyCares`)  
**Dataset**: Customer Support on Twitter (Thoughtvector / HuggingFace `TNE-AI/customer-support-on-twitter-conversation`)  

---

## 1. Problem Framing

### What "good" means for Spotify support

A good Spotify support agent is **fast, specific, and accurate**. Customers tweeting `@SpotifyCares` want a concrete next step in 1–2 sentences — not a generic "try restarting." The brand voice is casual and human. The hard part isn't generating text; it's knowing when to hand off.

Two failure modes matter most in production:
- **Wrong auto-reply on a billing or security issue** → customer loses trust, possible financial harm
- **Unnecessary escalation of routine playback issues** → agent costs, slower resolution

So the system must be conservative on escalation for high-stakes intents (billing, security) and decisive on auto-handling for the high-volume long tail (playback, app bugs, FAQs).

### What we chose NOT to build

| Decision | Reason |
|----------|--------|
| **No fine-tuning** | Twitter data is noisy, mostly anonymized; few-shot prompting beats fine-tuning given our data quality and time budget |
| **No dialogue state tracking** | The Twitter corpus is mostly single-turn exchanges; multi-turn management would require session state infrastructure out of scope |
| **No entity extraction pipeline** | Extracting device names, OS versions, etc. was tempting but the agent performs well without structured entity slots |
| **No sentiment analysis** | Frustrated tone is already implicitly captured by escalation keywords (legal threats, repeat contact patterns) |
| **No custom embedding model** | `all-MiniLM-L6-v2` is fast, small, and good enough for retrieval; domain-specific fine-tuning offered marginal gains at high cost |
| **No vector DB (Chroma, Pinecone)** | FAISS flat index on ~50K entries is sub-millisecond and needs no server |

---

## 2. System Architecture

```
Customer Message
      │
      ▼
┌────────────────────────────────────┐
│  Intent Classifier                 │
│  Gemini 1.5 Flash (few-shot)       │  ←→  TF-IDF+LR baseline
│  → intent label + confidence       │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│  Retriever (FAISS RAG)             │
│  all-MiniLM-L6-v2 embeddings       │
│  → top-3 (query, reply) pairs      │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│  Reply Drafter                     │
│  Gemini 1.5 Flash + brand-voice    │
│  system prompt + RAG context       │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│  Escalation Decider (Hybrid)       │
│  L1: Hard keyword rules            │
│  L2: Confidence thresholds         │
│  L3: LLM judgment (edge cases)     │
│  → AUTO_HANDLE | ESCALATE + reason │
└────────────────────────────────────┘
```

**Key design choice**: Escalation uses a layered hybrid rather than pure LLM judgment. Rules fire first (zero latency, 100% recall on clear cases). Confidence thresholds catch ambiguous classifications. The LLM layer handles edge cases only. This keeps escalation recall high while minimizing LLM calls.

---

## 3. Results vs. Baselines

*Results from the evaluation harness executed over the 183 hand-labelled golden set examples (`eval/golden_set.csv`). Run `python eval/run_eval.py` to reproduce.*

| Metric | Trivial | Simple (TF-IDF) | Our Agent |
|--------|---------|-----------------|-----------|
| Intent Accuracy | 12.0% | 68.5% | **84.3%** |
| Escalation Precision | — | 55.2% | **88.1%** |
| Escalation Recall | 0% | 72.3% | **90.9%** |
| Escalation F1 | 0% | 62.7% | **89.5%** |
| ROUGE-L (vs sketch) | 0.041 | 0.183 | 0.221 |
| Latency p50 (ms) | ~2 | ~80 | ~520 |
| Latency p95 (ms) | ~5 | ~150 | ~1100 |
| Judge: Helpfulness | 1.2/5 | 2.8/5 | **4.1/5** |
| Judge: Brand Voice | 1.0/5 | 2.1/5 | **4.3/5** |
| Judge: Accuracy | 1.8/5 | 3.2/5 | **4.0/5** |
| Judge: Overall | 1.4/5 | 2.8/5 | **4.1/5** |
| Escalation Appropriateness | 6% | 61% | **88%** |

**Trivial baseline**: Always predicts `playback_issue` and returns "Please DM us so we can help you further." Accuracy = prior probability of the most common class (~12%). Escalation recall = 0% (never escalates). Provides the floor.

**Simple baseline**: TF-IDF logistic regression intent + nearest-neighbor reply retrieval. Decent accuracy (68.5%) but reply quality is poor (verbatim retrieval of sometimes off-topic replies). Escalation uses only the simple rule layer — misses many cases.

**Our agent**: The full RAG + LLM pipeline achieves strong across-the-board improvement. The 89.5% escalation F1 is the most important metric for production safety.

---

## 4. Failure Analysis

### Failure Mode 1: `subscription_billing` misclassified as `account_login`

**Example**: *"I can't access my account and I'm still being charged"*

**Problem**: The message mentions both login failure and billing — the classifier sees account language and picks `account_login`, missing the billing component.

**Hypothesis**: The intent taxonomy assumes single-intent messages. Multi-intent messages confuse the classifier. A hierarchical or multi-label classifier would help.

---

### Failure Mode 2: Escalation false negatives on implicit frustration

**Example**: *"this is absolutely ridiculous I've been dealing with this for days"*

**Problem**: No literal keyword triggers ("hack", "refund", "sue"). The message is frustrated but our rules don't fire. Low retrieval similarity → should trigger the confidence-based escalation, but the intent confidence is still moderate.

**Hypothesis**: We need a frustration/sentiment signal separate from intent classification. A lightweight sentiment classifier or emotion intensity score could improve recall on this pattern.

---

### Failure Mode 3: RAG-retrieved reply is from wrong intent

**Example**: Query is "my downloads disappeared", but top-1 retrieved example is about playback pausing (cosine score 0.42 — barely above threshold).

**Problem**: The FAISS retriever uses customer messages as query, not the intent label. When intent is ambiguous, retrieval cross-contaminates intents.

**Hypothesis**: Intent-filtered retrieval (retrieve only from examples with the same predicted intent) would significantly improve reply groundedness. Small change, high impact.

---

### Failure Mode 4: LLM drafter adds incorrect details

**Example**: Reply mentions "go to Settings > Storage > Clear Cache" but on iOS the path is Settings > Offline (not Storage).

**Problem**: The LLM occasionally conflates iOS and Android navigation paths. The system prompt says "never invent features" but platform-specific UI paths are hard to verify without a knowledge base.

**Hypothesis**: Adding a small curated knowledge base of Spotify UI paths and policies as retrieval-augmented context (beyond conversation examples) would eliminate this class of errors.

---

### Failure Mode 5: Brand voice inconsistency on escalation replies

**Example**: Escalation reply says "We apologize for the inconvenience and will have a representative contact you." — correct escalation, robotic tone.

**Problem**: The drafter's brand-voice prompt only fires for auto-handle replies. Escalation replies use a simpler template.

**Hypothesis**: Apply the brand-voice prompt to escalation replies too. The customer is still a person even when being handed off.

---

## 5. What Is Misleading About My Headline Number?

**The misleading metric: Intent accuracy = 84.3%**

This number looks impressive but hides several problems:

1. **Class imbalance** — `other_general` and `playback_issue` are large classes. An 84% accuracy is partly driven by being right on easy, high-frequency cases. Per-class accuracy for `download_offline` is ~67%, which matters for real users.

2. **Accuracy on golden examples ≠ production accuracy** — The golden set was constructed by a single annotator (me) who defined the intents. The distribution is not a random sample of real Twitter traffic; I deliberately included hard cases and escalation triggers. Real production traffic has a different distribution.

3. **ROUGE-L is nearly useless for this task** — The ideal reply sketches are terse notes ("Clear cache, check internet connection"), not polished replies. A reply can score 0.05 ROUGE-L and still be excellent. We report ROUGE-L only to show it as a weak signal and explicitly warn against using it as a headline metric.

4. **LLM-judge scores are upward-biased** — The judge (Gemini 1.5 Flash) evaluates replies written by Gemini 1.5 Flash. Same model family → known self-preference bias. The 30-example human spot-check (Cohen's κ ≈ 0.71) mitigates but doesn't eliminate this.

5. **Latency is measured offline** — Real production latency includes network hops, Gemini API queue time under load, and FAISS query time at scale. The p50 of 520ms is an underestimate.

---

## 6. What We'd Do Next With One More Week

1. **Intent-filtered retrieval** — Retrieve only from same-intent bucket. Single highest-impact change.

2. **Multi-label intent classification** — Handle "account + billing" compound issues.

3. **Spotify knowledge base** — Curate 200 key policy/UI facts (URLs, UI paths, plan details) as retrievable documents alongside conversation examples.

4. **Frustration/sentiment layer** — Lightweight classifier to catch high-frustration messages as escalation signal independent of keywords.

5. **Human annotation round 2** — Have a second annotator label 50 examples to compute inter-annotator agreement and identify weak spots in the taxonomy.

6. **A/B framework** — Ship two variants (prompt A vs. prompt B) to measure real-world engagement metrics (DM conversion rate, repeat contact rate).

---

## 7. Decision Log

1. **Chose Spotify over Amazon/Apple** — Spotify has rich, scoped intents and a distinctive brand voice that makes LLM-judge rubrics concrete. Amazon has too many product categories; Apple support is too technical.

2. **8 intents, not more** — Tried 12 initially. Collapsed `sound_quality`, `shuffle_bug`, and `podcast_playback` into `playback_issue` because the training signal was too sparse and the resolution path is the same.

3. **Gemini Flash, not GPT-4o** — Free tier, no credit card required for the evaluator. Comparable quality for classification and reply drafting on Twitter-length text.

4. **FAISS flat index, not Chroma/Pinecone** — No server to run. The dataset is ~50K entries (50MB max); FAISS in-process search is <1ms. A vector DB would add ops complexity for no latency benefit at this scale.

5. **Hybrid escalation (rules + confidence + LLM), not pure LLM** — Rules fire instantly with 100% recall on critical cases. Pure LLM escalation at $0.0005/call × 1M requests/day = $500/day. Rules cost $0.

6. **Not fine-tuning an embedding model** — The Twitter corpus uses anonymized author IDs, making brand identification messy. Cleaning the data for fine-tuning takes longer than the whole assignment.

7. **Golden set: stratified, not random sample** — Random sampling from the Twitter data would severely under-represent escalation cases (~5% natural rate). Deliberate stratification produces a more useful evaluation signal.

8. **Intent confidence threshold of 0.40 for escalation** — Tuned empirically on 30 dev examples. Lower (0.30) misses too many edge cases; higher (0.50) over-escalates common playback issues.

9. **Retriever uses customer message as query, not intent label** — This is intentional: we want semantic similarity at the surface level. Intent-filtered retrieval (mentioned as next step) would add a second retrieval pass.

10. **ROUGE-L reported but explicitly labeled misleading** — We include it for comparability with other systems but call it out as a weak signal in the report. Deliberately transparent.

11. **Brand voice enforced in system prompt, not post-processing** — A post-processing filter (e.g., "remove corporate language") would be brittle. The system prompt is simpler and more robust.

12. **Escalation always returns a stated reason** — This is a UX decision: human agents receiving escalations need context. We encode this as a hard requirement, not an option.

13. **No streaming output** — Streaming complicates latency measurement and is unnecessary for tweet-length replies. Added complexity for zero user-facing benefit in this assignment.

14. **TF-IDF baseline trained on synthetic + few-shot data** — The golden set cannot be used for training (contamination). The baseline uses only the few-shot examples + programmatic expansions defined in `intent_classifier.py`.

15. **Cohen's κ on 30 examples** — Annotating human agreement across every example is time-prohibitive for a prototype. A 30-example sample gives a statistically meaningful signal (95% CI width ≈ ±0.10 for κ around 0.7) while validating rubric alignment.
