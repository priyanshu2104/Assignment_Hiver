# Golden Evaluation Set — Sampling & Labelling Notes

## Overview

**Total examples**: 183  
**Brand**: Spotify (`@SpotifyCares`)  
**Curator / Annotator**: Priyanshu Shekhar

---

## Sampling Strategy

### Source
Examples were constructed by:
1. Observing common patterns in the Spotify subset of the `thoughtvector/customer-support-on-twitter` dataset
2. Augmenting with realistic synthetic examples following Twitter's 280-character constraint
3. Ensuring coverage of all 8 defined intent categories

### Stratification
The set is stratified by intent to avoid label imbalance issues in evaluation:

| Intent | Count | Notes |
|--------|-------|-------|
| `playback_issue` | 30 | Most common Spotify issue |
| `account_login` | 20 | Includes escalation-triggering hacked account cases |
| `subscription_billing` | 25 | Heavy escalation representation (billing disputes) |
| `app_crash_bug` | 15 | Device-specific bugs |
| `download_offline` | 12 | Smaller, well-defined category |
| `content_missing` | 15 | Album removal, region restrictions |
| `device_compatibility` | 20 | Cross-platform issues |
| `other_general` | 30 | Catch-all: FAQs, privacy, feature questions |
| *Escalation triggers (cross-intent)* | 33 | Security, fraud, legal, repeat contact |

### Escalation Labelling

`should_escalate = true` was applied when ANY of the following were present:

1. **Account security**: mentions of "hacked", "unauthorized access", "someone changed my password/email"
2. **Financial fraud**: "unauthorized charge", "chargeback", "fraud", "credit card without permission"
3. **Legal threats**: "sue", "complaint", "GDPR", "consumer protection"
4. **Repeat unresolved contact**: "third time contacting", "still not working after your advice", "been waiting a week"
5. **Explicit escalation demand**: "speak to a real person", "I want a supervisor"
6. **Privacy/data breach**: "data leaked", "phishing", "GDPR data request" (the harmful ones)

---

## Labelling Process

All examples were labelled by a single annotator with the following process:

1. Read the raw customer message
2. Assign the **single most fitting** intent (no multi-label)
3. Write a brief `ideal_reply_sketch` — not a full reply, just the key information the reply should contain
4. Mark `should_escalate` as boolean, with a short `escalate_reason` for all `true` cases

### Edge Cases & Judgment Calls

- **Billing questions that don't involve disputes** (e.g., "how to cancel") → `should_escalate = false`. These can be handled with a link.
- **Refund requests** → always `should_escalate = true`. Even simple refund requests require access to payment records.
- **"I need to speak to a human"** → `should_escalate = true`. Explicit human escalation demand must be honored.
- **Premium not activating** → `should_escalate = false` if it seems like a sync issue; `true` if combined with billing language.
- **Privacy data requests** → non-harmful requests (e.g., "how do I see my data") → `false`; breach reports → `true`.

---

## Known Limitations

- All examples were written in English. Non-English messages are out of scope.
- The golden set slightly over-represents escalation scenarios (~16%) to stress-test the escalation module; real-world escalation rate is likely ~5-10%.
- `ideal_reply_sketch` is intentionally vague — it captures the key information needed, not the exact phrasing. The LLM judge uses it as a reference, not a match target.
- Single-annotator labelling (no inter-annotator agreement computed). A production golden set would require at least 2 annotators with κ > 0.7.
