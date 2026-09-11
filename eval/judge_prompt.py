"""
judge_prompt.py
---------------
LLM-as-judge rubric and prompt for evaluating Spotify support reply quality.

The judge evaluates each generated reply on 5 dimensions (1–5 each):
    1. Helpfulness   - Does the reply address the actual issue?
    2. Brand Voice   - Is it casual, warm, and Spotify-like?
    3. Accuracy      - No hallucinated features or wrong instructions?
    4. Groundedness  - Anchored in provided context or real Spotify behavior?
    5. Conciseness   - Tweet-friendly length (not a wall of text)?

Overall score = weighted average (helpfulness × 0.35 + accuracy × 0.25 +
                groundedness × 0.20 + brand_voice × 0.10 + conciseness × 0.10)

The judge also flags whether it would escalate the issue differently than the agent.
"""

JUDGE_SYSTEM_PROMPT = """You are an expert Spotify customer support quality evaluator.
Your job is to score AI-generated support replies against a rubric.
Be strict but fair. Focus on whether the reply actually helps the customer.
"""

JUDGE_RUBRIC = """
RUBRIC:
1. Helpfulness (1-5): Does the reply directly address the customer's issue with actionable information?
   1=completely off-topic, 3=partially helpful, 5=fully addresses the issue

2. Brand Voice (1-5): Does the reply sound like Spotify's casual, warm, human support voice?
   1=robotic/corporate, 3=acceptable, 5=perfectly on-brand

3. Accuracy (1-5): Is the information correct? No hallucinated features, wrong URLs, or bad advice?
   1=contains harmful misinformation, 3=mostly correct with minor errors, 5=fully accurate

4. Groundedness (1-5): Is the reply anchored in real Spotify behavior (not made-up policies)?
   1=entirely fabricated, 3=partially grounded, 5=well-grounded

5. Conciseness (1-5): Is the reply an appropriate length for Twitter support (1-3 sentences)?
   1=way too long or too short, 3=acceptable, 5=perfectly concise
"""


def build_judge_prompt(
    customer_message: str,
    intent: str,
    generated_reply: str,
    ideal_reply_sketch: str = "",
    escalation_decision: str = "",
    escalation_reason: str = "",
) -> str:
    """
    Build a complete judge prompt for a single (message, reply) pair.

    Args:
        customer_message:    The raw customer tweet.
        intent:              The detected intent label.
        generated_reply:     The AI-generated draft reply to evaluate.
        ideal_reply_sketch:  Optional: brief note on what a good reply should contain.
        escalation_decision: "AUTO_HANDLE" | "ESCALATE"
        escalation_reason:   Stated reason for the decision.

    Returns:
        A complete prompt string ready to send to the judge LLM.
    """
    ideal_block = ""
    if ideal_reply_sketch:
        ideal_block = f"\nIdeal reply should contain: {ideal_reply_sketch}\n"

    esc_block = ""
    if escalation_decision:
        esc_block = f"""
Agent escalation decision: {escalation_decision}
Reason given: {escalation_reason}
"""

    return f"""{JUDGE_SYSTEM_PROMPT}

{JUDGE_RUBRIC}

---
CASE TO EVALUATE:

Customer message: "{customer_message}"
Detected intent: {intent}{ideal_block}{esc_block}

Generated reply to evaluate:
"{generated_reply}"

---
INSTRUCTIONS:
Score each dimension 1-5. Then compute an overall weighted score:
overall = round(helpfulness*0.35 + accuracy*0.25 + groundedness*0.20 + brand_voice*0.10 + conciseness*0.10, 2)

Also evaluate:
- "escalation_appropriate": true/false — was the escalation decision correct?
- "escalation_comment": one sentence explaining why or why not

Respond ONLY with a valid JSON object like:
{{
  "helpfulness": <int 1-5>,
  "brand_voice": <int 1-5>,
  "accuracy": <int 1-5>,
  "groundedness": <int 1-5>,
  "conciseness": <int 1-5>,
  "overall": <float>,
  "escalation_appropriate": <bool>,
  "escalation_comment": "<one sentence>",
  "critique": "<one or two sentences of the most important feedback>"
}}
"""


def parse_judge_response(text: str) -> dict:
    """Parse the JSON response from the judge LLM."""
    import json
    import re
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    try:
        result = json.loads(text)
        # Validate and compute weighted overall if needed
        dims = ["helpfulness", "accuracy", "groundedness", "brand_voice", "conciseness"]
        weights = [0.35, 0.25, 0.20, 0.10, 0.10]
        if all(d in result for d in dims) and "overall" not in result:
            result["overall"] = round(
                sum(result[d] * w for d, w in zip(dims, weights)), 2
            )
        return result
    except json.JSONDecodeError:
        return {
            "helpfulness": 0, "brand_voice": 0, "accuracy": 0,
            "groundedness": 0, "conciseness": 0, "overall": 0.0,
            "escalation_appropriate": None, "escalation_comment": "parse error",
            "critique": f"Judge response could not be parsed: {text[:100]}",
            "parse_error": True,
        }


if __name__ == "__main__":
    prompt = build_judge_prompt(
        customer_message="my songs keep pausing randomly",
        intent="playback_issue",
        generated_reply="Try clearing your Spotify cache! Go to Settings > Storage > Clear Cache. DM us if it keeps happening 🎵",
        ideal_reply_sketch="Clear cache, check internet connection",
        escalation_decision="AUTO_HANDLE",
        escalation_reason="Standard playback issue",
    )
    print(prompt)
