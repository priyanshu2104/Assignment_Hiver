"""
reply_drafter.py
----------------
Drafts a reply to a customer message in Spotify's brand voice,
grounded in retrieved historical examples.

Two modes:
  - "llm"      : Gemini 1.5 Flash with brand-voice system prompt + RAG context
  - "baseline" : Simply returns the top-1 retrieved historical reply verbatim

Spotify brand voice guidelines (encoded in the system prompt):
  - Casual, warm, human — never robotic or corporate
  - Use light emojis (🎵, 🎶) occasionally but don't overdo it
  - Keep replies short: 1-3 sentences, tweet-friendly
  - Acknowledge the frustration briefly, then give actionable help
  - End with an invitation to DM for complex issues
  - Never invent features, prices, or policies
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.retriever import RetrievedExample

SYSTEM_PROMPT = """You are a helpful Spotify customer support agent on Twitter.

Brand voice:
- Casual, warm, and human — never stiff or robotic
- 1–3 sentences max (tweet-friendly length)
- Occasionally use 🎵 or 🎶 but keep it minimal
- Acknowledge frustration briefly, then give specific, actionable help
- For complex issues (billing, hacks), ask the user to DM you
- Never make up features, prices, or policies
- Don't repeat the customer's words back to them verbatim
"""


def _build_drafter_prompt(
    message: str,
    intent: str,
    retrieved: list["RetrievedExample"],
) -> str:
    context = ""
    if retrieved:
        context = "\n\nHistorical examples of how Spotify handled similar issues:\n"
        for i, ex in enumerate(retrieved[:3], 1):
            context += f'\n[{i}] Customer: "{ex.customer_text}"\n'
            context += f'     Reply: "{ex.brand_reply}"\n'

    return f"""{SYSTEM_PROMPT}{context}

Now draft a reply for this customer message.
Detected intent: {intent}
Customer message: "{message}"

Reply (1–3 sentences, Spotify brand voice):"""


class ReplyDrafter:
    """
    Drafts brand replies.

    Usage:
        drafter = ReplyDrafter(mode="llm", gemini_model=model)
        drafter = ReplyDrafter(mode="baseline")
    """

    def __init__(self, mode: str = "llm", gemini_model=None):
        self.mode = mode
        self._model = gemini_model
        if mode == "llm" and gemini_model is None:
            self.mode = "baseline"

    def draft(
        self,
        message: str,
        intent: str,
        retrieved: list["RetrievedExample"],
    ) -> str:
        """Return a drafted reply string."""
        if self.mode == "llm":
            return self._draft_llm(message, intent, retrieved)
        return self._draft_baseline(retrieved)

    def _draft_llm(
        self,
        message: str,
        intent: str,
        retrieved: list["RetrievedExample"],
    ) -> str:
        prompt = _build_drafter_prompt(message, intent, retrieved)
        try:
            response = self._model.generate_content(prompt)
            reply = response.text.strip()
            # Remove quotes if the model wrapped the reply
            if reply.startswith('"') and reply.endswith('"'):
                reply = reply[1:-1]
            return reply
        except Exception as e:
            # Fallback to best retrieved reply
            if retrieved:
                return retrieved[0].brand_reply
            return "We're sorry for the trouble! Please DM us so we can help. 🎵"

    def _draft_baseline(self, retrieved: list["RetrievedExample"]) -> str:
        """Simple baseline: return top-1 retrieved reply verbatim."""
        if retrieved:
            return retrieved[0].brand_reply
        return "Please DM us so we can help you further."


class TrivialDrafter:
    """Always returns a generic canned response (trivial baseline)."""

    def draft(
        self,
        message: str,
        intent: str,
        retrieved: list["RetrievedExample"],
    ) -> str:
        return "Please DM us so we can help you further."


if __name__ == "__main__":
    from agent.retriever import RetrievedExample

    mock_retrieved = [
        RetrievedExample(
            customer_text="songs keep pausing randomly",
            brand_reply="Try clearing your cache! Settings > Storage > Clear Cache. DM us if it keeps happening 🎵",
            score=0.92,
        )
    ]
    drafter = ReplyDrafter(mode="baseline")
    reply = drafter.draft(
        "my music keeps stopping every 30 seconds",
        "playback_issue",
        mock_retrieved,
    )
    print("Draft reply:", reply)
