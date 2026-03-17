"""
LLM interface. Keeps the engine model-agnostic.

Supports:
  - Anthropic Claude API (default)
  - OpenAI-compatible APIs
  - Mock mode for testing without API calls
"""

import os
import time


def make_claude_caller(model: str = "claude-sonnet-4-20250514",
                       max_tokens: int = 1500,
                       temperature: float = 0.8):
    """Create a caller function using the Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Install anthropic: pip install anthropic")

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    def call(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return call


def make_openai_caller(model: str = "gpt-4o",
                       max_tokens: int = 1500,
                       temperature: float = 0.8,
                       base_url: str = None):
    """Create a caller function using an OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)  # Uses OPENAI_API_KEY env var

    def call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    return call


def make_mock_caller():
    """Mock caller for testing without API access."""
    call_count = {"n": 0}

    def call(prompt: str) -> str:
        call_count["n"] += 1
        n = call_count["n"]

        # Detect what kind of prompt this is
        if "POSITION:" in prompt or "ARGUMENT:" in prompt:
            return f"""POSITION: This is mock position #{n}. The agent takes a moderate stance given the constraints.

ARGUMENT: In considering this question, the relevant factors include the stakeholders involved, the potential consequences, and the principles at stake. Given my particular situation and the pressures I face, I find myself drawn toward a position that balances competing concerns. I acknowledge the force of arguments on the other side, but my experience and my commitments lead me to prioritize stability and caution in this instance.

The deeper issue is whether any of us can truly separate our reasoning from our interests. I would like to believe I can, but honesty compels me to admit that my position aligns rather conveniently with my incentives.

CONFIDENCE: {0.4 + (n % 5) * 0.1:.1f}
MOVED_BY: none
PRIVATE_THOUGHT: I wonder if anyone in this room is actually reasoning from principle or if we are all just performing our roles."""
        elif "GAP" in prompt or "gap" in prompt:
            return ("The gap between ideal and actual reasoning in this deliberation is revealing. "
                    "Agents with strong financial incentives deviated most from their tradition's "
                    "recommendation, while those with lower personal stakes stayed closer to "
                    "principled positions. The most significant distortion came from reelection "
                    "pressure, which consistently pushed toward ambiguity over clarity. "
                    "This suggests that the incentive structure, not the quality of available "
                    "arguments, is the primary determinant of deliberative outcomes.")
        elif "narrative" in prompt.lower():
            return ("The room divided early and never fully reconciled. The scientist spoke first, "
                    "carefully, choosing words that committed her to nothing. The senator watched "
                    "the others before speaking, calibrating his position to the room's center of "
                    "gravity. The nonprofit director came in hot, unwilling to concede an inch, "
                    "her moral clarity both her strength and her limitation. The general said little "
                    "but what he said landed heavy — he had seen things the others had not, and "
                    "the weight of that classified knowledge bent the room toward caution. "
                    "The startup founder was the most honest, because she had the most to lose "
                    "and the least to gain from pretending otherwise.\n\n"
                    "What the room could not resolve was this: the same openness that democratizes "
                    "also endangers, and no one present had a framework for holding both truths "
                    "simultaneously without collapsing into either naive optimism or fearful restriction.")
        else:
            return (f"[Mock response #{n}] This tradition recommends careful consideration of "
                    "the competing values at stake, with particular attention to the human "
                    "consequences of the decision. The strongest argument available is that "
                    "we must weigh immediate harms against long-term benefits while remaining "
                    "honest about the limits of our foresight.")

    return call
