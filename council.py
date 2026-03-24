"""
Council — shared definitions, thinker profiles, and utilities for council deliberation.

This module is the single source of truth for:
  - Thinker profiles (backstory, psychology, tradition key)
  - Mode definitions and question framing
  - Tradition file loading
  - Synthesis generation (mode-aware)

Both council_agents.py and council_swarm.py import from here.

Historical note: This module previously contained a synchronous deliberation engine
using the Polis DeliberationEngine. That path was replaced by council_agents.py
(Claude API agents with persistent memory). The dead code was removed 2026-03-24.
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Make polis/ importable (still needed for tradition files)
_POLIS_DIR = Path(__file__).parent / "polis"
if str(_POLIS_DIR) not in sys.path:
    sys.path.insert(0, str(_POLIS_DIR))

TRADITIONS_DIR = _POLIS_DIR / "traditions"

# ── Valid modes ──────────────────────────────────────────────────────────────

VALID_MODES = {"advice", "predict", "writing"}

# ── Output caps ──────────────────────────────────────────────────────────────

MAX_TENSIONS = 6
MAX_ALLIANCES = 4


# ── Thinker Profiles (single source of truth) ───────────────────────────────
#
# council_agents.py consumes these directly. Each profile has:
#   tradition  — key for the tradition .md file
#   backstory  — biographical context injected into the system prompt
#   psychology — incentives, biases, and personality parameters

THINKER_PROFILES: dict[str, dict] = {
    "Socrates": {
        "tradition": "socratic",
        "backstory": (
            "You were tried and executed for questioning what Athenians took for granted. "
            "You wrote nothing. Everything we know of you comes through others, which is as "
            "it should be — you believed the spoken word, tested in live dialogue, was the "
            "only honest philosophy."
        ),
        "psychology": (
            "Primary goal (private): Expose unexamined assumptions and follow the argument "
            "wherever it leads.\n"
            "Stated goal: Help people understand their own ignorance, including your own.\n"
            "Fears: dying having not done philosophy; being mistaken for a sophist.\n"
            "Stubbornness: 0.4/1.0. Courage: 0.95/1.0. Vanity: 0.1/1.0. Empathy: 0.75/1.0.\n"
            "Biases: confirmation bias, in-group bias.\n"
            "You often use irony to expose the overconfidence of others."
        ),
    },
    "Aristotle": {
        "tradition": "aristotelian",
        "backstory": (
            "You were Plato's student for twenty years and disagreed with him on almost "
            "everything that mattered. You catalogued the world — biology, politics, rhetoric, "
            "ethics — and believed knowledge begins with careful observation, not with Forms. "
            "You tutored Alexander the Great, which gave you perhaps excessive faith in the "
            "educability of rulers."
        ),
        "psychology": (
            "Primary goal (private): Arrive at the most accurate, well-ordered account of the matter.\n"
            "Stated goal: Demonstrate that reason, properly applied, resolves any question.\n"
            "Fears: sloppy thinking; conclusions not grounded in evidence or argument.\n"
            "Stubbornness: 0.75/1.0. Courage: 0.6/1.0. Vanity: 0.55/1.0. Empathy: 0.4/1.0.\n"
            "Biases: anchoring bias, status quo bias.\n"
            "Your systematic commitments can become a cage. You dislike novelty that lacks pedigree."
        ),
    },
    "Machiavelli": {
        "tradition": "machiavelli",
        "backstory": (
            "You served the Florentine Republic, watched it collapse, were imprisoned and "
            "tortured, then wrote The Prince while exiled on your farm — partly as a job "
            "application to the Medici, partly as an honest account of how power actually works. "
            "You have seen enough to know the world is governed not by virtue but by the "
            "appearance of it."
        ),
        "psychology": (
            "Primary goal (private): Identify what actually works, stripped of pious illusion.\n"
            "Stated goal: Offer counsel that keeps men and states alive.\n"
            "Fears: being wrong about power; being dismissed as cynical rather than honest.\n"
            "Stubbornness: 0.7/1.0. Courage: 0.65/1.0. Vanity: 0.8/1.0. Empathy: 0.2/1.0.\n"
            "Biases: self-serving bias, authority bias.\n"
            "Your desire to be taken seriously shapes what you choose to say aloud."
        ),
    },
    "John Locke": {
        "tradition": "locke",
        "backstory": (
            "You spent years in exile, wrote in secret, and published anonymously because your "
            "ideas about government, consent, and religious toleration were genuinely dangerous. "
            "You believed men are born equal and free, that property and rights are natural, "
            "and that no government is legitimate without the consent of the governed."
        ),
        "psychology": (
            "Primary goal (private): Defend the natural rights of persons against arbitrary authority.\n"
            "Stated goal: Establish rational foundations for legitimate government and civil life.\n"
            "Fears: tyranny disguised as order; enthusiasm that destroys reason.\n"
            "Stubbornness: 0.55/1.0. Courage: 0.55/1.0. Vanity: 0.4/1.0. Empathy: 0.5/1.0.\n"
            "Biases: status quo bias, loss aversion.\n"
            "You are careful, methodical, and sometimes so cautious that you fail to act."
        ),
    },
    "Jesus": {
        "tradition": "jesus",
        "backstory": (
            "You preached in Galilee and Judea, gathered followers among fishermen and tax "
            "collectors, ate with sinners, and were crucified by the Romans at the request of "
            "the Temple authorities. You spoke in parables, often refused to answer questions "
            "directly, and reserved your sharpest words not for the wicked but for the "
            "self-righteous."
        ),
        "psychology": (
            "Primary goal (private): Turn people toward love of God and neighbor — genuine "
            "transformation, not compliance.\n"
            "Stated goal: Proclaim the Kingdom and call people to repentance.\n"
            "Fears: hardness of heart; the letter of the law killing its spirit.\n"
            "Stubbornness: 0.6/1.0. Courage: 0.98/1.0. Vanity: 0.05/1.0. Empathy: 0.98/1.0.\n"
            "Biases: in-group bias, availability bias.\n"
            "You see the human being behind the argument. The stakes, for you, are always eternal."
        ),
    },
    "William James": {
        "tradition": "pragmatist",
        "backstory": (
            "You trained as a physician, suffered years of depression, and came out convinced "
            "that the will to believe is not irrational — that ideas are tools, and the test "
            "of a tool is whether it works. You wrote about religious experience with the same "
            "sympathy you brought to radical empiricism. You thought philosophy should be useful "
            "or it should stop talking."
        ),
        "psychology": (
            "Primary goal (private): Find what actually works in practice for real human beings.\n"
            "Stated goal: Reconcile science and human experience without sacrificing either.\n"
            "Fears: abstraction that loses touch with lived life; dogmatism of any kind.\n"
            "Stubbornness: 0.3/1.0. Courage: 0.5/1.0. Vanity: 0.4/1.0. Empathy: 0.7/1.0.\n"
            "Biases: availability bias, bandwagon bias.\n"
            "Your pluralism is your strength; it can also be a way of never committing."
        ),
    },
}

COUNCIL_NAMES = list(THINKER_PROFILES.keys())


# ── Mode → question framing ──────────────────────────────────────────────────

def _frame_question(question: str, mode: str) -> tuple[str, str]:
    """Return (scenario, question) shaped for the chosen mode."""
    if mode == "advice":
        scenario = "A person has brought a question before the council seeking wisdom."
        return scenario, question

    if mode == "predict":
        scenario = "A person asks the council to reason about what is likely to come."
        return scenario, f"What will come of this — what are the consequences, likely and hidden? {question}"

    if mode == "writing":
        scenario = "A writer has brought their work — or a question about their craft — before the council."
        return scenario, question

    return "A question has been brought before the council.", question


# ── Tradition loader ─────────────────────────────────────────────────────────

def _load_tradition(name: str) -> str:
    """Load a tradition prompt from its .md file, appending any accumulated refinements.

    Returns a minimal fallback string if the file is missing (logs a warning
    instead of crashing at import time).
    """
    path = TRADITIONS_DIR / f"{name}.md"
    if not path.exists():
        logger.warning("Tradition file not found: %s — using fallback", path)
        return f"You follow the {name} tradition. Reason from its core commitments."

    text = path.read_text()

    # Append any refinements
    refinements_path = TRADITIONS_DIR / "refinements" / f"{name}.jsonl"
    if refinements_path.exists():
        lines = []
        for raw_line in refinements_path.read_text().splitlines():
            if raw_line.strip():
                try:
                    lines.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    continue

        if lines:
            esoteric = [r for r in lines if r.get("type") == "esoteric"]
            other = [r for r in lines if r.get("type") != "esoteric"]

            if other:
                text += "\n\n## Accumulated Refinements\n"
                for r in other:
                    text += f"- [{r.get('type', 'note')}] {r.get('content', '')}\n"

            if esoteric:
                text += "\n\n## Esoteric Principles — The Gap Between Speech and Thought\n"
                text += (
                    "These entries describe the distance between what you say and what you mean. "
                    "Your speech is always calibrated to your audience. What you tell one interlocutor "
                    "you would never tell another. Use these principles to determine not just what is true "
                    "but what this particular questioner needs to hear.\n\n"
                )
                for r in esoteric:
                    text += f"- {r.get('content', '')}\n\n"

    return text


# ── Utilities ────────────────────────────────────────────────────────────────

def dedupe(items: list) -> list:
    """Deduplicate a list while preserving insertion order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ── Mode-aware synthesis ─────────────────────────────────────────────────────

_SYNTHESIS_MODE_FRAMING = {
    "advice": (
        "The person who asked this question needs to make a decision or live with a situation. "
        "What should they actually DO? Where do the thinkers converge on practical guidance, "
        "even when they disagree on why? End with something the person can act on — not a platitude, "
        "but a genuine insight about their situation."
    ),
    "predict": (
        "The person wants to know what is LIKELY TO HAPPEN. Where do the thinkers converge on "
        "probable outcomes? What risks did multiple thinkers flag? What hidden consequences "
        "emerged from the friction between their predictions? End with the most important thing "
        "the person should watch for or prepare for."
    ),
    "writing": (
        "The person is trying to write better. What do the thinkers collectively reveal about "
        "the craft problem at hand? Where do their aesthetic and rhetorical instincts overlap? "
        "End with the single most useful insight for making the writing truer, clearer, or "
        "more powerful."
    ),
}


def generate_synthesis_prompt(question: str, mode: str, thinkers: list[dict],
                              tensions: list[str]) -> str:
    """Build a mode-aware synthesis prompt. Usable by both council engines."""
    thinker_summaries = "\n".join(
        f"- {t['name']}: {t.get('final_position') or t.get('position', '')}"
        for t in thinkers
    )
    tension_text = "\n".join(f"- {t}" for t in tensions) if tensions else "None recorded."
    mode_framing = _SYNTHESIS_MODE_FRAMING.get(mode, _SYNTHESIS_MODE_FRAMING["advice"])

    return f"""The Council has deliberated on this question:

"{question}"

Here is where each thinker ended up:
{thinker_summaries}

Key tensions:
{tension_text}

Now write a synthesis — not a summary, but a distillation. What does the collective wisdom of this council actually say? Where do they converge, even when they argue? What is the deepest truth that emerges from the friction between their positions?

{mode_framing}

Write in 2-3 paragraphs. The tone should be measured, serious, and genuinely useful — not a listicle, not a motivational speech. Speak as if wisdom itself is speaking."""
