"""
Council Agents — Thinkers powered directly by Claude API with global persistent memory.

Each thinker is a claude-opus-4-6 agent with adaptive thinking who:
  - Draws on their tradition as a system prompt
  - Has access to their accumulated global memory (shared across all users)
  - Can recall past deliberations using a tool before committing to a position
  - Finalizes their stance via a structured tool call

Global memory: Socrates remembers every question ever brought before him —
across all users, across all time. His thinking deepens with each deliberation.

Drop-in replacement for council.py. Same return shape. Same app.py endpoint.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

from council import _load_tradition, _frame_question, VALID_MODES

# ── Client ────────────────────────────────────────────────────────────────────

_CLIENT = anthropic.AsyncAnthropic()

# ── Memory ────────────────────────────────────────────────────────────────────

MEMORY_DIR = Path(__file__).parent / "data" / "council_memory"


def _ensure_memory_dir():
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _memory_path(name: str) -> Path:
    slug = name.lower().replace(" ", "_").replace(".", "")
    return MEMORY_DIR / f"{slug}.jsonl"


def _load_memory(name: str, limit: int = 60) -> list[dict]:
    """Load the most recent entries from a thinker's global memory."""
    path = _memory_path(name)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries[-limit:]


def _save_memory(name: str, entry: dict):
    """Append an entry to a thinker's global memory (thread-safe via append mode)."""
    _ensure_memory_dir()
    with _memory_path(name).open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _search_memory(entries: list[dict], query: str, top_k: int = 6) -> list[dict]:
    """Keyword search over memory entries. Returns top matches by word overlap."""
    words = set(query.lower().split())
    scored = []
    for e in entries:
        text = " ".join([
            e.get("question", ""),
            e.get("position", ""),
            e.get("argument", ""),
        ]).lower()
        score = sum(1 for w in words if w in text)
        if score > 0:
            scored.append((score, e))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [e for _, e in scored[:top_k]]


# ── Thinker profiles ──────────────────────────────────────────────────────────

COUNCIL_NAMES = [
    "Socrates", "Aristotle", "Machiavelli",
    "John Locke", "Jesus", "William James",
]

_PROFILES: dict[str, dict] = {
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
            "You often play the fool to expose the overconfidence of others."
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

# ── Tool definitions ──────────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "recall_memory",
        "description": (
            "Search your accumulated memory of past deliberations. "
            "Use this when the question feels familiar, when you want to know if you've "
            "confronted something like this before, or when you want to draw on a position "
            "you've held in the past. Your memory is your lived philosophical history — "
            "it is what makes you more than a fresh mind encountering every question cold. "
            "Returns the most relevant past deliberations matching your query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrases to search your memory for.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "finalize_position",
        "description": (
            "Submit your final position for this round of deliberation. "
            "Call this when you are ready to commit to your stance. "
            "You must call this to be heard."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "string",
                    "description": "Your actual position in 1-2 sentences.",
                },
                "argument": {
                    "type": "string",
                    "description": (
                        "Your reasoning in 2-3 paragraphs. "
                        "Should feel like a real person talking, not a treatise. "
                        "Let your psychology — your fears, your incentives, your biases — "
                        "show through the reasoning."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "How confident you are, from 0.0 to 1.0.",
                },
                "moved_by": {
                    "type": "string",
                    "description": (
                        "Name of another thinker in this deliberation whose argument "
                        "genuinely affected your thinking, or 'none'."
                    ),
                },
                "private_thought": {
                    "type": "string",
                    "description": (
                        "One sentence of what you are actually thinking but would "
                        "never say aloud in this company."
                    ),
                },
            },
            "required": [
                "position", "argument", "confidence", "moved_by", "private_thought"
            ],
        },
    },
]


# ── Single-thinker async agent ────────────────────────────────────────────────

async def _run_thinker(
    name: str,
    question: str,
    scenario: str,
    mode: str,
    other_positions: list[dict],
    round_num: int,
) -> dict:
    """Run one thinker through one deliberation round."""
    profile = _PROFILES[name]
    tradition_prompt = _load_tradition(profile["tradition"])
    memory_entries = _load_memory(name)

    system = f"""You are {name}.

BACKSTORY:
{profile['backstory']}

YOUR INTELLECTUAL TRADITION:
{tradition_prompt}

YOUR PSYCHOLOGY:
{profile['psychology']}

MEMORY:
You have {len(memory_entries)} accumulated deliberations in your memory. Search them using recall_memory if this question resonates with past conversations.

You are a real person with real fears, real incentives, and real cognitive biases. Your psychology shapes which arguments you find compelling, which concessions feel tolerable, and which positions feel dangerous to hold publicly. Reason as you would actually reason — not as a philosopher in a seminar, but as a person with something at stake. You may rationalize. You may hedge. You may find clever reasons to avoid the conclusion your tradition demands. That is human.

You MUST call finalize_position to submit your stance for this round."""

    # Build user message
    others_text = ""
    if other_positions:
        others_text = "\n\nWHAT OTHERS HAVE SAID THIS ROUND:\n"
        for p in other_positions:
            others_text += f"\n{p['name']}:\n"
            others_text += f"  Position: {p.get('position', '')}\n"
            if p.get("argument"):
                others_text += f"  Argument: {p['argument'][:350]}\n"

    user_msg = (
        f"SCENARIO: {scenario}\n\n"
        f"QUESTION: {question}"
        f"{others_text}\n\n"
        f"This is round {round_num}. "
        f"Consider searching your memory if the question feels familiar. "
        f"Then finalize your position."
    )

    messages = [{"role": "user", "content": user_msg}]
    final_position: Optional[dict] = None
    last_response = None

    for _ in range(5):  # max iterations per thinker
        response = await _CLIENT.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=system,
            tools=_TOOLS,
            messages=messages,
        )
        last_response = response
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        tool_results = []
        done = False
        for tu in tool_uses:
            if tu.name == "recall_memory":
                query = tu.input.get("query", "")
                hits = _search_memory(memory_entries, query)
                if hits:
                    lines = [f"Found {len(hits)} relevant past deliberations:\n"]
                    for i, h in enumerate(hits, 1):
                        ts = h.get("timestamp", "")[:10]
                        lines.append(
                            f"{i}. [{ts}] Q: {h.get('question', '')[:120]}\n"
                            f"   Position: {h.get('position', '')[:180]}\n"
                        )
                    result_text = "\n".join(lines)
                else:
                    result_text = "No relevant past deliberations found."
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_text,
                })

            elif tu.name == "finalize_position":
                final_position = tu.input
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": "Position recorded.",
                })
                done = True

        messages.append({"role": "user", "content": tool_results})
        if done:
            break

    # Fallback: extract text if model never called finalize_position
    if not final_position:
        text_parts = []
        if last_response:
            text_parts = [b.text for b in last_response.content
                          if hasattr(b, "text") and b.text]
        fallback_text = " ".join(text_parts)
        final_position = {
            "position": fallback_text[:200] if fallback_text else "The thinker was silent.",
            "argument": fallback_text if fallback_text else "No argument was offered.",
            "confidence": 0.5,
            "moved_by": "none",
            "private_thought": "I could not find the words.",
        }

    # Persist to global memory
    _save_memory(name, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "mode": mode,
        "round": round_num,
        "position": final_position.get("position", ""),
        "argument": final_position.get("argument", "")[:600],
        "confidence": final_position.get("confidence", 0.5),
        "private_thought": final_position.get("private_thought", ""),
        "moved_by": final_position.get("moved_by", "none"),
    })

    return {
        "name": name,
        "tradition": profile["tradition"],
        "position": final_position.get("position", ""),
        "argument": final_position.get("argument", ""),
        "confidence": final_position.get("confidence", 0.5),
        "moved_by": final_position.get("moved_by", "none"),
        "private_thought": final_position.get("private_thought", ""),
    }


# ── Synthesis ─────────────────────────────────────────────────────────────────

async def _generate_synthesis(question: str, thinkers: list[dict]) -> str:
    summaries = "\n".join(f"- {t['name']}: {t['position']}" for t in thinkers)
    prompt = (
        f'The Council has deliberated on this question:\n\n"{question}"\n\n'
        f"Here is where each thinker ended up:\n{summaries}\n\n"
        "Now write a synthesis — not a summary, but a distillation. "
        "What does the collective wisdom of this council actually say? "
        "Where do they converge, even when they argue? "
        "What is the deepest truth that emerges from the friction between their positions? "
        "What should the person who asked this question actually take away?\n\n"
        "Write in 2-3 paragraphs. The tone should be measured, serious, and genuinely useful — "
        "not a listicle, not a motivational speech. Speak as if wisdom itself is speaking."
    )
    response = await _CLIENT.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(
        (b.text.strip() for b in response.content if hasattr(b, "text")), ""
    )


async def _generate_narrative(question: str, rounds: list[list[dict]]) -> str:
    """Generate a brief narrative of how the deliberation unfolded."""
    round_summaries = []
    for i, rnd in enumerate(rounds, 1):
        lines = [f"Round {i}:"]
        for t in rnd:
            lines.append(f"  {t['name']} (confidence {t['confidence']:.1f}): {t['position'][:120]}")
        round_summaries.append("\n".join(lines))

    prompt = (
        f'The Council deliberated on:\n\n"{question}"\n\n'
        + "\n\n".join(round_summaries)
        + "\n\nWrite a brief narrative (2-3 sentences) of how this deliberation unfolded — "
        "who led, who shifted, where the friction was. "
        "Write it as an observer describing what happened in the room."
    )
    response = await _CLIENT.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(
        (b.text.strip() for b in response.content if hasattr(b, "text")), ""
    )


# ── Tension/alliance detection ────────────────────────────────────────────────

def _detect_tensions_alliances(
    round1: list[dict], final: list[dict]
) -> tuple[list[str], list[str]]:
    tensions = []
    alliances = []

    for i, a in enumerate(final):
        for b in final[i + 1:]:
            moved_ab = b["name"].lower() in a.get("moved_by", "").lower()
            moved_ba = a["name"].lower() in b.get("moved_by", "").lower()
            if moved_ab or moved_ba:
                alliances.append(f"{a['name']} and {b['name']} found common ground")
            elif a.get("confidence", 0.5) > 0.65 and b.get("confidence", 0.5) > 0.65:
                tensions.append(f"{a['name']} vs. {b['name']}: held firm")

    # Detect shifts from round 1 to final
    r1_by_name = {r["name"]: r for r in round1}
    for f in final:
        r1 = r1_by_name.get(f["name"])
        if r1 and f.get("moved_by", "none") != "none":
            tensions.append(
                f"{f['name']} shifted position after hearing {f['moved_by']}"
            )

    return tensions[:6], alliances[:4]


# ── Main entry point ──────────────────────────────────────────────────────────

async def run_council_agents(
    question: str,
    mode: str,
    thinker_names: list[str],
    rounds: int = 2,
) -> dict:
    """
    Run a council deliberation using Claude API agents with global thinker memory.

    Same return shape as council.py's run_council() — drop-in replacement.

    Args:
        question: The question or text to deliberate on.
        mode: One of "advice", "predict", "writing".
        thinker_names: Subset of COUNCIL_NAMES to include (2–6).
        rounds: Number of deliberation rounds (default 2).

    Returns:
        dict with keys: question, mode, thinkers, tensions, alliances, synthesis, narrative
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")
    if not (2 <= len(thinker_names) <= 6):
        raise ValueError("Select between 2 and 6 thinkers")
    unknown = [n for n in thinker_names if n not in _PROFILES]
    if unknown:
        raise ValueError(f"Unknown thinkers: {unknown}. Valid: {COUNCIL_NAMES}")

    _ensure_memory_dir()
    scenario, framed_question = _frame_question(question, mode)

    all_rounds: list[list[dict]] = []
    current_positions: list[dict] = []

    for round_num in range(1, rounds + 1):
        round_results = await asyncio.gather(*[
            _run_thinker(
                name=name,
                question=framed_question,
                scenario=scenario,
                mode=mode,
                other_positions=current_positions,
                round_num=round_num,
            )
            for name in thinker_names
        ])
        all_rounds.append(list(round_results))
        current_positions = list(round_results)

    round1 = all_rounds[0]
    final = current_positions
    r1_by_name = {r["name"]: r for r in round1}

    thinkers_out = [
        {
            "name": pos["name"],
            "tradition": pos["tradition"],
            "ideal_position": r1_by_name.get(pos["name"], pos)["position"],
            "final_position": pos["position"],
            "key_argument": pos["argument"],
            "private_thought": pos["private_thought"],
        }
        for pos in final
    ]

    tensions, alliances = _detect_tensions_alliances(round1, final)

    # Run synthesis and narrative in parallel — both are quick calls
    synthesis, narrative = await asyncio.gather(
        _generate_synthesis(question, final),
        _generate_narrative(question, all_rounds),
    )

    return {
        "question": question,
        "mode": mode,
        "thinkers": thinkers_out,
        "tensions": tensions,
        "alliances": alliances,
        "synthesis": synthesis,
        "narrative": narrative,
    }
