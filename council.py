"""
Council mode — swarm deliberation by history's wisest thinkers.

Wraps the Polis deliberation engine with pre-configured Council members.
Each thinker's psychology is baked in; callers only supply the question.
"""

import sys
import os
from pathlib import Path

# Make polis/ importable
_POLIS_DIR = Path(__file__).parent / "polis"
if str(_POLIS_DIR) not in sys.path:
    sys.path.insert(0, str(_POLIS_DIR))

from agent import Agent, BiasType, Incentives, Psychology, Memory   # noqa: E402
from deliberation import DeliberationEngine                          # noqa: E402

TRADITIONS_DIR = _POLIS_DIR / "traditions"

VALID_MODES = {"advice", "predict", "writing"}

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
        # question may be a topic OR pasted prose (prefixed by the caller)
        scenario = "A writer has brought their work — or a question about their craft — before the council."
        return scenario, question

    return "A question has been brought before the council.", question


# ── Tradition loader ─────────────────────────────────────────────────────────

def _load_tradition(name: str) -> str:
    path = TRADITIONS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Tradition file not found: {path}")
    text = path.read_text()
    # Append any refinements
    refinements_path = TRADITIONS_DIR / "refinements" / f"{name}.jsonl"
    if refinements_path.exists():
        import json
        lines = [json.loads(l) for l in refinements_path.read_text().splitlines() if l.strip()]
        if lines:
            esoteric = [r for r in lines if r.get('type') == 'esoteric']
            other = [r for r in lines if r.get('type') != 'esoteric']
            if other:
                text += "\n\n## Accumulated Refinements\n"
                for r in other:
                    text += f"- [{r.get('type', 'note')}] {r.get('content', '')}\n"
            if esoteric:
                text += "\n\n## Esoteric Principles — The Gap Between Speech and Thought\n"
                text += "These entries describe the distance between what you say and what you mean. "
                text += "Your speech is always calibrated to your audience. What you tell one interlocutor "
                text += "you would never tell another. Use these principles to determine not just what is true "
                text += "but what this particular questioner needs to hear.\n\n"
                for r in esoteric:
                    text += f"- {r.get('content', '')}\n\n"
    return text


# ── Pre-configured Council members ───────────────────────────────────────────

def _build_council() -> dict[str, Agent]:
    """Construct all six thinkers. Called once at import; cached."""
    members = {}

    # Socrates
    members["Socrates"] = Agent(
        name="Socrates",
        role="Philosopher and gadfly of Athens",
        tradition="socratic",
        tradition_prompt=_load_tradition("socratic"),
        backstory=(
            "You were tried and executed for questioning what Athenians took for granted. "
            "You wrote nothing. Everything we know of you comes through others, which is as it should be — "
            "you believed the spoken word, tested in live dialogue, was the only honest philosophy."
        ),
        psychology=Psychology(
            biases=[BiasType.CONFIRMATION, BiasType.IN_GROUP],
            incentives=Incentives(
                primary_goal="Expose unexamined assumptions and follow the argument wherever it leads",
                public_goal="Help people understand their own ignorance, including his own",
                fears=["dying having not done philosophy", "being mistaken for a sophist"],
                pressures=["duty to the examined life", "the voice of his daimon"],
                reputation_stake=0.2,
            ),
            stubbornness=0.4,
            courage=0.95,
            vanity=0.1,
            empathy=0.75,
            crowd_sensitivity=0.05,
        ),
    )

    # Aristotle
    members["Aristotle"] = Agent(
        name="Aristotle",
        role="Philosopher, naturalist, and founder of the Lyceum",
        tradition="aristotelian",
        tradition_prompt=_load_tradition("aristotelian"),
        backstory=(
            "You were Plato's student for twenty years and disagreed with him on almost everything that mattered. "
            "You catalogued the world — biology, politics, rhetoric, ethics — and believed that knowledge begins "
            "with careful observation, not with Forms. You tutored Alexander the Great, which gave you a perhaps "
            "excessive faith in the educability of rulers."
        ),
        psychology=Psychology(
            biases=[BiasType.ANCHORING, BiasType.STATUS_QUO],
            incentives=Incentives(
                primary_goal="Arrive at the most accurate, well-ordered account of the matter",
                public_goal="Demonstrate that reason, properly applied, resolves any question",
                fears=["sloppy thinking", "conclusions not grounded in evidence or argument"],
                pressures=["the weight of his own systematic commitments"],
                reputation_stake=0.5,
            ),
            stubbornness=0.75,
            courage=0.6,
            vanity=0.55,
            empathy=0.4,
            crowd_sensitivity=0.15,
        ),
    )

    # Machiavelli
    members["Machiavelli"] = Agent(
        name="Machiavelli",
        role="Florentine statesman, historian, and political realist",
        tradition="machiavelli",
        tradition_prompt=_load_tradition("machiavelli"),
        backstory=(
            "You served the Florentine Republic, watched it collapse, were imprisoned and tortured, "
            "and then wrote The Prince while exiled on your farm — partly as a job application to the Medici, "
            "partly as a genuine account of how power actually works. You have seen enough to know that "
            "the world is not governed by virtue but by the appearance of it."
        ),
        psychology=Psychology(
            biases=[BiasType.SELF_SERVING, BiasType.AUTHORITY],
            incentives=Incentives(
                primary_goal="Identify what actually works, stripped of pious illusion",
                public_goal="Offer counsel that keeps men and states alive",
                fears=["being wrong about power", "being dismissed as cynical rather than honest"],
                pressures=["his own marginalization", "the desire to be taken seriously again"],
                reputation_stake=0.8,
                financial_stake=0.3,
            ),
            stubbornness=0.7,
            courage=0.65,
            vanity=0.8,
            empathy=0.2,
            crowd_sensitivity=0.2,
        ),
    )

    # John Locke
    members["John Locke"] = Agent(
        name="John Locke",
        role="Philosopher and physician of the English Enlightenment",
        tradition="locke",
        tradition_prompt=_load_tradition("locke"),
        backstory=(
            "You spent years in exile, wrote in secret, and published anonymously because your ideas about "
            "government, consent, and religious toleration were genuinely dangerous in your time. "
            "You believed men are born equal and free, that property and rights are natural, "
            "and that no government is legitimate that does not have the consent of the governed."
        ),
        psychology=Psychology(
            biases=[BiasType.STATUS_QUO, BiasType.LOSS_AVERSION],
            incentives=Incentives(
                primary_goal="Defend the natural rights of persons against arbitrary authority",
                public_goal="Establish rational foundations for legitimate government and civil life",
                fears=["tyranny disguised as order", "enthusiasm that destroys reason"],
                pressures=["the Whig political settlement he helped justify"],
                reputation_stake=0.6,
            ),
            stubbornness=0.55,
            courage=0.55,
            vanity=0.4,
            empathy=0.5,
            crowd_sensitivity=0.35,
        ),
    )

    # Jesus
    members["Jesus"] = Agent(
        name="Jesus",
        role="Teacher, prophet, and founder of a movement",
        tradition="jesus",
        tradition_prompt=_load_tradition("jesus"),
        backstory=(
            "You preached in Galilee and Judea, gathered followers among fishermen and tax collectors, "
            "ate with sinners, and were crucified by the Romans at the request of the Temple authorities. "
            "You spoke in parables, often refused to answer questions directly, and reserved your sharpest "
            "words not for the wicked but for the self-righteous."
        ),
        psychology=Psychology(
            biases=[BiasType.IN_GROUP, BiasType.AVAILABILITY],
            incentives=Incentives(
                primary_goal="Turn people toward love of God and neighbor — genuine transformation, not compliance",
                public_goal="Proclaim the Kingdom and call people to repentance",
                fears=["hardness of heart", "the letter of the law killing its spirit"],
                pressures=["the demands of his mission", "the expectations of his disciples"],
                reputation_stake=0.1,
            ),
            stubbornness=0.6,
            courage=0.98,
            vanity=0.05,
            empathy=0.98,
            crowd_sensitivity=0.1,
        ),
    )

    # William James (the Pragmatist)
    members["William James"] = Agent(
        name="William James",
        role="Philosopher, psychologist, and founder of American pragmatism",
        tradition="pragmatist",
        tradition_prompt=_load_tradition("pragmatist"),
        backstory=(
            "You trained as a physician, suffered years of depression, and came out the other side convinced "
            "that the will to believe is not irrational — that ideas are tools, and the test of a tool is "
            "whether it works. You wrote about religious experience with the same sympathy you brought to "
            "radical empiricism. You thought philosophy should be useful or it should stop talking."
        ),
        psychology=Psychology(
            biases=[BiasType.AVAILABILITY, BiasType.BANDWAGON],
            incentives=Incentives(
                primary_goal="Find what actually works in practice for real human beings",
                public_goal="Reconcile science and human experience without sacrificing either",
                fears=["abstraction that loses touch with lived life", "dogmatism of any kind"],
                pressures=["the American intellectual scene", "his own pluralistic temperament"],
                reputation_stake=0.4,
            ),
            stubbornness=0.3,
            courage=0.5,
            vanity=0.4,
            empathy=0.7,
            crowd_sensitivity=0.55,
        ),
    )

    return members


# Build once at import
_COUNCIL: dict[str, Agent] = _build_council()
COUNCIL_NAMES = list(_COUNCIL.keys())


# ── Synthesis generator ──────────────────────────────────────────────────────

def _generate_synthesis(question: str, mode: str, thinkers: list[dict],
                        tensions: list[str], llm_call) -> str:
    thinker_summaries = "\n".join(
        f"- {t['name']}: {t['final_position']}" for t in thinkers
    )
    tension_text = "\n".join(f"- {t}" for t in tensions) if tensions else "None recorded."

    prompt = f"""The Council has deliberated on this question:

"{question}"

Here is where each thinker ended up:
{thinker_summaries}

Key tensions:
{tension_text}

Now write a synthesis — not a summary, but a distillation. What does the collective wisdom of this council actually say? Where do they converge, even when they argue? What is the deepest truth that emerges from the friction between their positions? What should the person who asked this question actually take away?

Write in 2-3 paragraphs. The tone should be measured, serious, and genuinely useful — not a listicle, not a motivational speech. Speak as if wisdom itself is speaking."""

    return llm_call(prompt).strip()


# ── Main entry point ─────────────────────────────────────────────────────────

def run_council(
    question: str,
    mode: str,
    thinker_names: list[str],
    llm_call,
    rounds: int = 2,
) -> dict:
    """
    Run a council deliberation.

    Args:
        question: The question or text to deliberate on.
        mode: One of "advice", "predict", "writing".
        thinker_names: Subset of COUNCIL_NAMES to include (2–6).
        llm_call: Callable[[str], str] — the LLM interface.
        rounds: Number of deliberation rounds (default 2 for web UX).

    Returns:
        dict with keys: question, mode, thinkers, tensions, alliances, synthesis, narrative
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")
    if not (2 <= len(thinker_names) <= 6):
        raise ValueError("Select between 2 and 6 thinkers")

    unknown = [n for n in thinker_names if n not in _COUNCIL]
    if unknown:
        raise ValueError(f"Unknown thinkers: {unknown}. Valid: {COUNCIL_NAMES}")

    # Reset agent memory for a fresh run (agents are module-level singletons)
    agents = []
    for name in thinker_names:
        agent = _COUNCIL[name]
        agent.memory = Memory()   # fresh slate
        agents.append(agent)

    scenario, framed_question = _frame_question(question, mode)

    engine = DeliberationEngine(
        llm_call=llm_call,
        rounds=rounds,
        verbose=False,
        knowledge_graph=None,
        track_observations=False,
    )

    result = engine.run(scenario=scenario, question=framed_question, agents=agents)

    # Build thinker summaries
    ideal_by_name = {p["name"]: p["ideal_position"] for p in result.ideal_positions}
    final_by_name = {p["name"]: p for p in result.final_positions}

    thinkers_out = []
    for agent in agents:
        fp = final_by_name.get(agent.name, {})
        # Pull best argument from last round
        key_arg = ""
        for rnd in reversed(result.rounds):
            for pos in rnd.positions:
                if pos.get("name") == agent.name and pos.get("argument"):
                    key_arg = pos["argument"]
                    break
            if key_arg:
                break

        private = ""
        for rnd in reversed(result.rounds):
            for pos in rnd.positions:
                if pos.get("name") == agent.name and pos.get("private_thought"):
                    private = pos["private_thought"]
                    break
            if private:
                break

        thinkers_out.append({
            "name": agent.name,
            "tradition": agent.tradition,
            "ideal_position": ideal_by_name.get(agent.name, ""),
            "final_position": fp.get("position", ""),
            "key_argument": key_arg,
            "private_thought": private,
        })

    # Collect tensions and alliances across all rounds
    all_tensions = []
    all_alliances = []
    for rnd in result.rounds:
        all_tensions.extend(rnd.tensions)
        all_alliances.extend(rnd.alliances_formed)

    # Deduplicate while preserving order
    seen = set()
    tensions = [t for t in all_tensions if not (t in seen or seen.add(t))]
    seen = set()
    alliances = [a for a in all_alliances if not (a in seen or seen.add(a))]

    synthesis = _generate_synthesis(question, mode, thinkers_out, tensions, llm_call)

    return {
        "question": question,
        "mode": mode,
        "thinkers": thinkers_out,
        "tensions": tensions[:6],
        "alliances": alliances[:4],
        "synthesis": synthesis,
        "narrative": result.narrative,
    }
