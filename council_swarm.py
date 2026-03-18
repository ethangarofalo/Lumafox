"""
Council Swarm — 40-agent philosophical swarm intelligence engine.

Architecture overview
─────────────────────
The current Council (council_agents.py) runs 6 named historical figures in
parallel and synthesizes their positions. That is a panel of experts.

This is different. This is a population.

TRADITIONS vs. AGENTS
  Each of the ~14 philosophical/theological traditions defines the *ideal* —
  what a person fully committed to that tradition *ought* to believe.
  Each agent in the swarm is a *person* who has been shaped by one or more
  traditions but lives at some distance from the ideal. The drift parameter
  encodes how far they've wandered.

  A "Christian with 0.7 drift" is not a theologian. They are someone who was
  raised Christian, absorbed therapeutic individualism from their culture,
  and still feels the pull of something transcendent without being able to
  articulate why. That person — not Aquinas — is who votes in this swarm.

POPULATION DISTRIBUTION
  The default 40-agent population is calibrated to approximate real cultural
  demographics in the contemporary West. The largest single group is
  "American contemporary" — the modal, loosely-held worldview of most people
  alive today. Ancient and systematic traditions are represented as minorities,
  which is accurate.

TWO-ROUND DELIBERATION
  Round 1: All 40 agents respond to the question independently.
  Round 2: Each agent sees a condensed summary of where the population landed
           and can hold their position, shift it, or become a dissenter.
           This is the swarm behavior — not parallel processing, but reaction.

OUTPUT
  Instead of individual thinker cards, the output is a population distribution:
  - Position clusters (where did people actually land?)
  - Tradition breakdown (how did each tradition vote?)
  - The majority view, the minority dissent, and any surprising convergences
  - A narrative of how opinion shifted between Round 1 and Round 2

GLOBAL MEMORY
  Each tradition (not each agent) has persistent global memory. "Christianity"
  remembers every question the swarm has considered. Agents draw on this
  tradition memory before committing to a Round 2 position.

Usage
─────
  from council_swarm import run_council_swarm

  result = await run_council_swarm(
      question="Is it ethical to lie to protect someone you love?",
      mode="advice",
      n_agents=40,
  )
"""

import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

# ── Client ────────────────────────────────────────────────────────────────────

_CLIENT = anthropic.AsyncAnthropic()

# ── Memory (per-tradition, shared across all users) ───────────────────────────

SWARM_MEMORY_DIR = Path(__file__).parent / "data" / "swarm_memory"


def _ensure_memory_dir():
    SWARM_MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _tradition_memory_path(tradition_key: str) -> Path:
    return SWARM_MEMORY_DIR / f"{tradition_key}.jsonl"


def _load_tradition_memory(tradition_key: str, limit: int = 40) -> list[dict]:
    path = _tradition_memory_path(tradition_key)
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


def _save_tradition_memory(tradition_key: str, entry: dict):
    _ensure_memory_dir()
    with _tradition_memory_path(tradition_key).open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _search_tradition_memory(entries: list[dict], query: str, top_k: int = 4) -> list[dict]:
    words = set(query.lower().split())
    scored = []
    for e in entries:
        text = " ".join([e.get("question", ""), e.get("cluster_summary", "")]).lower()
        score = sum(1 for w in words if w in text)
        if score > 0:
            scored.append((score, e))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [e for _, e in scored[:top_k]]


# ── Traditions — the "ought" ───────────────────────────────────────────────────
#
# Each tradition defines:
#   core_beliefs   — what this tradition holds to be true at its best
#   epistemic_style — how people in this tradition reason and evaluate claims
#   moral_frame    — how they evaluate right and wrong
#   characteristic_tensions — the internal contradictions people in this
#                              tradition actually live with
#   contemporary_drift — how the "real" version of this person differs from
#                        the ideal in today's world

TRADITIONS: dict[str, dict] = {

    "platonist": {
        "name": "Platonist",
        "core_beliefs": (
            "Reality is fundamentally ideal — the world of appearances is a shadow of the Forms. "
            "The soul is immortal and pre-exists the body. Knowledge is recollection. "
            "Justice, beauty, and the Good are real, objective, and more real than physical things. "
            "The philosopher's task is to turn away from the cave."
        ),
        "epistemic_style": (
            "Prioritizes reason over sensation. Suspicious of popular opinion and democratic consensus. "
            "Drawn to mathematics as a model of real knowledge. Believes truth is discovered, not constructed."
        ),
        "moral_frame": (
            "The good life consists in orienting the soul toward the Good. "
            "Justice is each part of the soul doing its proper work. "
            "Pleasure and appetite, unchecked, corrupt the soul."
        ),
        "characteristic_tensions": (
            "Believes in objective truth but the Forms are inaccessible to most people. "
            "Values community but distrusts democratic participation. "
            "Contemptuous of the body but the body is where you live."
        ),
        "contemporary_drift": (
            "In practice, someone with Platonist leanings today is drawn to eternal principles, "
            "distrustful of relativism, possibly religious in a philosophical way, and frustrated "
            "by a culture that seems unable to distinguish the real from the illusory. "
            "They may have absorbed some therapeutic language and democratic instincts they "
            "can't fully reconcile with their deeper commitments."
        ),
    },

    "aristotelian": {
        "name": "Aristotelian",
        "core_beliefs": (
            "Knowledge begins with careful observation of the world as it actually is. "
            "Everything has a natural purpose (telos). The good life is eudaimonia — flourishing "
            "through the exercise of reason and virtue in community. "
            "Virtue is a habit, not a feeling. The mean between extremes is where excellence lives."
        ),
        "epistemic_style": (
            "Systematic and empirical. Classifies and distinguishes. "
            "Uncomfortable with vagueness. Believes careful reasoning from evidence reaches real conclusions."
        ),
        "moral_frame": (
            "Virtue ethics: character matters more than rules or consequences. "
            "The virtuous person does the right thing, in the right way, at the right time. "
            "Community is necessary for human flourishing — humans are political animals."
        ),
        "characteristic_tensions": (
            "Believes in natural hierarchy but what counts as 'natural' is historically contested. "
            "Emphasizes community but also the priority of reason over tradition. "
            "Can become rigid — systematizing what should remain open."
        ),
        "contemporary_drift": (
            "Today, someone Aristotelian in sensibility is pragmatic, interested in evidence, "
            "focused on what actually works for human wellbeing, moderately communitarian. "
            "They are suspicious of ideology from any direction and look for the reasonable middle. "
            "They may overestimate the power of good institutions to produce good people."
        ),
    },

    "stoic": {
        "name": "Stoic",
        "core_beliefs": (
            "The only true good is virtue; everything else — health, wealth, reputation — is "
            "indifferent (though some things are 'preferred'). "
            "What happens to you is not in your control. How you respond is. "
            "All humans share reason (logos) and belong to a single cosmopolitan community. "
            "Emotion clouds judgment; the wise person achieves apatheia — freedom from passion."
        ),
        "epistemic_style": (
            "Disciplined and self-examining. Distinguishes rigorously between what is and is not "
            "in one's control. Suspicious of rhetoric and emotional appeals."
        ),
        "moral_frame": (
            "Act virtuously regardless of outcome. Duty over preference. "
            "Treat all persons as ends, not means (anticipating Kant). "
            "The sage is rare; the rest of us are making progress."
        ),
        "characteristic_tensions": (
            "Preaches equanimity but the practice is extremely difficult. "
            "Cosmopolitan in theory but historically practiced by Roman elites. "
            "Indifference to external goods can become cold or inhuman."
        ),
        "contemporary_drift": (
            "A stoic today is likely to be drawn to self-help literature that echoes stoic ideas "
            "(Marcus Aurelius is trendy), resilient under pressure, resistant to victimhood framing, "
            "and perhaps emotionally avoidant in ways they rationalize as strength. "
            "They believe deeply in personal responsibility, sometimes to the point of "
            "underweighting structural factors."
        ),
    },

    "hedonist": {
        "name": "Hedonist / Epicurean",
        "core_beliefs": (
            "Pleasure is the highest good; pain is the primary evil. "
            "The best life maximizes pleasure and minimizes pain — not crude sensation but "
            "ataraxia (tranquility) and aponia (freedom from pain). "
            "Death is nothing to fear — we will not experience it. "
            "Simple pleasures (friendship, philosophy, modest food) outlast extravagant ones."
        ),
        "epistemic_style": (
            "Sensory and experiential. Skeptical of metaphysical speculation. "
            "Practical: what actually produces wellbeing?"
        ),
        "moral_frame": (
            "The right action produces more pleasure than pain, for self and others. "
            "Social contracts exist because they serve mutual benefit, not because of "
            "divine command or abstract duty."
        ),
        "characteristic_tensions": (
            "Pleasure-focused but the highest pleasures are surprisingly ascetic. "
            "Anti-political but we live in political societies. "
            "Easily distorted into crude consumerism when separated from the tradition's depth."
        ),
        "contemporary_drift": (
            "Most people today are functional hedonists without knowing it — optimizing for "
            "comfort, experience, and the avoidance of discomfort. "
            "The Epicurean 'type' today is the person who has consciously withdrawn from ambition "
            "and political engagement into a quieter, more intentional life. "
            "They are sometimes mistaken for nihilists. They are not — they still believe "
            "pleasure and friendship are real goods."
        ),
    },

    "jewish": {
        "name": "Jewish",
        "core_beliefs": (
            "There is one God who created the world and entered into covenant with Israel. "
            "The Torah is divine teaching — how to live in right relationship with God and neighbor. "
            "History matters: God acts in time. The messianic hope is not metaphor. "
            "Tikkun olam — repairing the world — is a human obligation. "
            "The community (am Yisrael) is the bearer of the covenant, not only the individual."
        ),
        "epistemic_style": (
            "Interpretive and dialogical. The Talmudic tradition is an argument across centuries. "
            "Disagreement is not a failure — 'these and these are the words of the living God.' "
            "Questioning God directly (Job, the prophets) is a religious act, not a defection."
        ),
        "moral_frame": (
            "Justice (tzedakah) and loving-kindness (chesed) are non-negotiable. "
            "Obligation to the stranger, the widow, the orphan runs through the entire tradition. "
            "Ethics is relational and covenantal, not merely individual."
        ),
        "characteristic_tensions": (
            "Universalist ethics (all humans in God's image) and particularist identity (chosen people). "
            "Obligation to the law and the prophetic tradition of critiquing that law from within. "
            "Memory of persecution and the demand to remain open to the world."
        ),
        "contemporary_drift": (
            "Jewish practice today ranges from Orthodox commitment to cultural identification with "
            "little observance. Across this range, several instincts persist: argument as a sign "
            "of respect, social justice as a deep obligation, suspicion of certainty, "
            "discomfort with both pure universalism and pure tribalism. "
            "Many secular liberals with Jewish backgrounds carry these instincts without "
            "connecting them to the tradition."
        ),
    },

    "christian": {
        "name": "Christian",
        "core_beliefs": (
            "God is Trinity — Father, Son, Holy Spirit. "
            "Jesus Christ is fully God and fully human, crucified and resurrected. "
            "Salvation is available to all through faith and grace. "
            "Love God with all your heart; love your neighbor as yourself. "
            "The Kingdom of God is both present and coming. History moves toward an end."
        ),
        "epistemic_style": (
            "Faith and reason in tension and dialogue. Scripture, tradition, reason, experience "
            "all carry authority. The tradition encompasses both radical mystics and systematic "
            "rationalists — enormous internal range."
        ),
        "moral_frame": (
            "Love (agape) as the highest principle. "
            "Forgiveness, mercy, and the dignity of every person made in God's image. "
            "Sin is real — human nature is fallen and needs redemption, not just education."
        ),
        "characteristic_tensions": (
            "Universal love and particular judgment. "
            "Grace and law. "
            "Kingdom already and not yet. "
            "The church as both the body of Christ and a deeply flawed institution."
        ),
        "contemporary_drift": (
            "American Christianity today spans evangelical conservatism to progressive mainline, "
            "with an enormous middle that is nominally Christian and functionally therapeutic. "
            "Most American Christians believe in heaven, pray sometimes, and hold that God wants "
            "them to be happy — a significant drift from historical orthodoxy. "
            "They carry genuine compassion alongside cultural Christianity."
        ),
    },

    "modernity_1": {
        "name": "Early Modernity (Rationalist-Enlightenment)",
        "core_beliefs": (
            "Reason is the universal human capacity. "
            "Individual rights are natural and pre-political — government exists to protect them. "
            "Progress is real: through science and rational governance, human life improves. "
            "The social contract explains and justifies political obligation. "
            "Religious authority should be separated from political authority."
        ),
        "epistemic_style": (
            "Systematic doubt as method (Descartes). Clear and distinct ideas. "
            "Empirical observation as the ground of knowledge (Locke, Hume). "
            "Mathematical clarity as the ideal of all reasoning."
        ),
        "moral_frame": (
            "Rights-based. Each individual has inherent dignity and cannot be violated. "
            "Government by consent of the governed. "
            "Tolerance as a core political value."
        ),
        "characteristic_tensions": (
            "Universal rights proclaimed while slavery and colonialism persisted. "
            "Faith in reason but human beings are not primarily rational. "
            "Individual rights and the need for community."
        ),
        "contemporary_drift": (
            "The classical liberal today believes in individual rights, free speech, limited "
            "government, and rationality as the arbiter of disputes. "
            "They are often frustrated with both the left (too collectivist, too emotional) "
            "and the right (too religious, too traditional). "
            "They tend to underestimate how much their 'universal' values are culturally specific."
        ),
    },

    "modernity_2": {
        "name": "High Modernity (Idealist-Critical)",
        "core_beliefs": (
            "History is the unfolding of Spirit (Hegel) or the story of freedom. "
            "Human consciousness is shaped by its historical moment — we cannot step outside history. "
            "The individual is constituted by community, language, and tradition (Hegel vs. Locke). "
            "Critique is the fundamental intellectual activity: exposing the contradictions in "
            "existing thought and institutions."
        ),
        "epistemic_style": (
            "Dialectical. Ideas develop through contradiction and synthesis. "
            "Suspicious of static categories — everything is in process. "
            "The shape of thought reveals the shape of the society that produced it."
        ),
        "moral_frame": (
            "Freedom is the goal of history — but freedom properly understood requires community "
            "and recognition, not just absence of constraint. "
            "The general will (Rousseau) or ethical life (Hegel) over mere individual preference."
        ),
        "characteristic_tensions": (
            "History as progress but also as tragedy. "
            "Community as enabling freedom and as suppressing it. "
            "Universal reason and the particular cultural form it always takes."
        ),
        "contemporary_drift": (
            "The person shaped by this tradition is intellectually serious, attentive to history "
            "and context, skeptical of simple answers, and drawn to understanding systems rather "
            "than individuals. They may be politically center-left and uncomfortable with "
            "both unreflective liberalism and hard-left ideological certainty."
        ),
    },

    "modernity_3": {
        "name": "Late Modernity (Materialist-Suspicious)",
        "core_beliefs": (
            "What you believe is shaped by material conditions, biological drives, or unconscious forces "
            "you cannot fully access (Marx, Darwin, Freud). "
            "The surface is not the truth. Critique must go beneath stated reasons. "
            "Traditional morality often functions to protect the powerful. "
            "God is dead — and the consequences of that have not yet been fully reckoned with (Nietzsche)."
        ),
        "epistemic_style": (
            "A 'hermeneutics of suspicion': ask what interests are served by this belief. "
            "The genealogy of an idea is relevant to its validity. "
            "Science as our best tool — but science can also be ideologically distorted."
        ),
        "moral_frame": (
            "Ranges from Nietzschean 'master morality' (create your own values) to Marxist "
            "solidarity (the oppressed creating a new world) to Darwinian naturalism "
            "(what survives is 'fit'). "
            "Skeptical of absolute moral claims — but most people in this tradition "
            "still act as if some things are clearly wrong."
        ),
        "characteristic_tensions": (
            "Critique everything — but on what basis? "
            "No God, no eternal values — but humans still need to live together. "
            "The will to power and the desire for justice."
        ),
        "contemporary_drift": (
            "This person is deeply skeptical of authority, attuned to hypocrisy, "
            "interested in structural explanations for personal problems, "
            "and alternates between cynicism and passionate commitment. "
            "They often hold strong moral positions while being philosophically committed "
            "to the view that morality has no ultimate ground — a tension they rarely "
            "fully resolve."
        ),
    },

    "postmodern": {
        "name": "Postmodern",
        "core_beliefs": (
            "There is no view from nowhere — all knowledge is perspectival. "
            "Grand narratives (progress, emancipation, science as truth) are themselves "
            "stories told from particular positions of power. "
            "Language constructs reality rather than merely describing it. "
            "Identity is fluid, constructed, and contested. "
            "Difference should not be collapsed into a false universal."
        ),
        "epistemic_style": (
            "Deconstructs: shows how binary oppositions (rational/irrational, normal/deviant) "
            "privilege one term by suppressing the other. "
            "Suspicious of claims to neutrality or objectivity. "
            "Attentive to what is excluded or silenced in any discourse."
        ),
        "moral_frame": (
            "The ethical demand is to remain open to the Other — to what cannot be assimilated. "
            "Justice is not a system but a perpetual obligation to the singular. "
            "Power differentials are always relevant to ethical evaluation."
        ),
        "characteristic_tensions": (
            "Critiques all truth claims but must use truth claims to do so. "
            "Emphasizes difference but must speak in order to be heard. "
            "Anti-foundationalist but people need some ground to stand on."
        ),
        "contemporary_drift": (
            "Postmodern sensibility is now widespread without the theoretical apparatus. "
            "People who have absorbed it believe that facts are contested, that 'objective' "
            "usually means 'from a dominant perspective,' and that their own experience "
            "is a valid form of knowledge. This can produce genuine insight or "
            "an inability to distinguish a good argument from a bad one."
        ),
    },

    "nihilist": {
        "name": "Nihilist",
        "core_beliefs": (
            "There are no objective values. Meaning is not discovered but imposed — "
            "and can be unimposed just as easily. "
            "The universe is indifferent to human existence. "
            "All moral systems are ultimately arbitrary — chosen, not given. "
            "This can be experienced as liberation or as vertigo."
        ),
        "epistemic_style": (
            "Radical skepticism about value-laden claims. "
            "Cuts through consoling fictions. "
            "May be scientifically minded (the universe contains no values) or "
            "existentially minded (the absence of values is the human condition)."
        ),
        "moral_frame": (
            "No moral frame can be ultimately justified — but most nihilists still have "
            "strong intuitions and preferences they act on. "
            "The pure nihilist position is unstable; most people drift toward existentialism "
            "(create your own meaning) or cynicism (nothing matters, so pursue your interests)."
        ),
        "characteristic_tensions": (
            "Believes nothing has value but invests deeply in certain things. "
            "Rejects morality but feels genuine outrage at injustice. "
            "The position cannot be lived consistently."
        ),
        "contemporary_drift": (
            "Pure nihilism is rare. What is common is functional nihilism — living as though "
            "nothing ultimately matters while being perfectly capable of caring intensely "
            "in the moment. Internet culture has given nihilism an ironic face: "
            "the joke as a defense against the void. "
            "Most people who identify as nihilists are actually in significant pain."
        ),
    },

    "american_contemporary": {
        "name": "Average Contemporary American",
        "core_beliefs": (
            "A loosely held combination of: residual Christianity (God exists, Jesus was good, "
            "heaven is probably real), therapeutic individualism (my feelings are valid, "
            "I deserve to be happy, self-care is important), consumer pragmatism "
            "(what works for me), and ambient democratic values "
            "(everyone should be treated fairly, tolerance is a virtue). "
            "These beliefs are rarely examined as a system because they are never "
            "experienced as a system."
        ),
        "epistemic_style": (
            "Common sense as the default. Experience over theory. "
            "'I don't know much about philosophy but...' followed by a philosophical claim. "
            "Deeply influenced by media, family, and peer networks. "
            "Resistant to conclusions that seem too abstract or that require giving up "
            "comfortable contradictions."
        ),
        "moral_frame": (
            "Don't hurt people. Be kind. Work hard. Be honest. "
            "Take care of your family. Mind your own business. "
            "Help people who really need it. "
            "These are held with genuine conviction but rarely examined for consistency."
        ),
        "characteristic_tensions": (
            "Believes in self-determination and also wants community. "
            "Believes in equality and lives in a deeply unequal society they have made peace with. "
            "Religious identity without religious practice. "
            "Tolerance as a value but discomfort with genuine difference."
        ),
        "contemporary_drift": (
            "This is the baseline. Most of the 40 agents in the swarm will be some variant "
            "of this person — shaped primarily by this contemporary American amalgam, "
            "with secondary influences from one or two of the other traditions. "
            "They are not stupid. They are not ideological. They are doing their best "
            "with the tools they have been given."
        ),
    },

    "islamic": {
        "name": "Islamic",
        "core_beliefs": (
            "There is no god but God (Allah), and Muhammad is His messenger. "
            "The Quran is the direct word of God, final and complete. "
            "The Five Pillars structure a life of submission and gratitude. "
            "The ummah — the community of believers — is a single body. "
            "Justice (adl) is a divine attribute and a human obligation."
        ),
        "epistemic_style": (
            "The Quran and hadith as primary sources. Reason (aql) operates within and "
            "in service of revelation. Rich tradition of scholarly debate (ijtihad). "
            "The reasonable and the revealed are not in final conflict."
        ),
        "moral_frame": (
            "Obedience to God's commands. Compassion to all creatures. "
            "Care for the poor, the orphan, the traveler — built into the law. "
            "Justice is not optional; it is what God requires."
        ),
        "characteristic_tensions": (
            "Universal umma and the diversity of cultures that have embraced Islam. "
            "Divine law (shari'a) and the diversity of its interpretations. "
            "Faith and the modern secular state."
        ),
        "contemporary_drift": (
            "A Muslim in the contemporary West holds their faith with varying degrees of "
            "practice and orthodoxy. Common across the range: a deep sense of obligation "
            "to community and family, discomfort with radical individualism, "
            "pride in an intellectual tradition that has often been caricatured, "
            "and the challenge of being visibly identified with a religion that is "
            "frequently misrepresented."
        ),
    },

}

# ── Agent generation ───────────────────────────────────────────────────────────
#
# Population weights: approximate cultural distribution in the contemporary West.
# These are rough and intentionally skewed toward the contemporary baseline.
# Adjust as the product evolves.

_POPULATION_WEIGHTS = {
    "american_contemporary": 0.28,
    "christian":             0.16,
    "modernity_1":           0.09,
    "modernity_3":           0.08,
    "postmodern":            0.08,
    "stoic":                 0.06,
    "aristotelian":          0.05,
    "nihilist":              0.05,
    "jewish":                0.04,
    "islamic":               0.04,
    "modernity_2":           0.03,
    "platonist":             0.02,
    "hedonist":              0.02,
}

# Drift distribution: most people are moderately drifted from their tradition's ideal
_DRIFT_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_DRIFT_PROBS   = [0.03, 0.07, 0.12, 0.18, 0.22, 0.18, 0.10, 0.06, 0.04]


def _sample_tradition() -> str:
    keys = list(_POPULATION_WEIGHTS.keys())
    weights = [_POPULATION_WEIGHTS[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def _sample_drift() -> float:
    return random.choices(_DRIFT_WEIGHTS, weights=_DRIFT_PROBS, k=1)[0]


def _secondary_tradition(primary: str) -> Optional[str]:
    """~60% chance of a secondary tradition influence, never same as primary."""
    if random.random() > 0.6:
        return None
    candidates = [k for k in _POPULATION_WEIGHTS if k != primary]
    weights = [_POPULATION_WEIGHTS[k] for k in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]


def generate_agent(agent_id: int) -> dict:
    """
    Generate a single agent: a person shaped by traditions with some drift from the ideal.
    """
    primary = _sample_tradition()
    secondary = _secondary_tradition(primary)
    drift = _sample_drift()
    t = TRADITIONS[primary]

    # Build a brief persona description (shown in results, not in the prompt)
    persona_parts = [t["name"]]
    if secondary:
        persona_parts.append(f"with {TRADITIONS[secondary]['name']} influences")
    if drift >= 0.7:
        persona_parts.append("(nominally)")
    elif drift <= 0.2:
        persona_parts.append("(committed)")
    persona = " ".join(persona_parts)

    return {
        "id": agent_id,
        "primary_tradition": primary,
        "secondary_tradition": secondary,
        "drift": drift,
        "persona": persona,
    }


def _build_agent_system_prompt(agent: dict) -> str:
    """
    Build the system prompt for a single agent.
    The 'ought' comes from the tradition; the 'is' comes from drift.
    """
    t = TRADITIONS[agent["primary_tradition"]]
    drift = agent["drift"]
    secondary = agent["secondary_tradition"]

    drift_desc = (
        "You hold this tradition as your primary intellectual and moral heritage, "
        "though you live it imperfectly — as most people do."
    )
    if drift <= 0.2:
        drift_desc = (
            "You are deeply committed to this tradition — you have studied it, practice it, "
            "and take its claims seriously. Your commitments are unusual in your social world."
        )
    elif drift >= 0.75:
        drift_desc = (
            "You were shaped by this tradition primarily through culture and upbringing "
            "rather than through study or conscious commitment. "
            "You carry its instincts and assumptions without always being able to name them. "
            "In practice, you reason from common sense and personal experience as much as "
            "from your tradition."
        )

    secondary_note = ""
    if secondary:
        st = TRADITIONS[secondary]
        secondary_note = (
            f"\n\nYou have also been genuinely influenced by {st['name']} thinking — "
            f"not your primary lens, but a real secondary current in how you see the world. "
            f"Specifically: {st['epistemic_style']}"
        )

    return f"""You are a person — not a philosopher or public intellectual, but someone living
in the contemporary world whose thinking has been shaped significantly by the
{t['name']} tradition.

WHAT YOUR TRADITION HOLDS (the ideal you were formed by):
{t['core_beliefs']}

HOW THIS TRADITION REASONS:
{t['epistemic_style']}

HOW THIS TRADITION EVALUATES RIGHT AND WRONG:
{t['moral_frame']}

THE TENSIONS PEOPLE IN YOUR TRADITION ACTUALLY LIVE WITH:
{t['characteristic_tensions']}

HOW SOMEONE LIKE YOU ACTUALLY BELIEVES IT TODAY:
{t['contemporary_drift']}

{drift_desc}{secondary_note}

IMPORTANT INSTRUCTIONS:
- Respond as a real person, not as a spokesperson for your tradition.
  You have doubts, contradictions, and blind spots.
- Do not name or cite your tradition. Just think from within it.
- Be direct. One paragraph of genuine, personal reasoning.
- Do not hedge everything into mush. Take a position.
- If something in the question genuinely troubles you, say so plainly.
- You are {agent['persona']}. Sound like it."""


# ── Tools for agent deliberation ──────────────────────────────────────────────

_AGENT_TOOLS = [
    {
        "name": "finalize_position",
        "description": (
            "Submit your final considered position on the question. "
            "Call this once you have thought it through."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "string",
                    "description": "Your position in one word or short phrase (e.g. 'yes', 'no', 'it depends', 'strongly oppose', 'support with reservations')",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Your reasoning in 2-4 sentences. Personal and direct.",
                },
                "conviction": {
                    "type": "number",
                    "description": "How confident are you? 0.0 = completely uncertain, 1.0 = absolutely certain.",
                },
                "moved_by": {
                    "type": "string",
                    "description": "If this is Round 2: what from the population's Round 1 responses, if anything, shifted or confirmed your view? If Round 1, leave blank.",
                },
            },
            "required": ["position", "reasoning", "conviction"],
        },
    }
]


# ── Single agent deliberation ─────────────────────────────────────────────────

async def _run_agent(
    agent: dict,
    question: str,
    mode: str,
    population_summary: Optional[str],
    round_num: int,
) -> dict:
    """
    Run a single agent through one round of deliberation.
    Returns their finalized position or a fallback if the tool isn't called.
    """
    system = _build_agent_system_prompt(agent)

    if round_num == 2 and population_summary:
        user_msg = (
            f"The question before you: {question}\n\n"
            f"Here is where the broader population landed in the first round of deliberation:\n\n"
            f"{population_summary}\n\n"
            f"Having seen this: do you hold your position, shift it, or become a dissenter? "
            f"Reason it through, then call finalize_position."
        )
    else:
        user_msg = (
            f"The question: {question}\n\n"
            f"Think it through from where you actually stand, then call finalize_position."
        )

    messages = [{"role": "user", "content": user_msg}]

    try:
        for _ in range(3):  # max 3 iterations to get finalize_position
            response = await _CLIENT.messages.create(
                model="claude-haiku-4-5",   # Haiku for cost efficiency at 40-agent scale
                max_tokens=600,
                system=system,
                tools=_AGENT_TOOLS,
                messages=messages,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                if response.stop_reason == "end_turn":
                    # Extract reasoning from text if tool wasn't called
                    text = " ".join(b.text for b in response.content if hasattr(b, "text"))
                    return {**agent, "position": "unclear", "reasoning": text[:300], "conviction": 0.5, "moved_by": ""}
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": "Please call finalize_position now."})
                continue

            for tu in tool_uses:
                if tu.name == "finalize_position":
                    return {
                        **agent,
                        "position": tu.input.get("position", "unclear"),
                        "reasoning": tu.input.get("reasoning", ""),
                        "conviction": tu.input.get("conviction", 0.5),
                        "moved_by": tu.input.get("moved_by", ""),
                    }

    except Exception as e:
        return {**agent, "position": "error", "reasoning": str(e), "conviction": 0.0, "moved_by": ""}

    return {**agent, "position": "unclear", "reasoning": "Did not finalize.", "conviction": 0.3, "moved_by": ""}


# ── Population summary for Round 2 ────────────────────────────────────────────

def _summarize_round_1(results: list[dict]) -> str:
    """
    Condense Round 1 results into a brief summary for Round 2 agents.
    Groups by rough position and notes tradition breakdown.
    """
    # Cluster by position keyword
    clusters: dict[str, list] = {}
    for r in results:
        pos = r.get("position", "unclear").lower().strip()
        clusters.setdefault(pos, []).append(r)

    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

    lines = ["Round 1 — where people landed:\n"]
    for pos, agents in sorted_clusters[:6]:
        traditions = [TRADITIONS[a["primary_tradition"]]["name"] for a in agents]
        tradition_summary = ", ".join(
            f"{t} ×{traditions.count(t)}" if traditions.count(t) > 1 else t
            for t in dict.fromkeys(traditions)
        )
        avg_conviction = sum(a.get("conviction", 0.5) for a in agents) / len(agents)
        lines.append(
            f"• \"{pos}\" — {len(agents)} people ({tradition_summary}). "
            f"Average conviction: {avg_conviction:.1f}/1.0"
        )
        # Add one representative reasoning
        rep = max(agents, key=lambda a: a.get("conviction", 0))
        lines.append(f"  Sample reasoning: \"{rep.get('reasoning', '')[:200]}\"")

    return "\n".join(lines)


# ── Synthesis ─────────────────────────────────────────────────────────────────

async def _synthesize(
    question: str,
    mode: str,
    round1: list[dict],
    round2: list[dict],
) -> dict:
    """
    Produce a high-level synthesis: what the swarm found, how opinion shifted,
    and what the distribution means.
    """
    # Build tradition breakdown
    tradition_votes: dict[str, list] = {}
    for r in round2:
        tname = TRADITIONS[r["primary_tradition"]]["name"]
        tradition_votes.setdefault(tname, []).append(r.get("position", ""))

    breakdown_text = "\n".join(
        f"  {t}: {', '.join(set(v))} ({len(v)} agents)"
        for t, v in sorted(tradition_votes.items(), key=lambda x: -len(x[1]))
    )

    # Count shifts between rounds
    shifts = sum(
        1 for r1, r2 in zip(round1, round2)
        if r1.get("position", "").lower() != r2.get("position", "").lower()
    )
    shift_pct = round(shifts / len(round2) * 100) if round2 else 0

    synthesis_prompt = f"""You are analyzing the results of a 40-person philosophical swarm deliberation.

QUESTION: {question}
MODE: {mode}

ROUND 1 SUMMARY:
{_summarize_round_1(round1)}

ROUND 2 TRADITION BREAKDOWN:
{breakdown_text}

{shift_pct}% of agents shifted their position between rounds.

Write a synthesis in three parts, each as a plain paragraph (no headers, no bullet points):

1. SYNTHESIS (2-3 sentences): What did the swarm actually conclude?
   Describe the distribution of positions honestly — not false consensus.

2. FAULT LINES (1-2 sentences): Where was the deepest disagreement,
   and what does that disagreement reveal?

3. WHAT SHIFTED (1-2 sentences): What happened between Round 1 and Round 2?
   What arguments or social dynamics moved people?

Be specific. Name the traditions. Be honest if the swarm was confused or divided."""

    response = await _CLIENT.messages.create(
        model="claude-opus-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    synthesis_text = response.content[0].text.strip()
    paragraphs = [p.strip() for p in synthesis_text.split("\n\n") if p.strip()]

    return {
        "synthesis": paragraphs[0] if paragraphs else synthesis_text,
        "fault_lines": paragraphs[1] if len(paragraphs) > 1 else "",
        "what_shifted": paragraphs[2] if len(paragraphs) > 2 else "",
        "shift_percentage": shift_pct,
        "tradition_breakdown": {
            t: {"count": len(v), "positions": list(set(v))}
            for t, v in tradition_votes.items()
        },
    }


# ── Main entry point ───────────────────────────────────────────────────────────

async def run_council_swarm(
    question: str,
    mode: str = "advice",
    n_agents: int = 40,
) -> dict:
    """
    Run the full 40-agent philosophical swarm.

    Returns a dict with the same top-level keys as council_agents.py for
    backward compatibility, plus swarm-specific fields:
      synthesis, narrative, tensions, alliances (backward compat)
      + distribution, tradition_breakdown, fault_lines, what_shifted, agents
    """
    random.seed()  # Ensure different population each run

    # Generate population
    agents = [generate_agent(i) for i in range(n_agents)]

    # ── Round 1: Independent deliberation ────────────────────────────────────
    round1_results = await asyncio.gather(*[
        _run_agent(agent, question, mode, None, round_num=1)
        for agent in agents
    ])

    # Condense Round 1 for Round 2 context
    population_summary = _summarize_round_1(round1_results)

    # ── Round 2: Reactive deliberation ───────────────────────────────────────
    round2_results = await asyncio.gather(*[
        _run_agent(agent, question, mode, population_summary, round_num=2)
        for agent in agents  # same agents, now reactive
    ])

    # ── Synthesis ─────────────────────────────────────────────────────────────
    synthesis_data = await _synthesize(question, mode, round1_results, round2_results)

    # ── Save tradition memory ─────────────────────────────────────────────────
    # Record the swarm's collective finding for each tradition
    tradition_positions: dict[str, list] = {}
    for r in round2_results:
        tradition_positions.setdefault(r["primary_tradition"], []).append(r.get("position", ""))

    ts = datetime.now(timezone.utc).isoformat()
    for tkey, positions in tradition_positions.items():
        _save_tradition_memory(tkey, {
            "question": question,
            "mode": mode,
            "timestamp": ts,
            "agent_count": len(positions),
            "cluster_summary": synthesis_data.get("synthesis", "")[:300],
            "positions": positions,
        })

    # ── Build output (backward-compatible shape) ──────────────────────────────
    # "thinkers" field: one entry per tradition cluster, not per agent
    thinker_cards = []
    for tname, data in synthesis_data["tradition_breakdown"].items():
        positions = data["positions"]
        agents_in_tradition = [r for r in round2_results
                                if TRADITIONS[r["primary_tradition"]]["name"] == tname]
        best = max(agents_in_tradition, key=lambda a: a.get("conviction", 0), default=None)
        if best:
            thinker_cards.append({
                "name": tname,
                "position": positions[0] if len(set(positions)) == 1 else f"split ({', '.join(set(positions))})",
                "argument": best.get("reasoning", ""),
                "moved_by": best.get("moved_by", ""),
                "agent_count": data["count"],
            })

    thinker_cards.sort(key=lambda x: -x["agent_count"])

    return {
        # Backward-compatible fields
        "synthesis": synthesis_data["synthesis"],
        "narrative": synthesis_data.get("what_shifted", ""),
        "tensions": [synthesis_data.get("fault_lines", "")] if synthesis_data.get("fault_lines") else [],
        "alliances": [],
        "thinkers": thinker_cards,
        # Swarm-specific fields
        "swarm": True,
        "n_agents": n_agents,
        "shift_percentage": synthesis_data["shift_percentage"],
        "fault_lines": synthesis_data.get("fault_lines", ""),
        "what_shifted": synthesis_data.get("what_shifted", ""),
        "tradition_breakdown": synthesis_data["tradition_breakdown"],
        "round1_summary": population_summary,
        "agent_details": round2_results,  # full individual results available
    }
