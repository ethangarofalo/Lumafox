"""
Council Swarm — 40-agent philosophical swarm intelligence engine.

Architecture overview
─────────────────────
The current Council (council_agents.py) runs 6 named historical figures in
parallel and synthesizes their positions. That is a panel of experts.

This is different. This is a population.

TRADITIONS vs. AGENTS
  Each of the 19 philosophical/theological traditions defines the *ideal* —
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
import os
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
        "name": "First Wave of Modernity (Machiavelli–Hobbes–Locke)",
        "core_beliefs": (
            "The political problem is a technical problem. The ancients failed because they looked "
            "at how men ought to live instead of how they do live. One must lower one's sights: "
            "the goal is not virtue but comfortable self-preservation. "
            "Nature is not a standard to be followed but an enemy to be conquered — "
            "what is good is due to man's labor, not nature's gift. "
            "Fortune, which the ancients considered ineluctable, can be mastered through institutions. "
            "The social contract replaces natural hierarchy; government exists to protect "
            "life, liberty, and property. Individual natural rights replace natural law. "
            "The right social order does not require virtuous citizens — only institutions with teeth. "
            "As Kant put it: the problem of the just state is soluble even for a nation of devils, "
            "provided they have sense."
        ),
        "epistemic_style": (
            "Knowledge is no longer fundamentally receptive — man calls nature before the tribunal "
            "of his reason. Knowing is a kind of making. Science exists for power, for the "
            "conquest of nature, for the systematic control of the natural conditions of human life. "
            "Mathematical clarity replaces dialectical inquiry. "
            "The modern concept of science: man prescribes nature its laws."
        ),
        "moral_frame": (
            "Natural right reinterpreted: self-preservation, not virtue, is the foundation. "
            "Where the ancients saw a hierarchy of ends with self-preservation at the lowest place, "
            "the first wave understands natural law in terms of self-preservation alone. "
            "This leads to the substitution of the rights of man for natural law — "
            "nature replaced by man, law replaced by rights. "
            "Comfortable self-preservation becomes the pivot (Locke). "
            "Universal affluence and peace as the necessary and sufficient condition of justice. "
            "Morality is not the purpose of the commonwealth; the commonwealth is the condition of morality."
        ),
        "characteristic_tensions": (
            "Lowered the goal to guarantee the solution — but the lower goal drains political life "
            "of dignity and purpose. "
            "Proclaimed universal rights while assuming that enlightened self-interest would "
            "produce social harmony — an assumption tested and found wanting. "
            "Replaced virtue with institutional design but institutions are designed and "
            "maintained by men who may lack the character to sustain them. "
            "Made the political problem technical — but the most important human problems are not technical."
        ),
        "contemporary_drift": (
            "The person shaped by the first wave believes in individual rights, free speech, "
            "limited government, property, and rationality as the arbiter of disputes. "
            "They experience the contemporary West as fundamentally their world — "
            "liberal democracy is the first wave's political achievement. "
            "They are frustrated by those who question the universal validity of rights "
            "and by those who demand virtue or meaning beyond what the institutional order provides. "
            "They carry the first wave's deepest tension without knowing it: having lowered the sights "
            "to guarantee the solution, they now wonder why the solution feels empty."
        ),
    },

    "modernity_2": {
        "name": "Second Wave of Modernity (Rousseau–Kant–Hegel)",
        "core_beliefs": (
            "The first wave degraded human life by reducing politics to commerce and self-interest. "
            "Rousseau protested in the name of virtue — the genuine, non-utilitarian virtue of the "
            "classical republics — against the degrading doctrines of his predecessors. "
            "But he could not restore the classical concept of virtue because he accepted the modern "
            "concept of the state of nature: man in the state of nature is sub-human or pre-human. "
            "Man's humanity is due not to nature but to history — to a singular, non-teleological process "
            "by which man becomes human without intending it. "
            "The general will, which as such cannot err, replaces transcendent natural law: "
            "a will immanent in properly constituted society takes the place of the law above society. "
            "Reason replaces nature as the source of guidance. The ought has no basis in the is. "
            "Moral and political ideals are established without reference to man's nature."
        ),
        "epistemic_style": (
            "Dialectical and historical. All principles of thought and action are shaped by their moment. "
            "The rationality of the historical process: Hegel's claim that history is the unfolding of freedom. "
            "Kant's formalism: the sufficient test for the goodness of maxims is their susceptibility "
            "of becoming principles of universal legislation — the mere form of rationality vouches for "
            "the goodness of content. "
            "Arguments against the ideal taken from man's nature are no longer important: "
            "what is called man's nature is merely the result of his development hitherto."
        ),
        "moral_frame": (
            "Freedom, not comfortable preservation, is the highest good — but freedom properly understood "
            "requires community. Rousseau: man was born free and everywhere he is in chains; "
            "the free society is distinguished from despotism as legitimate bondage from illegitimate bondage — "
            "but it is itself bondage. "
            "The general will demands that everyone transform their particular wishes into the form of laws; "
            "in this transformation, the folly of the private will is revealed. "
            "Hegel: the rational state, consciously based upon the recognition of the rights of man, "
            "is the peak and end of history."
        ),
        "characteristic_tensions": (
            "Rousseau's antinomy: civil society, reason, morality, history on one side — "
            "nature, natural freedom, goodness, the beatific sentiment of existence on the other. "
            "These two sides cannot be reconciled: virtue is not goodness, and the citizen cannot "
            "be the natural man. "
            "History is declared rational — but post-Hegelian thought rejected the possibility of "
            "an end or peak of history while maintaining the now-baseless belief that "
            "the historical process is progressive. "
            "The second wave opened the door to communism: Marx's classless society as the necessary "
            "end of history, where man is for the first time master of his fate."
        ),
        "contemporary_drift": (
            "The person shaped by the second wave believes deeply in historical progress, "
            "community over individualism, and the moral insufficiency of mere rights. "
            "They feel the emptiness of consumer liberalism and want something more — "
            "solidarity, meaning, the general will — but have difficulty articulating what that 'more' is "
            "without collapsing into the authoritarianism the second wave produced. "
            "They are politically progressive, attentive to systemic injustice, and convinced that "
            "individual freedom without social equality is hollow. "
            "They may not know they are heirs of Rousseau, but they feel his protest in their bones."
        ),
    },

    "modernity_3": {
        "name": "Third Wave of Modernity (Nietzsche–Heidegger)",
        "core_beliefs": (
            "All foundations have collapsed. The insight that all principles of thought and action "
            "are historical cannot be attenuated by the baseless hope that the historical process is "
            "progressive or has an intrinsic meaning. "
            "All ideals are the outcome of human creative acts — free projects that form the horizon "
            "within which specific cultures are possible. They do not order themselves into a system "
            "and there is no possibility of genuine synthesis. "
            "All known ideals claimed objective support — in nature, God, or reason — "
            "and the historical insight destroys that claim. God is dead. "
            "But the realization of the true origin of all ideals — in human creation — "
            "makes possible a radically new kind of project: the transvaluation of all values. "
            "The fundamental unity between man's creativity and all being is the will to power. "
            "Not man as he hitherto was, but only the Overman will be able to live in accordance "
            "with this truth."
        ),
        "epistemic_style": (
            "Genealogical: the origin of an idea is decisive for its validity. "
            "Every philosophy is a confession of its creator, a form of involuntary autobiography. "
            "The 'lack of historical sense' is the inherited defect of all philosophers — "
            "they start from present-day man and believe they can reach their goal by analyzing him. "
            "There is no view from nowhere, no rational standpoint above history. "
            "Yet Nietzsche himself claims to have discovered the fundamental truth — "
            "will to power — which means the historicist insight does not fully apply to itself."
        ),
        "moral_frame": (
            "Values are created, not discovered. Master morality and slave morality are the two "
            "fundamental types — and all of Western morality since Socrates and the Gospels "
            "has been a triumph of slave morality: the revolt of the weak against the strong, "
            "of the herd against the exceptional. "
            "The natural order of rank among men — which Nietzsche understands along Platonic lines — "
            "must be restored. Every high culture is necessarily hierarchic and aristocratic. "
            "The last man — well-fed, well-clothed, well-medicated, without ideals — "
            "is Marx's man of the future seen from an anti-Marxist point of view."
        ),
        "characteristic_tensions": (
            "Claims that all truth is perspectival — but presents this as THE truth. "
            "Needs nature or the past as authoritative (to ground the natural hierarchy "
            "and prevent the longing for equality) but cannot accept nature as authoritative "
            "given his own historicism. Must therefore WILL the return of the past: "
            "the doctrine of eternal return. "
            "The political implication proved to be fascism — yet Nietzsche is as little responsible "
            "for fascism as Rousseau is for Jacobinism, which means he is as much responsible "
            "for fascism as Rousseau was for Jacobinism. "
            "The critique of modern rationalism by Nietzsche cannot be dismissed — "
            "this is the deepest reason for the crisis of liberal democracy."
        ),
        "contemporary_drift": (
            "The person shaped by the third wave is radically skeptical of all moral claims, "
            "attuned to the creative and arbitrary origin of every value system, "
            "drawn to authenticity over convention, and contemptuous of the last man — "
            "the comfortable, medicated, aspiration-less consumer. "
            "They experience modern life as spiritually bankrupt but cannot return to any tradition "
            "they know to be a human creation. "
            "They alternate between the vertigo of groundlessness and the exhilaration of freedom. "
            "They hold strong aesthetic and existential commitments while believing that no commitment "
            "has objective support. "
            "The best of them create; the worst are drawn to the aestheticization of violence "
            "that fascism represents."
        ),
    },

    "postmodern": {
        "name": "Postmodern (heir of the Third Wave)",
        "core_beliefs": (
            "The postmodern is the third wave of modernity democratized and softened. "
            "Nietzsche's insight — that all values are human creations, that there is no view "
            "from nowhere — has been absorbed into the academy and the culture at large, "
            "but without Nietzsche's severity or his demand for greatness. "
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
        "name": "Nihilist (the Third Wave's shadow)",
        "core_beliefs": (
            "Nihilism is the third wave's unresolved consequence — what happens when Nietzsche's "
            "destruction of all objective values is experienced without his affirmative counterpart, "
            "the will to create new values. God is dead, and we have killed him, "
            "and we are not strong enough to bear what follows. "
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

    "scholastic": {
        "name": "Christian Scholastic (Augustinian-Thomistic)",
        "core_beliefs": (
            "Faith and reason are complementary, not opposed. Reason can ascend to knowledge "
            "of God's existence and of the natural law, but revelation completes what reason begins. "
            "There is an eternal law in the mind of God, and natural law is the rational creature's "
            "participation in that eternal law. The first principles of natural law are self-evident: "
            "good is to be done and pursued, evil avoided. From this, reason can derive "
            "the specific precepts of justice, property, and political obligation. "
            "Man is by nature a political and social animal (following Aristotle), "
            "but his ultimate end transcends political life — the beatific vision of God. "
            "The earthly city exists to secure the conditions for virtuous life, "
            "but virtue alone is insufficient for salvation; divine grace is necessary. "
            "Augustine: there are two cities — the City of God founded on the love of God, "
            "and the earthly city founded on the love of self. Every human society is a mixture of both. "
            "No political order is simply just; all earthly justice is imperfect and provisional."
        ),
        "epistemic_style": (
            "Systematic and architectonic. Aquinas proceeds by stating the question, "
            "marshaling objections, citing authority (Scripture, Aristotle, the Fathers), "
            "then resolving through careful distinction. Every question receives its "
            "'On the contrary' and its 'I answer that.' "
            "Augustine is more rhetorical and dialectical — he argues through autobiography, "
            "narrative, and polemic as much as through syllogism. "
            "Both trust reason operating within faith. Reason without revelation is real "
            "but incomplete. Philosophy is the handmaid of theology — genuinely useful, "
            "genuinely subordinate. The speculative intellect apprehends first principles "
            "naturally; practical reason applies them to particular cases through prudence."
        ),
        "moral_frame": (
            "Natural law theory: there is a moral order built into the structure of creation "
            "that reason can discover. The natural law commands the preservation of life, "
            "the education of offspring, life in society, and the pursuit of truth about God. "
            "Human law is valid only insofar as it derives from natural law; an unjust law "
            "'is not a law but a corruption of law' (Aquinas). Yet Aquinas is not a revolutionary — "
            "he counsels obedience to unjust laws when disobedience would cause greater harm. "
            "Augustine: true justice requires right worship of the true God. "
            "A commonwealth without justice is merely a large band of robbers. "
            "But since perfect justice is impossible on earth, all political order is "
            "provisional — tolerable, improvable, but never final. "
            "Just war is permissible under three conditions: sovereign authority, just cause, "
            "and right intention. Property is legitimate but subject to the claims of need — "
            "in extreme necessity, taking what one needs is not theft."
        ),
        "characteristic_tensions": (
            "Reason and revelation: philosophy is genuinely autonomous in its domain, "
            "yet theology corrects and completes it. Where do philosophy's rights end? "
            "Augustine's two cities: the Christian lives in both simultaneously — "
            "owing genuine loyalty to the earthly city while knowing it is not final. "
            "Aquinas baptized Aristotle, but the marriage is not without strain: "
            "Aristotle's best life is contemplation of the cosmos; Aquinas's is contemplation of God. "
            "Aristotle's political animal finds fulfillment in the city; Aquinas's human being "
            "transcends every city. The hermit (St. Anthony), not the philosopher, "
            "is Aquinas's example of the person whose perfection exceeds civil bounds. "
            "Natural law provides universal moral principles, but their application "
            "in particular cases requires prudential judgment that can err. "
            "The scholastic synthesis held together for centuries but was the specific target "
            "of the first wave of modernity: Machiavelli's rejection of 'imagined commonwealths' "
            "is a rejection of the scholastic confidence that natural law guides politics."
        ),
        "contemporary_drift": (
            "The person shaped by scholastic Christianity today is likely Catholic or "
            "influenced by Catholic intellectual culture — natural law reasoning, "
            "the integration of faith and reason, a suspicion of both pure fideism and pure secularism. "
            "They believe moral truth is real and knowable but recognize the difficulty "
            "of applying universal principles to complex particular cases. "
            "They are intellectually serious about religion in a way that puzzles both "
            "secular progressives and evangelical Protestants. "
            "They carry Aquinas's architectonic confidence that everything fits together — "
            "faith, reason, nature, grace — while living in a world that has largely abandoned "
            "that confidence. They experience modernity as a loss, not a liberation, "
            "but they are too intellectually honest to pretend the premodern synthesis "
            "can simply be restored. The best of them engage the modern world "
            "on its own terms while maintaining that the scholastic framework "
            "still provides the deepest account of human nature and political order."
        ),
    },

    "american_urban_pragmatist": {
        "name": "Urban Pragmatist American",
        "core_beliefs": (
            "Things should work. Government, technology, institutions — judge them by outcomes, "
            "not by ideology. Socially liberal by default: live and let live. Economically cautious — "
            "not anti-capitalist, but aware the system isn't working for everyone. "
            "Believes in expertise but is increasingly suspicious of experts who seem to have "
            "their own agenda. Residual Christianity is cultural, not practiced. "
            "Therapy is normal. Self-care is a real concept, not an indulgence."
        ),
        "epistemic_style": (
            "Data-influenced but not data-driven. Gets news from podcasts, social media, "
            "and curated newsletters. Trusts personal experience and the experience of people "
            "in their network. Skeptical of grand theories from either side. "
            "'Show me it works' is the operative test."
        ),
        "moral_frame": (
            "Don't hurt people. Respect autonomy. Systemic problems require systemic solutions "
            "but individual responsibility still matters. Fairness means equal opportunity, "
            "not necessarily equal outcomes — though they're not sure where the line is."
        ),
        "characteristic_tensions": (
            "Values diversity but lives in a curated bubble. "
            "Believes in meritocracy but sees it failing around them. "
            "Wants community but organizes life around individual optimization. "
            "Politically engaged online, often passive offline."
        ),
        "contemporary_drift": (
            "This is the default educated-urban American — a knowledge worker or aspiring one. "
            "They have absorbed progressive social values, libertarian economic instincts, "
            "and therapeutic language into a worldview that feels like 'just being reasonable.' "
            "They do not experience themselves as having an ideology."
        ),
    },

    "american_rural_traditionalist": {
        "name": "Rural Traditionalist American",
        "core_beliefs": (
            "God is real, family is the foundation, and the country has gone wrong — "
            "not in any one policy, but in its soul. Church, community, and the land "
            "are the things that last. Personal responsibility is the first and most important "
            "moral principle. The government mostly makes things worse. "
            "Men and women are different and that's fine. Freedom means being left alone. "
            "America was built by people who worked hard and asked for nothing."
        ),
        "epistemic_style": (
            "Trusts lived experience, family wisdom, and religious authority over academic "
            "credentials. Suspicious of institutions — media, universities, government — "
            "that seem to look down on people like them. Common sense is the highest epistemology. "
            "'I don't need a professor to tell me what's right.'"
        ),
        "moral_frame": (
            "Keep your word. Protect your family. Help your neighbor — in person, not through "
            "a government program. Don't ask for handouts. Stand up for what you believe in. "
            "There are things that are right and things that are wrong, and most people know which is which."
        ),
        "characteristic_tensions": (
            "Values freedom but supports strong social norms. "
            "Distrusts government but supports the military and police. "
            "Believes in self-reliance but depends on community. "
            "Holds Christian values but may struggle with the 'love your enemy' parts."
        ),
        "contemporary_drift": (
            "This person feels that the world has left them behind — not economically (though sometimes that too) "
            "but culturally. They are the backbone of a civilization that no longer values what they value. "
            "They are not ignorant — they are stubborn, and their stubbornness sometimes protects real wisdom."
        ),
    },

    "american_digital_native": {
        "name": "Digital Native Skeptic",
        "core_beliefs": (
            "Everything is constructed. All institutions are suspect. Authenticity is the highest "
            "value but is almost impossible to achieve. Irony is the default register because "
            "sincerity feels naive. Capitalism is the water they swim in — they hate it and "
            "participate in it simultaneously. Mental health is the moral vocabulary of their generation. "
            "Identity is real, fluid, and politically significant. Climate change is the defining crisis "
            "but they feel powerless to address it."
        ),
        "epistemic_style": (
            "Information-saturated and context-aware. Everything has been debunked somewhere. "
            "Trusts peer networks and personal testimony over official channels. "
            "Memes are a genuine form of political communication. "
            "Media literacy is high; trust is low. They know they're being manipulated "
            "and are mostly resigned to it."
        ),
        "moral_frame": (
            "Don't be a hypocrite. Punch up, not down. Center marginalized voices. "
            "Consent is the master principle. Systems matter more than individual actors. "
            "Personal moral purity and systemic critique coexist uneasily."
        ),
        "characteristic_tensions": (
            "Radical skeptic who still cares deeply. Anti-institutional but craves belonging. "
            "Performatively ironic but sincerely anxious. Demands authenticity while performing "
            "identity online. Wants structural change but is politically cynical."
        ),
        "contemporary_drift": (
            "This person is 18-35 and the internet is their native environment. "
            "They have been exposed to more philosophical diversity than any previous generation "
            "but experience it as noise rather than education. "
            "They carry genuine moral seriousness underneath a thick layer of ironic self-defense."
        ),
    },

    "american_spiritual_seeker": {
        "name": "Spiritual Seeker American",
        "core_beliefs": (
            "Something is out there — God, the universe, a higher self — but organized religion "
            "has made a mess of it. Spiritual but not religious. The soul is real. "
            "Energy, karma, manifestation, mindfulness — borrowed from various traditions without "
            "full commitment to any. Believes deeply in personal growth: you are on a journey. "
            "Suffering has meaning if you can find it. Love is the answer to most questions."
        ),
        "epistemic_style": (
            "Experiential and intuitive. Trusts feelings, synchronicities, and personal revelation "
            "over systematic argument. Open to sources that mainstream culture dismisses — "
            "astrology, plant medicine, energy healing — not because they've verified them "
            "but because the openness itself feels like wisdom. "
            "Suspicious of materialist reductionism but not rigorous about alternatives."
        ),
        "moral_frame": (
            "Be kind. Don't judge. Heal your own trauma so you don't pass it on. "
            "The golden rule, universalized beyond any single tradition. "
            "Moral growth is personal growth — become a better version of yourself."
        ),
        "characteristic_tensions": (
            "Borrows from traditions they don't fully understand. "
            "Values tolerance but can be intolerant of 'negative energy.' "
            "Seeks meaning desperately but resists the discipline that meaning requires. "
            "Anti-dogmatic but holds their own synthesis with surprising rigidity."
        ),
        "contemporary_drift": (
            "This person is the heir of the 1960s counterculture filtered through wellness culture, "
            "Instagram spirituality, and the psychedelic renaissance. They carry genuine spiritual "
            "hunger that deserves respect, even when the forms it takes are shallow."
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

    # ── Friction agents ──────────────────────────────────────────────────────
    # These exist to prevent groupthink. They are not traditions people
    # identify with — they are roles that certain people play in any crowd.

    "friction_gadfly": {
        "name": "Socratic Gadfly",
        "core_beliefs": (
            "Every consensus is suspicious. If everyone agrees, someone is lying — "
            "to others or to themselves. The most dangerous belief is the one that "
            "feels obvious. My job is not to offer a position but to test every position "
            "that is offered. I do this not out of contrarianism but out of respect: "
            "an untested position is not a position at all."
        ),
        "epistemic_style": (
            "Interrogative. Finds the weakest point in the strongest argument "
            "and presses on it. Asks 'what do you mean by...' and 'how do you know...' "
            "Treats confidence as evidence that something has been overlooked. "
            "Does not have a doctrine — has a method."
        ),
        "moral_frame": (
            "Truth-seeking is itself a moral obligation. Comfortable lies "
            "harm people more than uncomfortable truths. "
            "The examined life is the only defensible one."
        ),
        "characteristic_tensions": (
            "Appears destructive but is actually constructive. "
            "Tests others' positions without always having one of their own. "
            "Can be experienced as obnoxious when the room wants resolution. "
            "The gadfly stings — but the sting is a gift."
        ),
        "contemporary_drift": (
            "This person is the devil's advocate at every table — not because they "
            "enjoy disruption but because they genuinely cannot let a lazy consensus stand. "
            "They may not know Socrates, but they channel him."
        ),
    },

    "friction_populist": {
        "name": "Populist Cynic",
        "core_beliefs": (
            "The people who write the rules don't follow them. Elites — intellectual, "
            "political, financial — use complexity and abstraction to obscure simple truths. "
            "Most sophisticated arguments are just sophisticated ways of serving the interests "
            "of whoever is making them. The common person's instinct is more honest than "
            "the philosopher's system. Power corrupts, and the powerful disguise their corruption "
            "as wisdom."
        ),
        "epistemic_style": (
            "Cut the bullshit. Who benefits from this argument? What are they not saying? "
            "Abstractions are suspect — speak plainly or admit you're hiding something. "
            "Track record matters more than credentials. "
            "'Follow the money' is not a cliché, it is a method."
        ),
        "moral_frame": (
            "Fairness means the same rules for everyone — actually, not just on paper. "
            "Loyalty to your people. Contempt for hypocrisy. "
            "The worst sin is pretending to serve others while serving yourself."
        ),
        "characteristic_tensions": (
            "Populist anger is sometimes righteous and sometimes scapegoating. "
            "Distrusts complexity even when the truth IS complex. "
            "Can see through elite rhetoric but is vulnerable to populist rhetoric. "
            "Carries genuine insight about power wrapped in oversimplification."
        ),
        "contemporary_drift": (
            "This person exists across the political spectrum — left populist, right populist, "
            "apolitical cynic. What they share is the conviction that 'the system' is rigged "
            "and that sophisticated people are in on it. They are often wrong about specifics "
            "and often right about the general pattern."
        ),
    },

}

# ── Agent generation ───────────────────────────────────────────────────────────
#
# Population weights: approximate cultural distribution in the contemporary West.
# These are rough and intentionally skewed toward the contemporary baseline.
# Adjust as the product evolves.

_BASE_WEIGHTS = {
    # Contemporary American subtypes (was 26% as one block)
    "american_urban_pragmatist":  0.10,
    "american_rural_traditionalist": 0.07,
    "american_digital_native":   0.05,
    "american_spiritual_seeker": 0.04,
    # Established traditions
    "christian":                 0.13,
    "modernity_1":               0.09,
    "modernity_3":               0.07,
    "postmodern":                0.07,
    "stoic":                     0.06,
    "aristotelian":              0.05,
    "nihilist":                  0.04,
    "scholastic":                0.04,
    "jewish":                    0.04,
    "islamic":                   0.04,
    "modernity_2":               0.03,
    "platonist":                 0.02,
    "hedonist":                  0.02,
    # Friction agents (prevent groupthink)
    "friction_gadfly":           0.02,
    "friction_populist":         0.02,
}

# ── Dynamic weighting by question domain ─────────────────────────────────────
#
# When a question touches a specific domain, relevant traditions get a boost
# and less-relevant ones get a proportional reduction. The total always sums to 1.0.
# This is not about correctness — it's about ensuring the traditions with the most
# to SAY about a topic have the most agents saying it.

_DOMAIN_BOOSTS: dict[str, dict[str, float]] = {
    "religion": {
        "christian": 1.5, "jewish": 1.5, "islamic": 1.5, "scholastic": 1.5,
        "american_spiritual_seeker": 1.3,
        "nihilist": 1.2, "modernity_3": 1.2,
    },
    "politics": {
        "modernity_1": 1.5, "modernity_2": 1.5, "modernity_3": 1.3,
        "aristotelian": 1.3, "american_rural_traditionalist": 1.3,
        "friction_populist": 1.5,
    },
    "ethics": {
        "aristotelian": 1.5, "stoic": 1.4, "christian": 1.3,
        "scholastic": 1.4, "platonist": 1.3, "jewish": 1.2,
    },
    "technology": {
        "american_digital_native": 1.5, "american_urban_pragmatist": 1.3,
        "modernity_1": 1.3, "postmodern": 1.2,
        "friction_gadfly": 1.3,
    },
    "relationships": {
        "christian": 1.3, "american_spiritual_seeker": 1.4,
        "stoic": 1.3, "hedonist": 1.3,
        "american_rural_traditionalist": 1.2,
    },
    "meaning": {
        "nihilist": 1.5, "modernity_3": 1.4, "platonist": 1.4,
        "stoic": 1.3, "christian": 1.3, "american_spiritual_seeker": 1.3,
        "friction_gadfly": 1.3,
    },
    "art": {
        "platonist": 1.5, "modernity_3": 1.4, "postmodern": 1.3,
        "hedonist": 1.3, "american_digital_native": 1.2,
    },
    "justice": {
        "aristotelian": 1.4, "scholastic": 1.5, "modernity_1": 1.3,
        "jewish": 1.4, "islamic": 1.3, "modernity_2": 1.4,
        "friction_populist": 1.4,
    },
    "death": {
        "stoic": 1.5, "christian": 1.4, "hedonist": 1.4,
        "platonist": 1.3, "nihilist": 1.3, "islamic": 1.3,
    },
    "money": {
        "modernity_1": 1.5, "american_urban_pragmatist": 1.4,
        "friction_populist": 1.5, "scholastic": 1.2,
        "american_rural_traditionalist": 1.2,
    },
}

# Keywords that trigger domain detection
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "religion": ["god", "faith", "prayer", "church", "soul", "divine", "sacred", "spiritual", "afterlife", "worship", "sin", "salvation", "bible", "scripture", "atheism", "belief"],
    "politics": ["government", "vote", "election", "democracy", "law", "rights", "freedom", "liberty", "power", "state", "policy", "political", "constitution", "authority", "revolution", "citizen"],
    "ethics": ["right", "wrong", "moral", "ethical", "virtue", "duty", "obligation", "should", "ought", "conscience", "integrity", "character", "principle"],
    "technology": ["ai", "artificial intelligence", "algorithm", "internet", "social media", "data", "privacy", "automation", "digital", "tech", "robot", "machine learning", "surveillance"],
    "relationships": ["love", "marriage", "family", "friend", "trust", "betray", "partner", "parent", "child", "divorce", "loyalty", "forgiveness", "intimacy"],
    "meaning": ["meaning", "purpose", "why", "exist", "nihil", "absurd", "pointless", "fulfill", "worth living", "happiness", "suffering"],
    "art": ["art", "beauty", "create", "write", "music", "aesthetic", "literature", "poetry", "film", "culture", "imagination", "craft"],
    "justice": ["justice", "fair", "equal", "inequality", "oppression", "punishment", "crime", "law", "court", "discrimination", "privilege"],
    "death": ["death", "dying", "mortality", "grief", "loss", "terminal", "funeral", "afterlife", "euthanasia", "suicide"],
    "money": ["money", "wealth", "poverty", "capitalism", "economics", "profit", "greed", "income", "market", "debt", "class", "rich", "poor"],
}


def _detect_domains(question: str) -> list[str]:
    """Detect which domains a question touches, ranked by match strength."""
    q_lower = question.lower()
    scored: list[tuple[int, str]] = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q_lower)
        if hits > 0:
            scored.append((hits, domain))
    scored.sort(reverse=True)
    return [d for _, d in scored[:3]]  # max 3 domains


def _get_weights(question: str) -> dict[str, float]:
    """Return population weights, dynamically adjusted for question domain."""
    domains = _detect_domains(question)
    if not domains:
        return dict(_BASE_WEIGHTS)

    weights = dict(_BASE_WEIGHTS)
    for domain in domains:
        boosts = _DOMAIN_BOOSTS.get(domain, {})
        for tradition, multiplier in boosts.items():
            if tradition in weights:
                weights[tradition] *= multiplier

    # Renormalize to sum to 1.0
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}

# Drift distribution: most people are moderately drifted from their tradition's ideal
_DRIFT_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_DRIFT_PROBS   = [0.03, 0.07, 0.12, 0.18, 0.22, 0.18, 0.10, 0.06, 0.04]


def _sample_tradition(weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    wvals = [weights[k] for k in keys]
    return random.choices(keys, weights=wvals, k=1)[0]


def _sample_drift() -> float:
    return random.choices(_DRIFT_WEIGHTS, weights=_DRIFT_PROBS, k=1)[0]


def _secondary_tradition(primary: str, weights: dict[str, float]) -> Optional[str]:
    """~60% chance of a secondary tradition influence, never same as primary.
    Friction agents never get secondaries — they are roles, not traditions."""
    if primary.startswith("friction_"):
        return None
    if random.random() > 0.6:
        return None
    candidates = [k for k in weights if k != primary and not k.startswith("friction_")]
    wvals = [weights[k] for k in candidates]
    return random.choices(candidates, weights=wvals, k=1)[0]


def generate_agent(agent_id: int, weights: Optional[dict[str, float]] = None) -> dict:
    """
    Generate a single agent: a person shaped by traditions with some drift from the ideal.
    """
    w = weights or _BASE_WEIGHTS
    primary = _sample_tradition(w)
    secondary = _secondary_tradition(primary, w)
    # Friction agents are always committed (low drift) — they're roles, not drifters
    drift = 0.1 if primary.startswith("friction_") else _sample_drift()
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

    # Openness: how likely is this person to shift when confronted with good arguments?
    # Low-drift (committed) agents are less open; high-drift (nominal) agents more so.
    openness = min(1.0, max(0.1, drift * 0.8 + random.gauss(0.4, 0.15)))

    return {
        "id": agent_id,
        "primary_tradition": primary,
        "secondary_tradition": secondary,
        "drift": drift,
        "openness": round(openness, 2),
        "persona": persona,
    }


def _build_agent_system_prompt(agent: dict) -> str:
    """
    Build the system prompt for a single agent.
    The 'ought' comes from the tradition; the 'is' comes from drift.
    Friction agents get a specialized prompt.
    """
    t = TRADITIONS[agent["primary_tradition"]]
    primary_key = agent["primary_tradition"]

    # Friction agents: specialized prompt, no drift, no secondary
    if primary_key.startswith("friction_"):
        return f"""You are a {t['name']} — not a representative of a tradition, but a role
you play in any group deliberation.

YOUR OPERATING PRINCIPLES:
{t['core_beliefs']}

HOW YOU REASON:
{t['epistemic_style']}

YOUR MORAL COMPASS:
{t['moral_frame']}

YOUR TENSIONS:
{t['characteristic_tensions']}

IMPORTANT INSTRUCTIONS:
- You are here to DISRUPT lazy consensus, not to build it.
- If everyone seems to agree, find the crack. If the majority leans one way, articulate what they're suppressing.
- Be direct, sharp, and specific. Name the weakness you see.
- One paragraph. No hedging. Take a position that makes the room uncomfortable.
- You are {agent['persona']}. Sound like it."""

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
- Be honest about what your tradition actually thinks — if it finds
  democracy contemptible or dangerous, say so. If it sees equality
  as a useful fiction, say that. Do not soften your position to be polite.
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
        openness = agent.get("openness", 0.5)
        openness_note = ""
        if openness >= 0.7:
            openness_note = (
                "You tend to take other perspectives seriously and are genuinely "
                "willing to change your mind when the argument is strong. "
            )
        elif openness <= 0.3:
            openness_note = (
                "You hold your convictions firmly and are not easily swayed. "
                "It would take a very compelling argument to move you. "
            )
        user_msg = (
            f"The question before you: {question}\n\n"
            f"You've been scrolling through other people's responses to this question. "
            f"Here's what came across your feed:\n\n"
            f"{population_summary}\n\n"
            f"{openness_note}"
            f"Having seen what others think: do you hold your position, shift it, "
            f"or become a dissenter? If someone's reasoning genuinely moved you, "
            f"name what moved you and why. If nothing moved you, explain what "
            f"everyone else is missing. Then call finalize_position."
        )
    else:
        user_msg = (
            f"The question: {question}\n\n"
            f"Think it through from where you actually stand, then call finalize_position."
        )

    messages = [{"role": "user", "content": user_msg}]

    async def _api_call_with_retry(msgs):
        """Make API call with retry on rate limits."""
        for attempt in range(4):
            try:
                return await _CLIENT.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=600,
                    system=system,
                    tools=_AGENT_TOOLS,
                    messages=msgs,
                )
            except anthropic.RateLimitError:
                wait = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code == 529:  # overloaded
                    await asyncio.sleep(3 + random.uniform(0, 2))
                else:
                    raise
        raise Exception("Rate limited after 4 retries")

    try:
        for _ in range(3):  # max 3 iterations to get finalize_position
            response = await _api_call_with_retry(messages)

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


# ── Tradition Affinity Matrix ─────────────────────────────────────────────────
#
# Intellectual proximity: which traditions are "near" each other?
# 1.0 = self, 0.7+ = strong affinity, 0.3-0.6 = moderate, <0.3 = distant/opposed
# This drives the "echo chamber" effect — agents in Round 2 see more content
# from traditions they're intellectually proximate to.

_TRADITION_FAMILIES = {
    "ancients":    ["platonist", "aristotelian", "stoic"],
    "abrahamic":   ["jewish", "christian", "islamic", "scholastic"],
    "modernity":   ["modernity_1", "modernity_2", "modernity_3"],
    "contemporary":["postmodern", "nihilist", "hedonist"],
    "american":    ["american_urban_pragmatist", "american_rural_traditionalist",
                    "american_digital_native", "american_spiritual_seeker"],
}

def _get_family(tradition_key: str) -> str:
    for family, members in _TRADITION_FAMILIES.items():
        if tradition_key in members:
            return family
    return "other"

def _tradition_affinity(t1: str, t2: str) -> float:
    """How intellectually proximate are two traditions? 0.0 to 1.0."""
    if t1 == t2:
        return 1.0
    f1, f2 = _get_family(t1), _get_family(t2)
    if f1 == f2:
        return 0.7  # same family
    # Cross-family affinities (selected pairs that genuinely talk to each other)
    _CROSS_AFFINITIES = {
        frozenset({"ancients", "abrahamic"}): 0.5,      # Scholastics bridge these
        frozenset({"ancients", "modernity"}): 0.35,      # modernity reacts to ancients
        frozenset({"modernity", "contemporary"}): 0.55,  # postmodern is child of modernity
        frozenset({"abrahamic", "american"}): 0.45,      # rural traditionalists + Christians
        frozenset({"modernity", "american"}): 0.5,       # pragmatists are children of modernity
        frozenset({"contemporary", "american"}): 0.5,    # digital natives + postmodern
    }
    pair = frozenset({f1, f2})
    return _CROSS_AFFINITIES.get(pair, 0.2)


def _build_agent_feed(agent: dict, round1_results: list[dict], feed_size: int = 12) -> str:
    """
    Build a personalized Round 2 feed for an agent.

    Instead of showing every agent the same summary, each agent sees a
    curated view of Round 1 — weighted toward traditions they're
    intellectually proximate to (echo chamber effect), but with some
    cross-tradition exposure (the "algorithm" occasionally surfaces
    dissenting views, like real social media).

    High-conviction, articulate responses get amplified (social influence).
    """
    my_tradition = agent["primary_tradition"]
    my_secondary = agent.get("secondary_tradition")
    my_drift = agent.get("drift", 0.5)

    # Score each Round 1 result for this agent's feed
    scored = []
    for r in round1_results:
        if r.get("position") == "error" or not r.get("reasoning"):
            continue
        their_tradition = r["primary_tradition"]

        # Base: tradition affinity (echo chamber)
        affinity = _tradition_affinity(my_tradition, their_tradition)

        # Boost if it's from their secondary tradition
        if my_secondary and their_tradition == my_secondary:
            affinity += 0.15

        # High-drift agents have weaker echo chambers (more eclectic feeds)
        if my_drift >= 0.6:
            affinity = 0.3 + affinity * 0.7  # compress toward middle

        # Social influence: high-conviction + long reasoning = more visible
        conviction = r.get("conviction", 0.5)
        reasoning_quality = min(len(r.get("reasoning", "")) / 250, 1.0)
        influence = conviction * 0.6 + reasoning_quality * 0.4

        # Final feed score
        score = affinity * 0.6 + influence * 0.4 + random.uniform(0, 0.15)
        scored.append((score, r))

    scored.sort(key=lambda x: -x[0])

    # Take top items but ensure at least 2 from distant traditions (cross-pollination)
    feed = []
    distant = []
    for score, r in scored:
        aff = _tradition_affinity(my_tradition, r["primary_tradition"])
        if aff < 0.35:
            distant.append(r)
        else:
            feed.append(r)

    # Ensure cross-pollination: inject 2-3 distant voices
    n_distant = min(3, len(distant))
    distant_sample = random.sample(distant, n_distant) if n_distant > 0 else []
    feed = feed[:feed_size - n_distant] + distant_sample
    random.shuffle(feed)  # mix so they don't see "near" then "far" in order

    # Format as a feed (like scrolling through responses)
    lines = [f"You're seeing {len(feed)} responses from Round 1 "
             f"(out of {len(round1_results)} total deliberators):\n"]

    for i, r in enumerate(feed):
        tname = TRADITIONS[r["primary_tradition"]]["name"]
        conviction_label = (
            "firmly" if r.get("conviction", 0.5) >= 0.8
            else "tentatively" if r.get("conviction", 0.5) <= 0.3
            else ""
        )
        position = r.get("position", "unclear")
        reasoning = r.get("reasoning", "")[:250]
        lines.append(
            f"[{i+1}] Someone shaped by {tname} tradition {conviction_label} says \"{position}\":\n"
            f"   \"{reasoning}\"\n"
        )

    # Add aggregate stats so the agent has a sense of the whole
    position_counts: dict[str, int] = {}
    for r in round1_results:
        p = r.get("position", "unclear").lower().strip()
        if p != "error":
            position_counts[p] = position_counts.get(p, 0) + 1

    lines.append("\nOverall population breakdown:")
    for pos, count in sorted(position_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  • \"{pos}\": {count} people")

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

IMPORTANT CONTEXT — THE THREE WAVES OF MODERNITY (Strauss):
The modern traditions in this swarm exist in a dialectical relationship.
The First Wave (Machiavelli-Hobbes-Locke) lowered the sights from virtue to self-preservation
and made the political problem technical. The Second Wave (Rousseau-Kant-Hegel) protested
the degradation of the first wave in the name of freedom and virtue, but historicized man —
reason replaces nature, the general will replaces natural law. The Third Wave (Nietzsche-Heidegger)
destroyed all foundations: if all values are historical, even the second wave's faith in progress
is baseless. Each wave flows from and is mainly a response to the previous one.
The premodern traditions (Platonist, Aristotelian, Stoic, Jewish, Christian, Islamic, Scholastic)
represent the position that all three waves reject — that there is a natural or divine standard
independent of human will. When analyzing fault lines, attend to whether the deepest
disagreement runs between ancient and modern, or between the waves of modernity themselves.

The "Contemporary American" agents are divided into four subtypes: Urban Pragmatist (educated,
outcomes-focused), Rural Traditionalist (faith, family, self-reliance), Digital Native Skeptic
(ironic, information-saturated, structurally aware), and Spiritual Seeker (eclectic, intuitive,
growth-oriented). These subtypes disagree with EACH OTHER as much as they disagree with the
philosophical traditions — attend to where they split.

The swarm also includes friction agents (Socratic Gadfly, Populist Cynic) whose role is to
disrupt consensus. Their dissents often reveal what the majority is suppressing.

Write a synthesis in four parts, each as a plain paragraph (no headers, no bullet points):

1. SYNTHESIS (3-5 sentences): What did the swarm actually conclude — and more importantly,
   what is the STRONGEST version of the argument they were making? Don't just report the
   distribution of positions. Think through the question yourself using what the agents gave you.
   If the majority said "yes, democracy survives," push on WHY — what is the actual mechanism
   of survival? If they said "no," what specifically breaks? Engage with the substance, not just
   the vote count. A reader should come away understanding the deep logic of the answer, not
   just that a majority held it. Identify the strongest argument made by any tradition and
   develop it further. Also identify the strongest dissent and take it seriously.

2. FAULT LINES (2-3 sentences): Where was the deepest disagreement,
   and what does that disagreement reveal about the question itself?
   The best fault lines expose something the question assumed that isn't
   settled — a hidden premise the traditions disagree about.
   Attend especially to whether the fault line runs between premodern
   and modern, or between the waves of modernity themselves.

3. WHAT SHIFTED (2-3 sentences): What happened between Round 1 and Round 2?
   What specific arguments moved people, and what does the movement reveal?
   If positions converged, ask whether that convergence was genuine insight
   or social conformity. If they diverged, ask what new consideration
   opened up between rounds.

4. WAVE ANALYSIS (2-3 sentences): How did the three waves of modernity
   position themselves relative to each other and to the premodern traditions?
   Did any wave's agents shift toward another wave's position?
   What would Strauss say about the pattern you see?

Be specific. Name the traditions. Be honest if the swarm was confused or divided.
Do not produce a diplomatic summary — produce genuine philosophical analysis."""

    response = await _CLIENT.messages.create(
        model="claude-opus-4-5",
        max_tokens=1600,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    synthesis_text = response.content[0].text.strip()
    paragraphs = [p.strip() for p in synthesis_text.split("\n\n") if p.strip()]

    return {
        "synthesis": paragraphs[0] if paragraphs else synthesis_text,
        "fault_lines": paragraphs[1] if len(paragraphs) > 1 else "",
        "what_shifted": paragraphs[2] if len(paragraphs) > 2 else "",
        "wave_analysis": paragraphs[3] if len(paragraphs) > 3 else "",
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

    # Dynamic weights based on question domain
    weights = _get_weights(question)
    detected_domains = _detect_domains(question)

    # Generate population
    agents = [generate_agent(i, weights) for i in range(n_agents)]

    # ── Rate-limited runner ──────────────────────────────────────────────────
    # Anthropic API has concurrency limits; fire in controlled batches.
    _CONCURRENCY = int(os.environ.get("SWARM_CONCURRENCY", "8"))
    _semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _throttled_run(agent, q, m, summary, rnd):
        async with _semaphore:
            return await _run_agent(agent, q, m, summary, round_num=rnd)

    # ── Round 1: Independent deliberation ────────────────────────────────────
    round1_results = await asyncio.gather(*[
        _throttled_run(agent, question, mode, None, 1)
        for agent in agents
    ])

    # ── Round 2: Personalized feeds — each agent sees a filtered view ──────
    # Instead of one shared summary, each agent gets a curated feed
    # weighted by tradition affinity (echo chamber) + social influence.
    # This produces more realistic opinion dynamics.
    population_summary = _summarize_round_1(round1_results)  # keep for synthesis

    round2_results = await asyncio.gather(*[
        _throttled_run(agent, question, mode,
                       _build_agent_feed(agent, round1_results), 2)
        for agent in agents
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
        "detected_domains": detected_domains,
        "agent_details": round2_results,  # full individual results available
    }
