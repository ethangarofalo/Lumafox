"""
Scenario loading and agent generation.

A scenario is a YAML or Python dict that describes:
  - The situation
  - The question to be deliberated
  - The cast of agents with their roles, traditions, and psychological profiles
"""

import json
import os
from pathlib import Path
from agent import Agent, Psychology, Incentives, BiasType, Memory


TRADITIONS_DIR = Path(__file__).parent / "traditions"
REFINEMENTS_DIR = TRADITIONS_DIR / "refinements"


def load_tradition(name: str) -> str:
    """Load a tradition's prompt text from the traditions/ directory,
    including any refinements from teaching sessions."""
    path = TRADITIONS_DIR / f"{name}.md"
    if not path.exists():
        available = [f.stem for f in TRADITIONS_DIR.glob("*.md") if not f.stem.endswith("_refined")]
        raise FileNotFoundError(
            f"Tradition '{name}' not found. Available: {available}"
        )
    base_text = path.read_text()

    # Load refinements if they exist
    refinement_path = REFINEMENTS_DIR / f"{name}.jsonl"
    if refinement_path.exists():
        refinements = []
        with open(refinement_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    refinements.append(json.loads(line))

        if refinements:
            sections = {
                "correction": [], "example": [], "principle": [],
                "voice_note": [], "anti_pattern": [],
            }
            for r in refinements:
                rtype = r.get("type", "correction")
                if rtype in sections:
                    sections[rtype].append(r.get("content", ""))

            lines = ["\n\n## REFINEMENTS FROM THE TEACHER\n"]
            lines.append("The following have been provided by someone who has studied")
            lines.append("this tradition deeply. They take PRECEDENCE over the base description.\n")

            if sections["principle"]:
                lines.append("### Core Principles (learned)")
                for p in sections["principle"]:
                    lines.append(f"- {p}")
                lines.append("")
            if sections["correction"]:
                lines.append("### Corrections")
                for c in sections["correction"]:
                    lines.append(f"- {c}")
                lines.append("")
            if sections["example"]:
                lines.append("### Examples of How This Tradition Actually Speaks")
                for e in sections["example"]:
                    lines.append(f"\n> {e}")
                lines.append("")
            if sections["voice_note"]:
                lines.append("### Voice and Tone Notes")
                for v in sections["voice_note"]:
                    lines.append(f"- {v}")
                lines.append("")
            if sections["anti_pattern"]:
                lines.append("### What This Tradition Would NEVER Say")
                for a in sections["anti_pattern"]:
                    lines.append(f"- {a}")
                lines.append("")

            base_text += "\n".join(lines)

    return base_text


def build_scenario(config: dict) -> tuple[str, str, list[Agent]]:
    """
    Build a scenario from a config dict.

    Config shape:
    {
        "scenario": str,       # Description of the situation
        "question": str,       # The question to deliberate
        "agents": [
            {
                "name": str,
                "role": str,
                "tradition": str,       # Must match a file in traditions/
                "backstory": str,
                "psychology": {
                    "biases": [str],    # BiasType values
                    "incentives": {
                        "primary_goal": str,
                        "public_goal": str,
                        "fears": [str],
                        "pressures": [str],
                        "reelection": bool,
                        "financial_stake": float,
                        "reputation_stake": float,
                    },
                    "stubbornness": float,
                    "courage": float,
                    "vanity": float,
                    "empathy": float,
                    "crowd_sensitivity": float,
                },
            },
            ...
        ]
    }
    """
    scenario = config["scenario"]
    question = config["question"]
    agents = []

    for agent_config in config["agents"]:
        psych_config = agent_config["psychology"]
        inc_config = psych_config["incentives"]

        incentives = Incentives(
            primary_goal=inc_config["primary_goal"],
            public_goal=inc_config["public_goal"],
            fears=inc_config.get("fears", []),
            pressures=inc_config.get("pressures", []),
            reelection=inc_config.get("reelection", False),
            financial_stake=inc_config.get("financial_stake", 0.0),
            reputation_stake=inc_config.get("reputation_stake", 0.0),
        )

        biases = [BiasType(b) for b in psych_config.get("biases", [])]

        psychology = Psychology(
            biases=biases,
            incentives=incentives,
            stubbornness=psych_config.get("stubbornness", 0.5),
            courage=psych_config.get("courage", 0.5),
            vanity=psych_config.get("vanity", 0.5),
            empathy=psych_config.get("empathy", 0.5),
            crowd_sensitivity=psych_config.get("crowd_sensitivity", 0.5),
        )

        tradition_name = agent_config["tradition"]
        tradition_prompt = load_tradition(tradition_name)

        agent = Agent(
            name=agent_config["name"],
            role=agent_config["role"],
            tradition=tradition_name,
            tradition_prompt=tradition_prompt,
            backstory=agent_config["backstory"],
            psychology=psychology,
            memory=Memory(),
        )
        agents.append(agent)

    return scenario, question, agents


def filter_agents(agents: list[Agent], filter_str: str) -> list[Agent]:
    """
    Filter a list of agents by name or tradition.

    Accepts a comma-separated string. Matches case-insensitively against
    both agent.name and agent.tradition. So 'Socrates', 'socrates',
    'socratic', and 'Socratic' all match the Socratic agent.

    Examples:
        filter_agents(agents, "Socrates,Machiavelli")
        filter_agents(agents, "socratic,locke")
        filter_agents(agents, "Mill")
    """
    keys = [k.strip().lower() for k in filter_str.split(",") if k.strip()]
    if not keys:
        return agents

    filtered = []
    for agent in agents:
        name_lower = agent.name.lower()
        tradition_lower = agent.tradition.lower()
        for key in keys:
            if key in name_lower or key in tradition_lower:
                filtered.append(agent)
                break
    return filtered


# ──────────────────────────────────────────────────────────────────────
# DEMO SCENARIO: Should frontier AI models be open-sourced?
# ──────────────────────────────────────────────────────────────────────

DEMO_SCENARIO = {
    "scenario": """The year is 2026. A major frontier AI lab has developed a model that
significantly advances autonomous agent capabilities — it can write and execute complex
code, conduct multi-step research, and operate computer interfaces with near-human
proficiency. The model's weights are currently proprietary. A coalition of researchers,
startups, and civil society organizations is pressuring the lab to open-source the weights,
arguing that concentrated control over transformative AI is dangerous. The lab's leadership
is divided. Governments are watching but have not yet legislated. The public is largely
unaware of the technical stakes but increasingly anxious about AI in general.

A closed-door deliberation has been convened with five participants representing different
stakeholder perspectives. They must produce a recommendation.""",

    "question": "Should the weights of frontier AI models be released as open-source, and if so, under what conditions?",

    "agents": [
        {
            "name": "Dr. Sarah Chen",
            "role": "Chief scientist at the AI lab that developed the model",
            "tradition": "pragmatist",
            "backstory": """Sarah led the team that built the model. She is a genuine
believer in open science and published her PhD work openly. But she has watched the
capabilities of this model in internal testing and is quietly terrified by some of what
it can do. She has also built her career at this company and owns significant equity.
The board is split and watching how she handles this. Her public reputation is as an
advocate for responsible AI — she gave a TED talk on it last year that went viral.
She knows that if the model causes harm after release, she will be the face of the failure.""",
            "psychology": {
                "biases": ["loss_aversion", "status_quo", "self_serving"],
                "incentives": {
                    "primary_goal": "Protect her legacy and avoid being responsible for catastrophic misuse",
                    "public_goal": "Ensure AI development benefits humanity through thoughtful, evidence-based policy",
                    "fears": ["Being blamed for harm", "Losing scientific credibility", "The model being weaponized"],
                    "pressures": ["Board pressure to maintain competitive advantage", "Research community pressure to open-source", "Media scrutiny"],
                    "reelection": False,
                    "financial_stake": 0.6,
                    "reputation_stake": 0.9,
                },
                "stubbornness": 0.4,
                "courage": 0.6,
                "vanity": 0.7,
                "empathy": 0.6,
                "crowd_sensitivity": 0.5,
            },
        },
        {
            "name": "Marcus Rivera",
            "role": "U.S. Senator, chair of the Senate AI subcommittee, facing reelection in 8 months",
            "tradition": "machiavelli",
            "backstory": """Marcus represents a state with both a major tech corridor and
a large working-class population anxious about automation. He genuinely cares about his
constituents but is also acutely aware that his reelection depends on not alienating either
the tech donors or the labor base. He does not deeply understand the technology — his
briefings come from staffers with their own agendas. He has been burned before by taking
a strong position on tech policy that was later used against him in attack ads. His instinct
is to find a position that sounds strong but commits to nothing irreversible.""",
            "psychology": {
                "biases": ["bandwagon", "loss_aversion", "anchoring", "authority"],
                "incentives": {
                    "primary_goal": "Survive reelection without taking a position that can be used against him",
                    "public_goal": "Protect American workers and families while fostering responsible innovation",
                    "fears": ["Attack ads", "Alienating donors", "Looking ignorant about technology", "Being on the wrong side of history"],
                    "pressures": ["Tech industry donors", "Labor union constituents", "Media coverage", "Party leadership"],
                    "reelection": True,
                    "financial_stake": 0.0,
                    "reputation_stake": 0.8,
                },
                "stubbornness": 0.3,
                "courage": 0.3,
                "vanity": 0.6,
                "empathy": 0.5,
                "crowd_sensitivity": 0.9,
            },
        },
        {
            "name": "Amara Osei",
            "role": "Executive director of a digital rights nonprofit",
            "tradition": "locke",
            "backstory": """Amara built her organization from nothing into one of the most
respected voices on technology and civil liberties. She has spent her career fighting
concentrated power — first government surveillance, now corporate AI monopolies. She
genuinely believes that open-source is the only check on power that works. She has seen
too many 'responsible governance' frameworks become tools for incumbents to lock out
competitors. She is also aware that her organization's donors are heavily aligned with
the open-source movement, and that taking a nuanced position could cost her funding.
Her public brand is fearlessness and moral clarity.""",
            "psychology": {
                "biases": ["confirmation", "in_group", "self_serving"],
                "incentives": {
                    "primary_goal": "Maintain her organization's brand as the uncompromising voice for openness",
                    "public_goal": "Prevent dangerous concentration of AI power in the hands of a few corporations",
                    "fears": ["Being seen as compromising with power", "Losing donor support", "Irrelevance"],
                    "pressures": ["Donor expectations", "Coalition partners", "Staff who are true believers", "Media positioning"],
                    "reelection": False,
                    "financial_stake": 0.3,
                    "reputation_stake": 0.8,
                },
                "stubbornness": 0.8,
                "courage": 0.8,
                "vanity": 0.5,
                "empathy": 0.4,
                "crowd_sensitivity": 0.3,
            },
        },
        {
            "name": "James Whitfield",
            "role": "Retired four-star general, former head of U.S. Cyber Command",
            "tradition": "utilitarian",
            "backstory": """James spent 35 years in military and intelligence. He has seen
classified briefings on what nation-state actors are attempting with AI. He is genuinely
frightened — not performatively, but with the quiet fear of a man who has seen what
determined adversaries do with powerful tools. He also sits on the boards of three defense
contractors that would benefit from restricted AI access. He does not think of this as a
conflict of interest — he believes national security and his board positions are aligned.
He respects expertise and chain of command. He is uncomfortable with the informality of
this civilian deliberation.""",
            "psychology": {
                "biases": ["authority", "availability", "loss_aversion", "in_group"],
                "incentives": {
                    "primary_goal": "Prevent adversaries from accessing capabilities that threaten national security",
                    "public_goal": "Ensure AI development does not compromise American security or global stability",
                    "fears": ["Adversary use of open-source AI for cyberweapons", "Loss of American technological advantage", "Being seen as naive"],
                    "pressures": ["Defense contractor board obligations", "National security community expectations", "Intelligence briefing knowledge he cannot share"],
                    "reelection": False,
                    "financial_stake": 0.5,
                    "reputation_stake": 0.6,
                },
                "stubbornness": 0.7,
                "courage": 0.7,
                "vanity": 0.4,
                "empathy": 0.3,
                "crowd_sensitivity": 0.2,
            },
        },
        {
            "name": "Priya Sharma",
            "role": "Founder and CEO of an AI startup in Bangalore building on open-source models",
            "tradition": "aristotelian",
            "backstory": """Priya left Google to build an AI company that brings advanced
capabilities to markets the big labs ignore — agricultural planning, regional language
translation, healthcare triage in areas with no doctors. Her entire business depends on
access to open-source model weights. She fine-tunes them for specific applications that
the frontier labs would never build because the markets are too small. She has seen
firsthand what happens when communities are locked out of transformative technology —
her grandmother died of a treatable condition because there was no diagnostic capacity
in her village. She is also a rigorous thinker who studied philosophy at Oxford before
switching to computer science. She understands the safety concerns better than her
advocacy position might suggest.""",
            "psychology": {
                "biases": ["confirmation", "sunk_cost", "availability"],
                "incentives": {
                    "primary_goal": "Ensure continued access to open-source weights so her company and its users survive",
                    "public_goal": "Democratize AI access so its benefits reach the communities that need them most",
                    "fears": ["Regulatory capture by incumbents", "Her company being locked out of the foundation models it depends on", "The global south being left behind again"],
                    "pressures": ["Investors expecting growth", "Customers depending on her products", "The moral weight of the communities she serves"],
                    "reelection": False,
                    "financial_stake": 0.9,
                    "reputation_stake": 0.6,
                },
                "stubbornness": 0.5,
                "courage": 0.7,
                "vanity": 0.3,
                "empathy": 0.9,
                "crowd_sensitivity": 0.3,
            },
        },
    ],
}
