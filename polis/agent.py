"""
Agent system with dual-layer reasoning.

Each agent has two layers:
  1. Tradition — the best arguments their intellectual framework can produce
  2. Psychology — the biases, incentives, fears, and ego that constrain
     how they actually behave

The gap between what an agent's tradition recommends and what the agent
actually does is where the insight lives.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BiasType(Enum):
    """Cognitive biases that warp ideal reasoning."""
    LOSS_AVERSION = "loss_aversion"           # Overweights potential losses
    STATUS_QUO = "status_quo"                 # Resists change regardless of merit
    CONFIRMATION = "confirmation"             # Seeks evidence for existing beliefs
    SUNK_COST = "sunk_cost"                   # Doubles down on past commitments
    BANDWAGON = "bandwagon"                   # Follows perceived majority
    AUTHORITY = "authority"                    # Defers to power regardless of argument
    IN_GROUP = "in_group"                     # Favors own coalition's position
    AVAILABILITY = "availability"             # Overweights recent/vivid events
    ANCHORING = "anchoring"                   # Fixates on first information received
    SELF_SERVING = "self_serving"             # Interprets evidence to benefit self


@dataclass
class Incentives:
    """What the agent stands to gain or lose. These warp reasoning."""
    primary_goal: str               # What they actually want
    public_goal: str                # What they claim to want
    fears: list[str]                # What they're afraid of losing
    pressures: list[str]            # External forces acting on them
    reelection: bool = False        # Must answer to a constituency
    financial_stake: float = 0.0    # -1.0 (loses everything) to 1.0 (gains everything)
    reputation_stake: float = 0.0   # How much their reputation rides on this


@dataclass
class Psychology:
    """The human constraints that sit between ideal reasoning and actual behavior."""
    biases: list[BiasType]
    incentives: Incentives
    stubbornness: float = 0.5       # 0.0 = will change position instantly, 1.0 = immovable
    courage: float = 0.5            # 0.0 = caves under any pressure, 1.0 = will die on this hill
    vanity: float = 0.5             # 0.0 = ego-free, 1.0 = cannot admit being wrong
    empathy: float = 0.5            # 0.0 = purely transactional, 1.0 = deeply moved by others
    crowd_sensitivity: float = 0.5  # 0.0 = immune to social pressure, 1.0 = pure conformist

    def describe(self) -> str:
        """Human-readable description for prompt injection."""
        bias_names = [b.value.replace("_", " ") for b in self.biases]
        lines = [
            f"Primary goal (private): {self.incentives.primary_goal}",
            f"Stated goal (public): {self.incentives.public_goal}",
            f"Fears: {', '.join(self.incentives.fears)}",
            f"External pressures: {', '.join(self.incentives.pressures)}",
            f"Cognitive biases: {', '.join(bias_names)}",
            f"Stubbornness: {self.stubbornness:.1f}/1.0",
            f"Courage: {self.courage:.1f}/1.0",
            f"Vanity: {self.vanity:.1f}/1.0",
            f"Empathy: {self.empathy:.1f}/1.0",
            f"Crowd sensitivity: {self.crowd_sensitivity:.1f}/1.0",
        ]
        if self.incentives.reelection:
            lines.append("CONSTRAINT: Must answer to a constituency — survival depends on popular approval.")
        if abs(self.incentives.financial_stake) > 0.5:
            direction = "gain" if self.incentives.financial_stake > 0 else "lose"
            lines.append(f"CONSTRAINT: Stands to {direction} significantly from the outcome.")
        return "\n".join(lines)


@dataclass
class Memory:
    """What the agent remembers from previous deliberation rounds."""
    arguments_heard: list[dict] = field(default_factory=list)
    positions_taken: list[dict] = field(default_factory=list)
    alliances: dict = field(default_factory=dict)          # agent_name -> sentiment (-1 to 1)
    concessions_made: list[str] = field(default_factory=list)
    wounds: list[str] = field(default_factory=list)         # moments of public embarrassment or challenge

    def record_argument(self, round_num: int, speaker: str, argument: str, persuasive: bool):
        self.arguments_heard.append({
            "round": round_num,
            "speaker": speaker,
            "argument": argument,
            "found_persuasive": persuasive,
        })

    def record_position(self, round_num: int, position: str, confidence: float):
        self.positions_taken.append({
            "round": round_num,
            "position": position,
            "confidence": confidence,
        })

    def record_wound(self, round_num: int, description: str):
        self.wounds.append(f"Round {round_num}: {description}")

    def update_alliance(self, agent_name: str, delta: float):
        current = self.alliances.get(agent_name, 0.0)
        self.alliances[agent_name] = max(-1.0, min(1.0, current + delta))

    def summary(self) -> str:
        """Compressed memory for prompt context."""
        lines = []
        if self.positions_taken:
            last = self.positions_taken[-1]
            lines.append(f"Your current position: {last['position']} (confidence: {last['confidence']:.1f})")
        if self.alliances:
            allies = [n for n, s in self.alliances.items() if s > 0.3]
            rivals = [n for n, s in self.alliances.items() if s < -0.3]
            if allies:
                lines.append(f"You feel aligned with: {', '.join(allies)}")
            if rivals:
                lines.append(f"You feel opposed to: {', '.join(rivals)}")
        if self.wounds:
            lines.append(f"Moments that stung: {'; '.join(self.wounds[-3:])}")
        if self.concessions_made:
            lines.append(f"Concessions you've made: {'; '.join(self.concessions_made[-3:])}")
        recent = [a for a in self.arguments_heard if a["found_persuasive"]][-3:]
        if recent:
            for a in recent:
                lines.append(f"Argument from {a['speaker']} that moved you: {a['argument'][:120]}")
        return "\n".join(lines) if lines else "No prior deliberation history."


@dataclass
class Agent:
    """
    A deliberating agent with both ideal reasoning and psychological constraints.

    The tradition gives them access to the best arguments.
    The psychology determines what they actually do with those arguments.
    The memory tracks how the deliberation has changed them.
    """
    name: str
    role: str                       # e.g., "State legislator from a swing district"
    tradition: str                  # Name of intellectual tradition (key into traditions/)
    tradition_prompt: str           # Full tradition description loaded from file
    backstory: str                  # Specific biographical/contextual details
    psychology: Psychology
    memory: Memory = field(default_factory=Memory)

    def ideal_prompt(self, scenario: str, question: str,
                     source_context: str = "") -> str:
        """Prompt for what this agent's tradition recommends — no psychological constraints."""
        source_section = ""
        if source_context:
            source_section = f"\n\n{source_context}\nUse these source passages to ground your reasoning in the actual texts, not summaries.\n"

        return f"""You are reasoning from the {self.tradition} tradition.

{self.tradition_prompt}
{source_section}
SCENARIO:
{scenario}

QUESTION:
{question}

Reason carefully from within this tradition. What is the strongest, most honest argument this framework can produce? Do not hedge or qualify — give the best answer this tradition has to offer. Be specific and grounded, not abstract.

Respond in 2-3 paragraphs."""

    def constrained_prompt(self, scenario: str, question: str,
                           other_positions: list[dict], round_num: int,
                           source_context: str = "") -> str:
        """Prompt for what the agent actually says — ideal reasoning filtered through psychology."""
        others_text = ""
        if other_positions:
            others_text = "\n\nWHAT OTHERS HAVE SAID THIS ROUND:\n"
            for p in other_positions:
                others_text += f"- {p['name']} ({p['role']}): {p['position'][:200]}\n"

        memory_text = self.memory.summary()

        source_section = ""
        if source_context:
            source_section = f"\n\nSOURCE MATERIAL FROM YOUR TRADITION:\n{source_context}\n"

        return f"""You are {self.name}, {self.role}.

BACKSTORY:
{self.backstory}

YOUR INTELLECTUAL FRAMEWORK:
You draw on the {self.tradition} tradition:
{self.tradition_prompt}
{source_section}

YOUR PSYCHOLOGY:
{self.psychology.describe()}

YOUR MEMORY OF THIS DELIBERATION:
{memory_text}

SCENARIO:
{scenario}

QUESTION:
{question}
{others_text}

This is round {round_num} of deliberation.

You have access to the best arguments your tradition offers. But you are also a real person with real fears, real incentives, and real cognitive biases. Your biases and incentives WILL affect how you reason — not by making you stupid, but by subtly shaping which arguments you find compelling, which evidence you weight heavily, which concessions feel tolerable, and which positions feel dangerous.

Respond as {self.name} would actually respond in this moment — not as a philosopher in a seminar, but as a person with something at stake. If your incentives push against what your tradition recommends, feel that tension. You may rationalize, you may hedge, you may find clever reasons to avoid the conclusion your framework demands. That is human.

Respond with:
1. POSITION: Your actual position in 1-2 sentences.
2. ARGUMENT: Your reasoning in 2-3 paragraphs. This should feel like a real person talking, not a treatise.
3. CONFIDENCE: A number from 0.0 to 1.0 indicating how confident you are.
4. MOVED_BY: Name any other agent whose argument affected your thinking this round, or "none".
5. PRIVATE_THOUGHT: One sentence of what you're actually thinking but would never say aloud."""

    def __repr__(self):
        return f"Agent({self.name}, {self.role}, tradition={self.tradition})"
