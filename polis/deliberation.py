"""
The deliberation engine.

Manages rounds of multi-agent deliberation, collecting positions,
routing arguments between agents, tracking shifts, and producing
both the ideal-reasoning layer and the psychologically-constrained layer.
"""

import json
import re
import time
import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

from agent import Agent


@dataclass
class RoundResult:
    """What happened in a single round of deliberation."""
    round_num: int
    positions: list[dict]       # [{name, role, position, argument, confidence, moved_by, private_thought}]
    shifts: list[dict]          # [{name, old_position, new_position, reason}]
    alliances_formed: list[str] # Human-readable descriptions
    tensions: list[str]         # Points of irreconcilable disagreement


@dataclass
class DeliberationResult:
    """The full output of a deliberation run."""
    scenario: str
    question: str
    rounds: list[RoundResult]
    ideal_positions: list[dict]     # What each tradition recommends without psychological constraints
    final_positions: list[dict]     # Where agents actually ended up
    gap_analysis: str               # The distance between ideal and actual — this is the insight
    narrative: str                  # Human-readable story of what happened


class DeliberationEngine:
    """
    Runs multi-round deliberation between agents.

    The engine makes two passes:
      1. IDEAL PASS — each agent's tradition reasons without constraints
      2. CONSTRAINED PASS — agents deliberate as real people with psychology

    The gap between the two passes is the report's central finding.
    """

    def __init__(self, llm_call: Callable[[str], str], rounds: int = 5,
                 verbose: bool = True, knowledge_graph=None, track_observations: bool = True):
        """
        Args:
            llm_call: Function that takes a prompt string and returns a completion string.
                      This is the only interface to the LLM — keeps the engine model-agnostic.
            rounds: Number of deliberation rounds.
            verbose: Print progress to stdout.
            knowledge_graph: Optional KnowledgeGraph instance for source-grounded retrieval.
            track_observations: Whether to record observations for the self-improvement system.
        """
        self.llm_call = llm_call
        self.rounds = rounds
        self.verbose = verbose
        self.knowledge_graph = knowledge_graph
        self.track_observations = track_observations
        self._source_contexts: dict[str, str] = {}  # tradition -> retrieved context text

    def run(self, scenario: str, question: str, agents: list[Agent]) -> DeliberationResult:
        """Execute a full deliberation and return results."""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  POLIS — Deliberation Engine")
            print(f"  Scenario: {scenario[:80]}...")
            print(f"  Question: {question[:80]}...")
            print(f"  Agents: {', '.join(a.name for a in agents)}")
            print(f"  Rounds: {self.rounds}")
            print(f"{'='*70}\n")

        # === RETRIEVE SOURCE CONTEXT (if knowledge graph available) ===
        if self.knowledge_graph:
            if self.verbose:
                print("─── RETRIEVING SOURCE CONTEXT ───")
            self._retrieve_source_contexts(question, agents)

        # === PASS 1: Ideal reasoning (no psychology) ===
        if self.verbose:
            print("─── IDEAL REASONING PASS ───")
            print("Each tradition reasons without psychological constraints.\n")

        ideal_positions = self._run_ideal_pass(scenario, question, agents)
        self._ideal_positions_cache = ideal_positions  # Cache for observation recording

        # === PASS 2: Constrained deliberation (real people) ===
        if self.verbose:
            print("\n─── CONSTRAINED DELIBERATION PASS ───")
            print("Agents deliberate as real people with biases, incentives, and ego.\n")

        round_results = []
        for round_num in range(1, self.rounds + 1):
            result = self._run_round(scenario, question, agents, round_num)
            round_results.append(result)

        # === Collect final positions ===
        final_positions = []
        for agent in agents:
            if agent.memory.positions_taken:
                last = agent.memory.positions_taken[-1]
                final_positions.append({
                    "name": agent.name,
                    "role": agent.role,
                    "tradition": agent.tradition,
                    "position": last["position"],
                    "confidence": last["confidence"],
                })

        # === Gap analysis ===
        gap_analysis = self._analyze_gap(ideal_positions, final_positions, agents)

        # === Narrative ===
        narrative = self._generate_narrative(scenario, question, ideal_positions,
                                              round_results, final_positions, agents)

        return DeliberationResult(
            scenario=scenario,
            question=question,
            rounds=round_results,
            ideal_positions=ideal_positions,
            final_positions=final_positions,
            gap_analysis=gap_analysis,
            narrative=narrative,
        )

    def _retrieve_source_contexts(self, question: str, agents: list[Agent]):
        """Retrieve relevant source material for each tradition from the knowledge graph."""
        seen_traditions = set()
        for agent in agents:
            if agent.tradition in seen_traditions:
                continue
            seen_traditions.add(agent.tradition)

            try:
                # Run async retrieval synchronously
                loop = asyncio.new_event_loop()
                ctx = loop.run_until_complete(
                    self.knowledge_graph.retrieve(agent.tradition, question)
                )
                loop.close()

                context_text = ctx.to_prompt()
                if context_text:
                    self._source_contexts[agent.tradition] = context_text
                    if self.verbose:
                        print(f"  Retrieved {len(ctx.passages)} passages for {agent.tradition}")
            except Exception as e:
                if self.verbose:
                    print(f"  Could not retrieve context for {agent.tradition}: {e}")

        if self.verbose:
            print()

    def _run_ideal_pass(self, scenario: str, question: str, agents: list[Agent]) -> list[dict]:
        """Ask each agent's tradition what it recommends — pure reasoning, no psychology."""
        positions = []
        seen_traditions = set()

        for agent in agents:
            # Deduplicate — if two agents share a tradition, only query it once
            if agent.tradition in seen_traditions:
                # Find the existing position for this tradition
                existing = next(p for p in positions if p["tradition"] == agent.tradition)
                positions.append({
                    "name": agent.name,
                    "tradition": agent.tradition,
                    "ideal_position": existing["ideal_position"],
                })
                continue

            seen_traditions.add(agent.tradition)
            source_ctx = self._source_contexts.get(agent.tradition, "")
            prompt = agent.ideal_prompt(scenario, question, source_context=source_ctx)

            if self.verbose:
                print(f"  Querying {agent.tradition} tradition...")

            response = self.llm_call(prompt)
            positions.append({
                "name": agent.name,
                "tradition": agent.tradition,
                "ideal_position": response.strip(),
            })

            if self.verbose:
                print(f"    {agent.tradition}: {response.strip()[:120]}...\n")

        return positions

    def _run_round(self, scenario: str, question: str, agents: list[Agent],
                   round_num: int) -> RoundResult:
        """Run one round of constrained deliberation."""
        if self.verbose:
            print(f"\n── Round {round_num} ──")

        # Collect positions from previous round (empty for round 1)
        previous_positions = []
        if round_num > 1:
            for agent in agents:
                if agent.memory.positions_taken:
                    last = agent.memory.positions_taken[-1]
                    previous_positions.append({
                        "name": agent.name,
                        "role": agent.role,
                        "position": last["position"],
                    })

        positions = []
        for agent in agents:
            # Other agents' positions (exclude self)
            other_positions = [p for p in previous_positions if p["name"] != agent.name]

            source_ctx = self._source_contexts.get(agent.tradition, "")
            prompt = agent.constrained_prompt(scenario, question, other_positions, round_num,
                                              source_context=source_ctx)

            if self.verbose:
                print(f"  {agent.name} deliberating...")

            response = self.llm_call(prompt)
            parsed = self._parse_response(response, agent.name)
            positions.append(parsed)

            # Update agent memory
            old_position = None
            if agent.memory.positions_taken:
                old_position = agent.memory.positions_taken[-1]["position"]

            agent.memory.record_position(round_num, parsed["position"], parsed["confidence"])

            if parsed["moved_by"] and parsed["moved_by"].lower() != "none":
                agent.memory.update_alliance(parsed["moved_by"], 0.2)

            if self.verbose:
                print(f"    Position: {parsed['position'][:100]}")
                print(f"    Confidence: {parsed['confidence']}")
                if parsed["private_thought"]:
                    print(f"    [Private: {parsed['private_thought'][:100]}]")
                print()

        # Detect shifts
        shifts = self._detect_shifts(agents, positions, round_num)

        # Record arguments heard
        for agent in agents:
            for p in positions:
                if p["name"] != agent.name:
                    persuasive = p["name"] == agent.memory.positions_taken[-1].get("moved_by", "")
                    agent.memory.record_argument(round_num, p["name"], p["argument"][:200], persuasive)

        # Detect tensions
        tensions = self._detect_tensions(positions)

        # Record observations for self-improvement system
        if self.track_observations:
            self._record_observations(agents, positions, round_num, scenario, question)

        return RoundResult(
            round_num=round_num,
            positions=positions,
            shifts=shifts,
            alliances_formed=[],
            tensions=tensions,
        )

    def _record_observations(self, agents: list[Agent], positions: list[dict],
                              round_num: int, scenario: str, question: str):
        """Record observations for each agent for the self-improvement system."""
        try:
            from observe import Observation, save_observation

            scenario_hash = hashlib.md5(
                (scenario + question).encode()
            ).hexdigest()[:8]

            for agent, pos in zip(agents, positions):
                # Find the corresponding ideal position for this tradition
                ideal_pos = None
                for ip in getattr(self, '_ideal_positions_cache', []):
                    if ip.get("tradition") == agent.tradition:
                        ideal_pos = ip.get("ideal_position", "")
                        break

                # Estimate consistency: simple heuristic based on confidence
                # In a full system, the LLM would assess this
                consistency = pos.get("confidence", 0.5)

                # Check if agent deviated (simplified: if moved by another)
                moved = pos.get("moved_by", "none")
                deviated = moved and moved.lower() != "none"

                obs = Observation(
                    tradition=agent.tradition,
                    agent_name=agent.name,
                    scenario_hash=scenario_hash,
                    round_num=round_num,
                    timestamp=datetime.now().isoformat(),
                    position_taken=pos.get("position", ""),
                    argument_given=pos.get("argument", "")[:500],
                    confidence=pos.get("confidence", 0.5),
                    private_thought=pos.get("private_thought", ""),
                    deviated_from_ideal=deviated,
                    deviation_description=f"Moved by {moved}" if deviated else "",
                    moved_by_other=moved if deviated else None,
                    consistency_with_tradition=consistency,
                )
                save_observation(obs)

        except ImportError:
            pass  # observe.py not available — skip silently
        except Exception as e:
            if self.verbose:
                print(f"  [observe] Warning: Could not record observation: {e}")

    def _parse_response(self, response: str, agent_name: str) -> dict:
        """Parse structured response from agent."""
        result = {
            "name": agent_name,
            "position": "",
            "argument": "",
            "confidence": 0.5,
            "moved_by": None,
            "private_thought": "",
        }

        # Try to parse structured fields
        sections = {
            "POSITION": "position",
            "ARGUMENT": "argument",
            "CONFIDENCE": "confidence",
            "MOVED_BY": "moved_by",
            "PRIVATE_THOUGHT": "private_thought",
        }

        for label, key in sections.items():
            pattern = rf"{label}:\s*(.*?)(?=\n(?:POSITION|ARGUMENT|CONFIDENCE|MOVED_BY|PRIVATE_THOUGHT):|$)"
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if key == "confidence":
                    try:
                        result[key] = float(re.search(r"[\d.]+", value).group())
                    except (ValueError, AttributeError):
                        result[key] = 0.5
                else:
                    result[key] = value

        # Fallback: if parsing fails, use the whole response
        if not result["position"] and not result["argument"]:
            result["argument"] = response.strip()
            result["position"] = response.strip()[:200]

        return result

    def _detect_shifts(self, agents: list[Agent], current_positions: list[dict],
                       round_num: int) -> list[dict]:
        """Detect which agents changed position this round."""
        shifts = []
        for agent in agents:
            history = agent.memory.positions_taken
            if len(history) >= 2:
                old = history[-2]["position"]
                new = history[-1]["position"]
                if old != new:
                    shifts.append({
                        "name": agent.name,
                        "old_position": old[:100],
                        "new_position": new[:100],
                        "round": round_num,
                    })
        return shifts

    def _detect_tensions(self, positions: list[dict]) -> list[str]:
        """Identify fundamental disagreements between agents."""
        tensions = []
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                if p1["confidence"] > 0.7 and p2["confidence"] > 0.7:
                    tensions.append(
                        f"{p1['name']} and {p2['name']} both hold strong, "
                        f"potentially opposing positions."
                    )
        return tensions

    def _analyze_gap(self, ideal: list[dict], actual: list[dict],
                     agents: list[Agent]) -> str:
        """Generate the gap analysis — the distance between ideal and actual reasoning."""
        prompt = f"""You are an analyst examining a deliberation simulation.

Below are two sets of results for the same question:

IDEAL POSITIONS (what each intellectual tradition recommends through pure reasoning):
{json.dumps(ideal, indent=2, default=str)}

ACTUAL POSITIONS (where agents ended up after deliberating as real people with biases, incentives, and psychological constraints):
{json.dumps(actual, indent=2, default=str)}

AGENT PSYCHOLOGICAL PROFILES:
{chr(10).join(f"- {a.name}: {a.psychology.describe()}" for a in agents)}

Analyze the GAP between ideal and actual. This is the most important part of the analysis. Specifically:

1. Where did agents deviate from what their tradition recommends? Why?
2. Which psychological constraints had the most distorting effect?
3. Which agents held closest to their tradition's ideal? What gave them the courage or the luxury to do so?
4. What does the gap reveal about the question itself — is it the kind of question where human psychology reliably warps reasoning, or did the agents stay relatively close to the ideal?
5. What would have to change in the incentive structure for the actual outcome to match the ideal?

Write 3-4 paragraphs. Be specific. Name names. This is not a summary — it is a diagnosis."""

        return self.llm_call(prompt)

    def _generate_narrative(self, scenario: str, question: str,
                            ideal: list[dict], rounds: list[RoundResult],
                            final: list[dict], agents: list[Agent]) -> str:
        """Generate a human-readable narrative of what happened."""
        round_summaries = []
        for r in rounds:
            positions_text = "\n".join(
                f"  {p['name']}: {p['position'][:150]} (confidence: {p['confidence']})"
                for p in r.positions
            )
            shifts_text = "\n".join(
                f"  {s['name']} shifted from '{s['old_position'][:80]}' to '{s['new_position'][:80]}'"
                for s in r.shifts
            ) if r.shifts else "  No position shifts."

            round_summaries.append(
                f"ROUND {r.round_num}:\nPositions:\n{positions_text}\nShifts:\n{shifts_text}"
            )

        prompt = f"""You are writing a narrative account of a deliberation simulation.

SCENARIO: {scenario}
QUESTION: {question}

WHAT PURE REASONING RECOMMENDED:
{json.dumps(ideal, indent=2, default=str)}

WHAT ACTUALLY HAPPENED (round by round):
{chr(10).join(round_summaries)}

WHERE THEY ENDED UP:
{json.dumps(final, indent=2, default=str)}

Write a narrative account of this deliberation — not a summary, but a story. Tell it like a historian watching a room full of people argue. Notice who spoke first and why. Notice who changed their mind and what it cost them. Notice the private thoughts they would never say aloud. Notice the moment the room shifted, if it did.

The narrative should make the reader feel they were in the room. End by posing the question the deliberation leaves unanswered — the thing the room could not resolve no matter how many rounds it ran.

Write 4-6 paragraphs."""

        return self.llm_call(prompt)
