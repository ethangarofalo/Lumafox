"""
Observation, inspection, and self-improvement system for traditions.

Implements the cognee-skills loop adapted for Polis:
  OBSERVE  → Record what happened when a tradition reasoned
  INSPECT  → Analyze patterns of failure and weakness
  AMEND    → Propose improvements to the tradition's instructions
  EVALUATE → Test whether the amendment actually improved things

Every deliberation run generates observations. Over time, traditions
that reason poorly accumulate evidence of their failures, and the
system can propose targeted amendments — not wild rewrites, but
specific patches grounded in what actually went wrong.

The human (the teacher) always has final say. Amendments can be
auto-proposed but require approval before they take effect.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable


OBSERVATIONS_DIR = Path(__file__).parent / "traditions" / "observations"
AMENDMENTS_DIR = Path(__file__).parent / "traditions" / "amendments"

# Ensure directories exist
OBSERVATIONS_DIR.mkdir(parents=True, exist_ok=True)
AMENDMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# 1. OBSERVE — Record what happened
# ──────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """A single observation from a deliberation run."""
    tradition: str
    agent_name: str
    scenario_hash: str              # Short identifier for the scenario
    round_num: int
    timestamp: str

    # What happened
    position_taken: str
    argument_given: str
    confidence: float
    private_thought: str

    # Quality signals
    deviated_from_ideal: bool       # Did the agent deviate from its tradition's ideal?
    deviation_description: str      # How it deviated
    moved_by_other: Optional[str]   # Was it swayed? By whom?
    consistency_with_tradition: float  # 0.0 = completely off-tradition, 1.0 = perfectly on

    # Feedback (can be added later)
    teacher_rating: Optional[float] = None   # 0.0 to 1.0, set by human
    teacher_note: Optional[str] = None
    auto_quality_score: Optional[float] = None  # LLM-assessed quality


def save_observation(obs: Observation):
    """Append observation to the tradition's observation log."""
    path = OBSERVATIONS_DIR / f"{obs.tradition}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(asdict(obs)) + "\n")


def load_observations(tradition: str) -> list[dict]:
    """Load all observations for a tradition."""
    path = OBSERVATIONS_DIR / f"{tradition}.jsonl"
    if not path.exists():
        return []
    observations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                observations.append(json.loads(line))
    return observations


# ──────────────────────────────────────────────────────────────────
# 2. INSPECT — Analyze patterns of weakness
# ──────────────────────────────────────────────────────────────────

@dataclass
class InspectionReport:
    """Analysis of a tradition's performance across deliberation runs."""
    tradition: str
    total_observations: int
    avg_consistency: float
    avg_confidence: float
    deviation_rate: float           # How often does the agent deviate from ideal?
    common_deviations: list[str]    # Recurring patterns of failure
    weak_topics: list[str]          # Topics where the tradition underperforms
    strength_topics: list[str]      # Topics where it performs well
    teacher_avg_rating: Optional[float]
    recommendations: list[str]      # Specific suggestions for amendment


def inspect_tradition(tradition: str, llm_call: Optional[Callable] = None) -> InspectionReport:
    """
    Analyze a tradition's performance and identify patterns of weakness.

    If an llm_call is provided, uses it for deeper pattern analysis.
    Otherwise, does statistical analysis only.
    """
    observations = load_observations(tradition)

    if not observations:
        return InspectionReport(
            tradition=tradition,
            total_observations=0,
            avg_consistency=0.0,
            avg_confidence=0.0,
            deviation_rate=0.0,
            common_deviations=[],
            weak_topics=[],
            strength_topics=[],
            teacher_avg_rating=None,
            recommendations=["No observations yet. Run some deliberations first."],
        )

    # Statistical analysis
    consistencies = [o.get("consistency_with_tradition", 0.5) for o in observations]
    confidences = [o.get("confidence", 0.5) for o in observations]
    deviations = [o for o in observations if o.get("deviated_from_ideal", False)]
    teacher_ratings = [o["teacher_rating"] for o in observations if o.get("teacher_rating") is not None]

    avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0.0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    deviation_rate = len(deviations) / len(observations) if observations else 0.0
    teacher_avg = sum(teacher_ratings) / len(teacher_ratings) if teacher_ratings else None

    # Collect deviation descriptions
    deviation_descriptions = [d.get("deviation_description", "") for d in deviations if d.get("deviation_description")]

    # LLM-powered deep inspection
    common_deviations = []
    weak_topics = []
    strength_topics = []
    recommendations = []

    if llm_call and len(observations) >= 3:
        # Build a summary for the LLM to analyze
        obs_summary = json.dumps(observations[-20:], indent=2, default=str)  # Last 20

        analysis = llm_call(f"""Analyze this tradition agent's performance across multiple deliberation runs.

TRADITION: {tradition}
TOTAL RUNS: {len(observations)}
AVERAGE CONSISTENCY WITH TRADITION: {avg_consistency:.2f}
DEVIATION RATE: {deviation_rate:.2%}
TEACHER AVERAGE RATING: {teacher_avg if teacher_avg else 'Not yet rated'}

RECENT OBSERVATIONS:
{obs_summary}

Identify:
1. COMMON DEVIATIONS: Recurring patterns where the agent fails to reason properly within its tradition (list 2-4)
2. WEAK TOPICS: Specific topics or question types where the tradition agent underperforms (list 2-3)
3. STRONG TOPICS: Topics where it performs well (list 2-3)
4. RECOMMENDATIONS: Specific, actionable changes to the tradition's instructions that would address the weaknesses (list 2-4). Be precise — suggest exact phrases to add, remove, or modify.

Respond as JSON with keys: common_deviations, weak_topics, strength_topics, recommendations. Each value is a list of strings.""")

        try:
            parsed = json.loads(analysis)
            common_deviations = parsed.get("common_deviations", [])
            weak_topics = parsed.get("weak_topics", [])
            strength_topics = parsed.get("strength_topics", [])
            recommendations = parsed.get("recommendations", [])
        except (json.JSONDecodeError, KeyError):
            recommendations = [f"LLM analysis available but unparseable. Raw: {analysis[:300]}"]

    elif deviation_descriptions:
        common_deviations = deviation_descriptions[:5]
        recommendations = [
            f"Deviation rate is {deviation_rate:.0%}. Consider strengthening the tradition's core commitments.",
            "Review the deviation descriptions above for patterns.",
        ]

    return InspectionReport(
        tradition=tradition,
        total_observations=len(observations),
        avg_consistency=avg_consistency,
        avg_confidence=avg_confidence,
        deviation_rate=deviation_rate,
        common_deviations=common_deviations,
        weak_topics=weak_topics,
        strength_topics=strength_topics,
        teacher_avg_rating=teacher_avg,
        recommendations=recommendations,
    )


# ──────────────────────────────────────────────────────────────────
# 3. AMEND — Propose improvements
# ──────────────────────────────────────────────────────────────────

@dataclass
class Amendment:
    """A proposed change to a tradition's instructions."""
    tradition: str
    timestamp: str
    trigger: str                    # What prompted this amendment
    description: str                # What the amendment does
    changes: list[dict]             # [{type: "add_principle"|"add_correction"|..., content: str}]
    evidence: list[str]             # Observations that support this amendment
    status: str = "proposed"        # proposed | approved | rejected | rolled_back
    evaluation_score: Optional[float] = None  # Post-amendment performance


def propose_amendment(tradition: str, report: InspectionReport,
                      llm_call: Callable) -> Amendment:
    """
    Propose a specific amendment to a tradition based on inspection results.

    The amendment is a set of changes that would be applied as refinements
    (same system as teach.py). The teacher must approve before they take effect.
    """
    prompt = f"""You are proposing improvements to a philosophical tradition agent in a deliberation system.

TRADITION: {tradition}
INSPECTION RESULTS:
  Total observations: {report.total_observations}
  Average consistency with tradition: {report.avg_consistency:.2f}
  Deviation rate: {report.deviation_rate:.2%}
  Common deviations: {json.dumps(report.common_deviations)}
  Weak topics: {json.dumps(report.weak_topics)}
  Recommendations: {json.dumps(report.recommendations)}

Propose a targeted amendment. This is NOT a rewrite — it is a patch. Think of it like a
code review: identify the specific instructions that need to change and propose the minimal
edit that would fix the identified weaknesses.

Your amendment should be a list of changes, where each change is one of:
  - add_principle: A new core principle to add
  - add_correction: A correction to an existing instruction
  - add_voice_note: A note about how the tradition should speak
  - add_anti_pattern: Something the tradition should never do
  - add_example: An example of correct reasoning

Respond as JSON:
{{
    "description": "One sentence describing what this amendment does",
    "changes": [
        {{"type": "add_correction", "content": "The specific correction text"}},
        {{"type": "add_principle", "content": "The principle text"}},
        ...
    ],
    "evidence": ["Brief description of what observation prompted each change"]
}}"""

    response = llm_call(prompt)
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = {
            "description": "Amendment proposal (unparseable — review manually)",
            "changes": [],
            "evidence": [response[:300]],
        }

    amendment = Amendment(
        tradition=tradition,
        timestamp=datetime.now().isoformat(),
        trigger=f"Inspection: {report.deviation_rate:.0%} deviation rate, {report.total_observations} observations",
        description=parsed.get("description", "Proposed amendment"),
        changes=parsed.get("changes", []),
        evidence=parsed.get("evidence", []),
        status="proposed",
    )

    # Save the proposal
    path = AMENDMENTS_DIR / f"{tradition}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(asdict(amendment)) + "\n")

    return amendment


def apply_amendment(tradition: str, amendment: Amendment):
    """
    Apply an approved amendment by writing its changes as refinements.

    This feeds into the same refinement system that teach.py uses,
    so amendments and manual teaching coexist in the same pipeline.
    """
    from teach import save_refinement

    for change in amendment.changes:
        rtype = change["type"].replace("add_", "")
        # Map amendment types to refinement types
        type_map = {
            "principle": "principle",
            "correction": "correction",
            "voice_note": "voice_note",
            "anti_pattern": "anti_pattern",
            "example": "example",
        }
        refinement_type = type_map.get(rtype, "correction")

        save_refinement(tradition, {
            "type": refinement_type,
            "content": change["content"],
            "context": f"[AUTO-AMENDMENT] {amendment.description}",
            "timestamp": datetime.now().isoformat(),
            "session": -1,  # -1 indicates auto-generated
        })

    # Update amendment status
    amendment.status = "approved"
    print(f"  Applied {len(amendment.changes)} changes to {tradition}")


# ──────────────────────────────────────────────────────────────────
# 4. EVALUATE — Did the amendment help?
# ──────────────────────────────────────────────────────────────────

def evaluate_amendment(tradition: str, amendment: Amendment,
                       pre_observations: list[dict],
                       post_observations: list[dict]) -> dict:
    """
    Compare tradition performance before and after an amendment.

    Returns a dict with before/after metrics and a verdict.
    """
    def avg_metric(obs_list, key):
        values = [o.get(key, 0.5) for o in obs_list if key in o]
        return sum(values) / len(values) if values else 0.0

    pre_consistency = avg_metric(pre_observations, "consistency_with_tradition")
    post_consistency = avg_metric(post_observations, "consistency_with_tradition")
    pre_deviations = sum(1 for o in pre_observations if o.get("deviated_from_ideal")) / max(len(pre_observations), 1)
    post_deviations = sum(1 for o in post_observations if o.get("deviated_from_ideal")) / max(len(post_observations), 1)

    improved = post_consistency > pre_consistency and post_deviations < pre_deviations
    verdict = "improved" if improved else "no_improvement"

    evaluation = {
        "tradition": tradition,
        "amendment_description": amendment.description,
        "pre_consistency": pre_consistency,
        "post_consistency": post_consistency,
        "pre_deviation_rate": pre_deviations,
        "post_deviation_rate": post_deviations,
        "verdict": verdict,
        "recommendation": "keep" if improved else "rollback",
    }

    amendment.evaluation_score = post_consistency
    return evaluation


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def print_inspection(report: InspectionReport):
    """Pretty-print an inspection report."""
    print(f"\n{'='*60}")
    print(f"  INSPECTION: {report.tradition.upper()} TRADITION")
    print(f"{'='*60}")
    print(f"  Observations: {report.total_observations}")
    print(f"  Avg consistency: {report.avg_consistency:.2f}")
    print(f"  Avg confidence: {report.avg_confidence:.2f}")
    print(f"  Deviation rate: {report.deviation_rate:.0%}")
    if report.teacher_avg_rating is not None:
        print(f"  Teacher avg rating: {report.teacher_avg_rating:.2f}")
    print()

    if report.common_deviations:
        print("  Common deviations:")
        for d in report.common_deviations:
            print(f"    - {d}")
        print()

    if report.weak_topics:
        print("  Weak topics:")
        for t in report.weak_topics:
            print(f"    - {t}")
        print()

    if report.strength_topics:
        print("  Strong topics:")
        for t in report.strength_topics:
            print(f"    + {t}")
        print()

    if report.recommendations:
        print("  Recommendations:")
        for r in report.recommendations:
            print(f"    → {r}")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="POLIS Observation & Self-Improvement System")
    sub = parser.add_subparsers(dest="command")

    inspect_cmd = sub.add_parser("inspect", help="Inspect a tradition's performance")
    inspect_cmd.add_argument("tradition", help="Tradition to inspect")
    inspect_cmd.add_argument("--llm", action="store_true", help="Use LLM for deeper analysis")

    amend_cmd = sub.add_parser("amend", help="Propose an amendment for a tradition")
    amend_cmd.add_argument("tradition", help="Tradition to amend")

    rate_cmd = sub.add_parser("rate", help="Rate a tradition's recent performance")
    rate_cmd.add_argument("tradition", help="Tradition to rate")
    rate_cmd.add_argument("rating", type=float, help="Rating 0.0-1.0")
    rate_cmd.add_argument("--note", type=str, default="", help="Optional note")

    args = parser.parse_args()

    if args.command == "inspect":
        llm_call = None
        if args.llm:
            from llm import make_claude_caller
            llm_call = make_claude_caller()
        report = inspect_tradition(args.tradition, llm_call=llm_call)
        print_inspection(report)

    elif args.command == "amend":
        from llm import make_claude_caller
        llm_call = make_claude_caller()
        report = inspect_tradition(args.tradition, llm_call=llm_call)
        print_inspection(report)

        if report.total_observations < 3:
            print("  Need at least 3 observations before proposing amendments.")
            return

        amendment = propose_amendment(args.tradition, report, llm_call)
        print(f"\n  Proposed amendment: {amendment.description}")
        print(f"  Changes: {len(amendment.changes)}")
        for c in amendment.changes:
            print(f"    [{c['type']}] {c['content'][:100]}")

        approval = input("\n  Apply this amendment? [y/n] > ").strip().lower()
        if approval == "y":
            apply_amendment(args.tradition, amendment)
            print("  Amendment applied.")
        else:
            print("  Amendment saved as proposed (not applied).")

    elif args.command == "rate":
        observations = load_observations(args.tradition)
        if not observations:
            print(f"  No observations for {args.tradition}")
            return
        # Rate the most recent observation
        last = observations[-1]
        last["teacher_rating"] = args.rating
        if args.note:
            last["teacher_note"] = args.note
        # Rewrite the file with the updated observation
        path = OBSERVATIONS_DIR / f"{args.tradition}.jsonl"
        with open(path, "w") as f:
            for obs in observations:
                f.write(json.dumps(obs) + "\n")
        print(f"  Rated {args.tradition}'s last observation: {args.rating}")


if __name__ == "__main__":
    main()
