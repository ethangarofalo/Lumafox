"""
Report generator.

Produces a structured output showing:
  1. What each tradition recommends (ideal reasoning)
  2. What each agent actually did (constrained deliberation)
  3. The gap between the two — and what it reveals
  4. A narrative account of the deliberation
"""

import json
from datetime import datetime
from deliberation import DeliberationResult


def generate_report(result: DeliberationResult) -> str:
    """Generate a full text report from deliberation results."""
    lines = []
    divider = "═" * 70

    lines.append(divider)
    lines.append("  POLIS — Deliberation Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(divider)

    # ── Scenario ──
    lines.append("\n┌─ SCENARIO ─────────────────────────────────────────────┐")
    lines.append(f"\n{result.scenario.strip()}\n")
    lines.append(f"QUESTION: {result.question.strip()}")
    lines.append("\n└────────────────────────────────────────────────────────┘")

    # ── Section 1: Ideal Reasoning ──
    lines.append(f"\n{'─'*70}")
    lines.append("  SECTION 1: IDEAL REASONING")
    lines.append("  What each intellectual tradition recommends, unconstrained")
    lines.append(f"{'─'*70}\n")

    for pos in result.ideal_positions:
        lines.append(f"  ▸ {pos.get('tradition', 'Unknown').upper()} TRADITION")
        lines.append(f"    (represented by {pos['name']})\n")
        # Wrap the text
        text = pos.get("ideal_position", "No position recorded.")
        for paragraph in text.split("\n\n"):
            lines.append(f"    {paragraph.strip()}\n")

    # ── Section 2: Constrained Deliberation ──
    lines.append(f"\n{'─'*70}")
    lines.append("  SECTION 2: CONSTRAINED DELIBERATION")
    lines.append("  What actually happened when real people argued")
    lines.append(f"{'─'*70}\n")

    for i, round_result in enumerate(result.rounds):
        lines.append(f"  ── Round {round_result.round_num} ──\n")

        for pos in round_result.positions:
            conf_bar = "█" * int(pos["confidence"] * 10) + "░" * (10 - int(pos["confidence"] * 10))
            lines.append(f"    {pos['name']}")
            lines.append(f"    Position: {pos['position'][:200]}")
            lines.append(f"    Confidence: [{conf_bar}] {pos['confidence']:.1f}")
            if pos.get("moved_by") and pos["moved_by"].lower() != "none":
                lines.append(f"    Moved by: {pos['moved_by']}")
            if pos.get("private_thought"):
                lines.append(f"    💭 Private: {pos['private_thought']}")
            lines.append("")

        if round_result.shifts:
            lines.append("    ⚡ SHIFTS:")
            for shift in round_result.shifts:
                lines.append(f"      {shift['name']}: \"{shift['old_position'][:60]}\" → \"{shift['new_position'][:60]}\"")
            lines.append("")

        if round_result.tensions:
            lines.append("    ⚔ TENSIONS:")
            for tension in round_result.tensions:
                lines.append(f"      {tension}")
            lines.append("")

    # ── Section 3: Final Positions ──
    lines.append(f"\n{'─'*70}")
    lines.append("  SECTION 3: FINAL POSITIONS")
    lines.append(f"{'─'*70}\n")

    for pos in result.final_positions:
        lines.append(f"  {pos['name']} ({pos['tradition']})")
        lines.append(f"  Position: {pos['position'][:200]}")
        lines.append(f"  Confidence: {pos['confidence']:.1f}")
        lines.append("")

    # ── Section 4: The Gap ──
    lines.append(f"\n{'─'*70}")
    lines.append("  SECTION 4: THE GAP")
    lines.append("  The distance between what reason recommends and what people do")
    lines.append(f"{'─'*70}\n")

    for paragraph in result.gap_analysis.split("\n\n"):
        lines.append(f"  {paragraph.strip()}\n")

    # ── Section 5: Narrative ──
    lines.append(f"\n{'─'*70}")
    lines.append("  SECTION 5: WHAT HAPPENED IN THE ROOM")
    lines.append(f"{'─'*70}\n")

    for paragraph in result.narrative.split("\n\n"):
        lines.append(f"  {paragraph.strip()}\n")

    lines.append(divider)
    lines.append("  END OF REPORT")
    lines.append(divider)

    return "\n".join(lines)


def save_report(result: DeliberationResult, path: str):
    """Save report to a text file."""
    report_text = generate_report(result)
    with open(path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {path}")
    return report_text


def save_json(result: DeliberationResult, path: str):
    """Save raw deliberation data as JSON for further analysis."""
    data = {
        "scenario": result.scenario,
        "question": result.question,
        "ideal_positions": result.ideal_positions,
        "rounds": [
            {
                "round_num": r.round_num,
                "positions": r.positions,
                "shifts": r.shifts,
                "tensions": r.tensions,
            }
            for r in result.rounds
        ],
        "final_positions": result.final_positions,
        "gap_analysis": result.gap_analysis,
        "narrative": result.narrative,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"JSON data saved to: {path}")
