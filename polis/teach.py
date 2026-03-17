#!/usr/bin/env python3
"""
POLIS Teaching System

An interactive dialogue where you teach a tradition's agent how to think,
speak, and reason. Your corrections, examples, and calibrations are saved
as refinements that persist across sessions and feed into deliberations.

Modes:
  DIALOGUE  — The agent reasons about a question; you correct and refine
  EXAMPLE   — You provide examples of how the tradition actually speaks
  EXAMINE   — You ask the agent questions to test its understanding
  CORRECT   — You tell the agent what it got wrong and why

Every interaction is saved. The tradition gets sharper each session.

Usage:
    python teach.py aristotelian
    python teach.py utilitarian --model claude-sonnet-4-20250514
    python teach.py stoic              # Creates a new tradition if it doesn't exist
    python teach.py aristotelian --review   # Review all refinements so far
    python teach.py aristotelian --export   # Export the refined tradition as a single document
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from llm import make_claude_caller, make_mock_caller

TRADITIONS_DIR = Path(__file__).parent / "traditions"
REFINEMENTS_DIR = TRADITIONS_DIR / "refinements"


def load_tradition(name: str) -> str:
    """Load base tradition file."""
    path = TRADITIONS_DIR / f"{name}.md"
    if path.exists():
        return path.read_text()
    return ""


def load_refinements(name: str) -> list[dict]:
    """Load all refinements for a tradition."""
    path = REFINEMENTS_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    refinements = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                refinements.append(json.loads(line))
    return refinements


def save_refinement(name: str, refinement: dict):
    """Append a single refinement to the tradition's refinement log."""
    path = REFINEMENTS_DIR / f"{name}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(refinement) + "\n")


def build_refinement_context(refinements: list[dict]) -> str:
    """Build a prompt-ready summary of all refinements."""
    if not refinements:
        return ""

    sections = {
        "correction": [],
        "example": [],
        "principle": [],
        "voice_note": [],
        "anti_pattern": [],
    }

    for r in refinements:
        rtype = r.get("type", "correction")
        content = r.get("content", "")
        if rtype in sections:
            sections[rtype].append(content)
        else:
            sections["correction"].append(content)

    lines = ["\n\n## REFINEMENTS FROM THE TEACHER\n"]
    lines.append("The following corrections, examples, and principles have been")
    lines.append("provided by someone who has studied this tradition deeply.")
    lines.append("These take PRECEDENCE over the base description above.\n")

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

    return "\n".join(lines)


def export_tradition(name: str) -> str:
    """Export the full refined tradition as a single markdown document."""
    base = load_tradition(name)
    refinements = load_refinements(name)
    refinement_text = build_refinement_context(refinements)

    if not base and not refinements:
        return f"# {name.title()} Tradition\n\nNo content yet. Start teaching with: python teach.py {name}"

    return base + refinement_text


def review_refinements(name: str):
    """Print all refinements for review."""
    refinements = load_refinements(name)
    if not refinements:
        print(f"\nNo refinements yet for {name}. Start teaching!\n")
        return

    print(f"\n{'='*60}")
    print(f"  REFINEMENTS: {name.upper()} TRADITION")
    print(f"  {len(refinements)} refinements recorded")
    print(f"{'='*60}\n")

    for i, r in enumerate(refinements, 1):
        rtype = r.get("type", "unknown")
        content = r.get("content", "")
        timestamp = r.get("timestamp", "unknown")
        context = r.get("context", "")

        print(f"  [{i}] {rtype.upper()}")
        print(f"      {timestamp}")
        if context:
            print(f"      Context: {context[:80]}")
        print(f"      {content[:200]}")
        if len(content) > 200:
            print(f"      ...")
        print()


class TeachingSession:
    """Interactive teaching session with a tradition's agent."""

    def __init__(self, tradition_name: str, llm_call, verbose: bool = True):
        self.name = tradition_name
        self.llm_call = llm_call
        self.verbose = verbose
        self.base_text = load_tradition(tradition_name)
        self.refinements = load_refinements(tradition_name)
        self.conversation: list[dict] = []
        self.session_refinements: int = 0

    def get_full_tradition_text(self) -> str:
        """Base tradition + all refinements."""
        return self.base_text + build_refinement_context(self.refinements)

    def agent_respond(self, user_message: str, mode: str = "dialogue") -> str:
        """Get the tradition agent's response."""
        tradition_text = self.get_full_tradition_text()

        # Build conversation history for context
        history = ""
        if self.conversation:
            recent = self.conversation[-6:]  # Last 3 exchanges
            history = "\n\nRECENT CONVERSATION:\n"
            for msg in recent:
                role = "TEACHER" if msg["role"] == "user" else "AGENT"
                history += f"{role}: {msg['content'][:300]}\n"

        if mode == "dialogue":
            prompt = f"""You are an agent embodying the {self.name} tradition.

{tradition_text}

{history}

The teacher — someone who has studied this tradition deeply — is now engaging
you in dialogue. Respond as this tradition would ACTUALLY reason. Not a
textbook summary. Not a caricature. The way a genuine practitioner of this
tradition would think and speak if they were the smartest person in the room
who happened to hold these commitments.

Be specific. Be rigorous. If the tradition has internal tensions, acknowledge
them. If the tradition would disagree with common misreadings of itself,
say so. If the tradition has a characteristic way of phrasing things — a
rhythm, a set of key terms, a way of structuring arguments — use it.

TEACHER: {user_message}

Respond in 2-4 paragraphs."""

        elif mode == "examine":
            prompt = f"""You are being examined on your understanding of the {self.name} tradition.

{tradition_text}

{history}

A teacher who knows this tradition deeply is testing whether you truly
understand it — not the Wikipedia version, but the real thing. Answer with
the precision and depth the teacher expects. If you're uncertain about
something, say so rather than bluffing. Show your reasoning.

TEACHER'S QUESTION: {user_message}

Answer carefully."""

        elif mode == "demonstrate":
            prompt = f"""You are embodying the {self.name} tradition.

{tradition_text}

{history}

The teacher wants you to demonstrate how this tradition would address the
following topic. Speak AS a practitioner of this tradition — not about it,
but from within it. Use the tradition's characteristic language, its natural
mode of argument, its particular way of seeing.

TOPIC: {user_message}

Speak in 2-3 paragraphs, in the tradition's own voice."""

        else:
            prompt = f"""You are an agent embodying the {self.name} tradition.

{tradition_text}

{history}

TEACHER: {user_message}

Respond thoughtfully."""

        response = self.llm_call(prompt)
        return response.strip()

    def add_refinement(self, rtype: str, content: str, context: str = ""):
        """Save a refinement and reload."""
        refinement = {
            "type": rtype,
            "content": content,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "session": self.session_refinements,
        }
        save_refinement(self.name, refinement)
        self.refinements.append(refinement)
        self.session_refinements += 1

    def run(self):
        """Run the interactive teaching session."""
        print(f"\n{'='*60}")
        print(f"  POLIS Teaching Session")
        print(f"  Tradition: {self.name.upper()}")
        existing = len(self.refinements)
        if existing:
            print(f"  Existing refinements: {existing}")
        else:
            print(f"  Starting fresh — no refinements yet")
        print(f"{'='*60}")
        print()
        print("  Commands:")
        print("    /dialogue  [text]  — Talk with the agent (default mode)")
        print("    /examine   [text]  — Test the agent's understanding")
        print("    /demo      [text]  — Agent demonstrates reasoning on a topic")
        print("    /correct   [text]  — Tell the agent what it got wrong")
        print("    /example   [text]  — Provide an example of how the tradition speaks")
        print("    /principle [text]  — Teach a core principle")
        print("    /voice     [text]  — Note about tone, rhythm, or characteristic phrasing")
        print("    /never     [text]  — Something this tradition would NEVER say")
        print("    /review            — Review all refinements")
        print("    /export            — Export the refined tradition document")
        print("    /inspect           — Inspect tradition performance from observations")
        print("    /amend             — Propose an amendment based on inspection")
        print("    /status            — Show session stats")
        print("    /quit              — End session")
        print()
        print("  Or just type — it defaults to dialogue mode.")
        print()

        mode = "dialogue"

        while True:
            try:
                user_input = input(f"  [{self.name}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Session ended.")
                break

            if not user_input:
                continue

            # ── Command parsing ──
            if user_input.startswith("/quit") or user_input.startswith("/exit"):
                print(f"\n  Session complete. {self.session_refinements} refinements saved.")
                break

            elif user_input.startswith("/review"):
                review_refinements(self.name)
                continue

            elif user_input.startswith("/export"):
                exported = export_tradition(self.name)
                export_path = TRADITIONS_DIR / f"{self.name}_refined.md"
                with open(export_path, "w") as f:
                    f.write(exported)
                print(f"\n  Exported to: {export_path}\n")
                continue

            elif user_input.startswith("/inspect"):
                try:
                    from observe import inspect_tradition, print_inspection
                    report = inspect_tradition(self.name, llm_call=self.llm_call)
                    print_inspection(report)
                except ImportError:
                    print("  observe.py not found. Run some deliberations first.\n")
                continue

            elif user_input.startswith("/amend"):
                try:
                    from observe import (inspect_tradition, print_inspection,
                                         propose_amendment, apply_amendment)
                    report = inspect_tradition(self.name, llm_call=self.llm_call)
                    print_inspection(report)

                    if report.total_observations < 3:
                        print("  Need at least 3 observations before proposing amendments.\n")
                        continue

                    amendment = propose_amendment(self.name, report, self.llm_call)
                    print(f"\n  Proposed amendment: {amendment.description}")
                    print(f"  Changes: {len(amendment.changes)}")
                    for c in amendment.changes:
                        print(f"    [{c['type']}] {c['content'][:100]}")

                    approval = input("\n  Apply this amendment? [y/n] > ").strip().lower()
                    if approval == "y":
                        apply_amendment(self.name, amendment)
                        # Reload refinements
                        self.refinements = load_refinements(self.name)
                        print("  Amendment applied and refinements reloaded.\n")
                    else:
                        print("  Amendment saved as proposed (not applied).\n")
                except ImportError:
                    print("  observe.py not found.\n")
                continue

            elif user_input.startswith("/status"):
                print(f"\n  Tradition: {self.name}")
                print(f"  Total refinements: {len(self.refinements)}")
                print(f"  This session: {self.session_refinements}")
                print(f"  Conversation turns: {len(self.conversation)}")
                print()
                continue

            elif user_input.startswith("/correct"):
                text = user_input[len("/correct"):].strip()
                if not text:
                    text = input("  What did it get wrong? > ").strip()
                if text:
                    # Get the last agent response as context
                    last_response = ""
                    for msg in reversed(self.conversation):
                        if msg["role"] == "agent":
                            last_response = msg["content"][:200]
                            break
                    self.add_refinement("correction", text, context=last_response)
                    print(f"  ✓ Correction saved.\n")

                    # Let the agent acknowledge and learn
                    response = self.agent_respond(
                        f"I'm correcting you: {text}. Acknowledge this correction and "
                        f"explain how it changes your understanding.",
                        mode="dialogue"
                    )
                    print(f"\n  Agent: {response}\n")
                    self.conversation.append({"role": "user", "content": f"[CORRECTION] {text}"})
                    self.conversation.append({"role": "agent", "content": response})
                continue

            elif user_input.startswith("/example"):
                text = user_input[len("/example"):].strip()
                if not text:
                    print("  Paste an example of how this tradition actually speaks.")
                    print("  (Can be a quote, a paraphrase, or a description of the style)")
                    text = input("  Example > ").strip()
                if text:
                    self.add_refinement("example", text)
                    print(f"  ✓ Example saved.\n")
                continue

            elif user_input.startswith("/principle"):
                text = user_input[len("/principle"):].strip()
                if not text:
                    text = input("  State the principle > ").strip()
                if text:
                    self.add_refinement("principle", text)
                    print(f"  ✓ Principle saved.\n")
                continue

            elif user_input.startswith("/voice"):
                text = user_input[len("/voice"):].strip()
                if not text:
                    text = input("  Describe the voice/tone > ").strip()
                if text:
                    self.add_refinement("voice_note", text)
                    print(f"  ✓ Voice note saved.\n")
                continue

            elif user_input.startswith("/never"):
                text = user_input[len("/never"):].strip()
                if not text:
                    text = input("  What would this tradition never say? > ").strip()
                if text:
                    self.add_refinement("anti_pattern", text)
                    print(f"  ✓ Anti-pattern saved.\n")
                continue

            elif user_input.startswith("/examine"):
                text = user_input[len("/examine"):].strip()
                if text:
                    user_input = text
                    mode = "examine"
                else:
                    continue

            elif user_input.startswith("/demo"):
                text = user_input[len("/demo"):].strip()
                if text:
                    user_input = text
                    mode = "demonstrate"
                else:
                    continue

            elif user_input.startswith("/dialogue"):
                text = user_input[len("/dialogue"):].strip()
                if text:
                    user_input = text
                mode = "dialogue"
                if not text:
                    continue

            # ── Agent interaction ──
            self.conversation.append({"role": "user", "content": user_input})

            response = self.agent_respond(user_input, mode=mode)
            self.conversation.append({"role": "agent", "content": response})

            print(f"\n  {response}\n")

            # Reset mode to default after use
            mode = "dialogue"


def create_new_tradition(name: str):
    """Create a skeleton tradition file for a new tradition."""
    path = TRADITIONS_DIR / f"{name}.md"
    if path.exists():
        return

    skeleton = f"""# {name.title()} Tradition

You reason from the {name} tradition. Your framework:

## Core Commitments
[To be refined through teaching sessions]

## How You Reason
[To be refined through teaching sessions]

## Characteristic Concerns
[To be refined through teaching sessions]
"""
    with open(path, "w") as f:
        f.write(skeleton)
    print(f"  Created new tradition: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Teach and refine a POLIS tradition agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tradition", help="Name of the tradition to teach (e.g., aristotelian, stoic)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM for testing")
    parser.add_argument("--review", action="store_true", help="Review all refinements and exit")
    parser.add_argument("--export", action="store_true", help="Export refined tradition and exit")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")

    args = parser.parse_args()
    tradition_name = args.tradition.lower().replace(" ", "_")

    # Ensure tradition exists
    if not (TRADITIONS_DIR / f"{tradition_name}.md").exists():
        print(f"\n  Tradition '{tradition_name}' doesn't exist yet.")
        create = input(f"  Create it? [y/n] > ").strip().lower()
        if create == "y":
            create_new_tradition(tradition_name)
        else:
            print("  Exiting.")
            return

    # Review mode
    if args.review:
        review_refinements(tradition_name)
        return

    # Export mode
    if args.export:
        exported = export_tradition(tradition_name)
        export_path = TRADITIONS_DIR / f"{tradition_name}_refined.md"
        with open(export_path, "w") as f:
            f.write(exported)
        print(f"\n  Exported to: {export_path}")
        print(f"\n{exported}")
        return

    # Build LLM caller
    if args.mock:
        llm_call = make_mock_caller()
    else:
        llm_call = make_claude_caller(model=args.model, temperature=args.temperature)

    # Run teaching session
    session = TeachingSession(tradition_name, llm_call)
    session.run()


if __name__ == "__main__":
    main()
