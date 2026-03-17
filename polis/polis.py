#!/usr/bin/env python3
"""
POLIS — A Multi-Agent Deliberation Engine

Models the gap between ideal reasoning and actual human behavior
by running dual-layer simulations: what traditions recommend vs.
what psychologically-constrained agents actually do.

Usage:
    # Run the demo scenario with Claude
    python polis.py

    # Run with mock LLM (no API key needed)
    python polis.py --mock

    # Customize rounds
    python polis.py --rounds 3

    # Use a specific Claude model
    python polis.py --model claude-sonnet-4-20250514

    # Use OpenAI
    python polis.py --provider openai --model gpt-4o

    # Save report to file
    python polis.py --output report.txt

    # Save raw JSON data
    python polis.py --json data.json

    # Quiet mode (no live output, just the final report)
    python polis.py --quiet
"""

import argparse
import sys

from scenario import DEMO_SCENARIO, build_scenario, filter_agents
from deliberation import DeliberationEngine
from report import generate_report, save_report, save_json
from llm import make_claude_caller, make_openai_caller, make_mock_caller


def main():
    parser = argparse.ArgumentParser(
        description="POLIS — Multi-Agent Deliberation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python polis.py                          # Run demo with Claude
  python polis.py --mock                   # Test without API key
  python polis.py --rounds 3 --quiet       # 3 rounds, report only
  python polis.py --output report.txt      # Save report to file
        """,
    )

    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM (no API key needed, for testing)")
    parser.add_argument("--provider", choices=["claude", "openai"], default="claude",
                        help="LLM provider (default: claude)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: claude-sonnet-4-20250514 or gpt-4o)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of deliberation rounds (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to text file")
    parser.add_argument("--json", type=str, default=None,
                        help="Save raw data to JSON file")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress live output, print only final report")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Path to a custom scenario JSON file")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="LLM temperature (default: 0.8)")
    parser.add_argument("--knowledge", action="store_true",
                        help="Enable knowledge graph retrieval (requires source texts in traditions/sources/)")
    parser.add_argument("--no-observe", action="store_true",
                        help="Disable observation tracking for self-improvement")
    parser.add_argument("--agents", type=str, default=None,
                        help="Comma-separated list of agent names or traditions to include "
                             "(e.g., 'Socrates,Machiavelli' or 'socratic,locke'). "
                             "Filters the scenario to only these participants.")

    args = parser.parse_args()

    # ── Build LLM caller ──
    if args.mock:
        llm_call = make_mock_caller()
        if not args.quiet:
            print("Using mock LLM (no API calls)\n")
    elif args.provider == "claude":
        model = args.model or "claude-sonnet-4-20250514"
        llm_call = make_claude_caller(model=model, temperature=args.temperature)
        if not args.quiet:
            print(f"Using Claude: {model}\n")
    elif args.provider == "openai":
        model = args.model or "gpt-4o"
        llm_call = make_openai_caller(model=model, temperature=args.temperature)
        if not args.quiet:
            print(f"Using OpenAI: {model}\n")

    # ── Load scenario ──
    if args.scenario:
        import json
        with open(args.scenario) as f:
            config = json.load(f)
        scenario, question, agents = build_scenario(config)
    else:
        scenario, question, agents = build_scenario(DEMO_SCENARIO)

    # ── Filter agents (optional) ──
    if args.agents:
        agents = filter_agents(agents, args.agents)
        if not agents:
            print("Error: No agents matched the filter. Available agents:")
            _, _, all_agents = build_scenario(config if args.scenario else DEMO_SCENARIO)
            for a in all_agents:
                print(f"  - {a.name} ({a.tradition})")
            sys.exit(1)
        if not args.quiet:
            print(f"Filtered to {len(agents)} agent(s): {', '.join(a.name for a in agents)}\n")

    # ── Knowledge graph (optional) ──
    knowledge_graph = None
    if args.knowledge:
        try:
            import asyncio
            from knowledge import KnowledgeGraph
            knowledge_graph = KnowledgeGraph()
            asyncio.run(knowledge_graph.initialize())
            asyncio.run(knowledge_graph.ingest_all_traditions())
        except ImportError:
            if not args.quiet:
                print("knowledge.py not found. Running without source grounding.\n")
        except Exception as e:
            if not args.quiet:
                print(f"Knowledge graph init failed: {e}. Running without source grounding.\n")

    # ── Run deliberation ──
    engine = DeliberationEngine(
        llm_call=llm_call,
        rounds=args.rounds,
        verbose=not args.quiet,
        knowledge_graph=knowledge_graph,
        track_observations=not args.no_observe,
    )

    result = engine.run(scenario, question, agents)

    # ── Output ──
    report_text = generate_report(result)

    if args.quiet:
        print(report_text)

    if args.output:
        save_report(result, args.output)

    if args.json:
        save_json(result, args.json)

    if not args.output and not args.quiet:
        print("\n" + report_text)


if __name__ == "__main__":
    main()
