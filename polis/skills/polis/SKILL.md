---
name: polis
description: >
  POLIS multi-agent deliberation engine — project knowledge and development guide.
  Use this skill whenever working on the POLIS codebase: adding traditions, writing
  refinements, ingesting source texts, running deliberations, creating scenarios,
  or modifying any POLIS Python files. Also trigger when Ethan mentions "polis,"
  "deliberation engine," "traditions," "teach.py," "knowledge graph," "refinements,"
  "agents," "scenarios," or asks to work on philosophical tradition bots. If the task
  involves the POLIS project in any way, use this skill.
---

# POLIS — Project Knowledge

POLIS is a multi-agent deliberation engine that models the gap between ideal reasoning and actual human behavior. It runs dual-layer simulations: what intellectual traditions recommend (pure reasoning) versus what psychologically-constrained agents actually do (with biases, incentives, and ego). The gap between the two is where the insight lives.

## Project Location

The POLIS repo lives at `~/polis/` on Ethan's machine (macOS). The full path is `/Users/ethangarofalo/Polis/`. When working in Cowork, you'll need to mount this directory.

## Architecture

### Core Files

| File | Purpose |
|------|---------|
| `polis.py` | Main CLI entry point. Orchestrates deliberation runs. |
| `deliberation.py` | The deliberation engine. Runs ideal pass + constrained rounds. |
| `agent.py` | Agent class with dual layers: tradition (ideal) + psychology (constrained). |
| `teach.py` | Interactive teaching system for refining tradition voices. |
| `knowledge.py` | Knowledge graph: ingestion, chunking, retrieval. CLI for `ingest`, `ingest-all`, `search`, `xref`. |
| `llm.py` | LLM abstraction layer (Claude, OpenAI, mock). |
| `scenario.py` | Scenario loading and agent construction from JSON configs. |
| `report.py` | Report generation from deliberation results. |
| `observe.py` | Self-improvement observation tracking. |

### Directory Structure

```
polis/
├── traditions/
│   ├── socratic.md              # Base tradition files
│   ├── aristotelian.md
│   ├── jesus.md
│   ├── utilitarian.md
│   ├── libertarian.md
│   ├── communitarian.md
│   ├── pragmatist.md
│   ├── aristotelian_refined.md  # Example of refined output
│   ├── refinements/
│   │   ├── aristotelian.jsonl   # Refinement logs (JSONL)
│   │   ├── socratic.jsonl
│   │   └── jesus.jsonl
│   ├── sources/
│   │   ├── socratic/
│   │   │   ├── Laws.txt         # Strauss lecture transcripts
│   │   │   └── Republic.txt
│   │   ├── aristotelian/
│   │   │   └── sample.txt
│   │   └── jesus/
│   │       └── Jesus.txt        # Red-letter text, all four Gospels + epistles
│   ├── observations/            # Self-improvement tracking data
│   └── amendments/              # Proposed amendments from inspection
├── scenarios/                   # Custom scenario JSON files
└── [core .py files]
```

## How Deliberation Works

### The Two-Pass Architecture

1. **Ideal Pass** — Each tradition reasons about the question without psychological constraints. Pure philosophy. One query per unique tradition (deduplicated if two agents share a tradition).

2. **Constrained Pass** — Agents deliberate as real people across multiple rounds. Each agent has:
   - A tradition prompt (what they should think)
   - A psychology profile (biases, incentives, stubbornness, courage, vanity, empathy, crowd sensitivity)
   - A memory (positions taken, arguments heard, alliances, wounds)
   - Source context from the knowledge graph (if `--knowledge` flag is used)

3. **Gap Analysis** — The engine compares ideal vs. actual positions and diagnoses where psychology warped reasoning.

4. **Narrative** — A story of what happened in the room, told like a historian watching people argue.

### Running Deliberations

```bash
# Basic run with Claude
python3 polis.py

# With knowledge graph retrieval (feeds ingested source texts to agents)
python3 polis.py --knowledge

# Custom rounds, quiet mode
python3 polis.py --rounds 3 --quiet

# Mock LLM for testing (no API key)
python3 polis.py --mock

# Save outputs
python3 polis.py --output report.txt --json data.json
```

The `--knowledge` flag is the gate between ingested source material and the agents. Without it, agents reason only from their base tradition file + refinements.

## The Teaching System

`teach.py` is an interactive REPL for refining how a tradition thinks and speaks.

```bash
python3 teach.py socratic
python3 teach.py aristotelian --model claude-sonnet-4-20250514
```

### Commands

| Command | Purpose |
|---------|---------|
| `/dialogue [text]` | Talk with the agent (default mode) |
| `/examine [text]` | Test the agent's understanding |
| `/demo [text]` | Agent demonstrates reasoning on a topic |
| `/correct [text]` | Tell the agent what it got wrong |
| `/example [text]` | Provide an example of how the tradition speaks |
| `/principle [text]` | Teach a core principle |
| `/voice [text]` | Note about tone, rhythm, or characteristic phrasing |
| `/never [text]` | Something this tradition would NEVER say |
| `/review` | Review all refinements |
| `/export` | Export the refined tradition document |
| `/inspect` | Inspect tradition performance from observations |
| `/amend` | Propose an amendment based on inspection |
| `/status` | Show session stats |

### Refinement Format

Refinements are stored in `traditions/refinements/{tradition}.jsonl`. Each line is a JSON object:

```json
{"type": "principle", "content": "...", "context": "", "timestamp": "...", "session": 0}
{"type": "voice_note", "content": "...", "context": "", "timestamp": "...", "session": 1}
{"type": "anti_pattern", "content": "...", "context": "", "timestamp": "...", "session": 2}
{"type": "correction", "content": "...", "context": "...", "timestamp": "...", "session": 3}
{"type": "example", "content": "...", "context": "", "timestamp": "...", "session": 4}
```

The `type` field maps to teaching commands:
- `/principle` → `"principle"`
- `/voice` → `"voice_note"`
- `/never` → `"anti_pattern"`
- `/correct` → `"correction"`
- `/example` → `"example"`

When writing refinements directly to the JSONL file (bypassing teach.py), use this exact format. The `session` field is a counter; increment it for each entry. The `context` field can be empty string or can contain the agent's response that prompted the correction.

### Refined Tradition Files

The `/export` command (or the deliberation engine at runtime) combines:
1. Base tradition file (`traditions/{name}.md`)
2. All refinements from `traditions/refinements/{name}.jsonl`

Into a single prompt with sections: Core Principles (learned), Corrections, Examples, Voice and Tone Notes, What This Tradition Would NEVER Say. Refinements take PRECEDENCE over the base description.

You can also manually add a `## REFINEMENTS FROM THE TEACHER` section to the base .md file (as has been done with `jesus.md` and `aristotelian_refined.md`).

## The Knowledge Graph

`knowledge.py` handles source text ingestion and retrieval.

### Ingestion

```bash
# Ingest one tradition's sources
python3 knowledge.py ingest socratic

# Ingest all traditions
python3 knowledge.py ingest-all

# Search
python3 knowledge.py search socratic "what is justice?"

# Cross-reference across traditions
python3 knowledge.py xref justice
```

Source texts go in `traditions/sources/{tradition_name}/` as `.txt` files. The system chunks them (1000-char chunks, 200-char overlap) and stores them for retrieval. It supports two backends:
- **cognee** (full graph-RAG) — if installed
- **local fallback** (keyword-based) — default

### Current Source Status

| Tradition | Sources | Chunks |
|-----------|---------|--------|
| Socratic | Laws.txt (Strauss 1959), Republic.txt (Strauss 1957) | ~1,077 |
| Aristotelian | sample.txt | ~2 |
| Jesus | Jesus.txt (red-letter NT text) | Not yet ingested |

### How Retrieval Reaches Agents

The `DeliberationEngine` accepts an optional `knowledge_graph` parameter. When present, at the start of a run it calls `_retrieve_source_contexts()` which queries the knowledge graph once per unique tradition among the agents. Retrieved passages are injected into both `ideal_prompt()` and `constrained_prompt()` as `source_context`.

Agents are stateless about knowledge — they receive whatever the engine gives them for that particular question, scoped to their tradition.

## Current Refinement Status

### Traditions with refinements:
- **Aristotelian** — 10 refinements (principles, voice notes, corrections, examples, anti-patterns). Well-developed.
- **Socratic** — Refinements entered via teach.py (principles on incompleteness of philosophy, just vs. legal, philosophy-city tension; voice notes on inquiry mode, irony, concreteness; nevers on social science appeals, balancing, ideal city as program).
- **Jesus** — 15 refinements written directly to JSONL + embedded in jesus.md (principles on authority, personal address, Kingdom, reversal, absolute demand; voice on images, tone-shifting, counter-questions, brevity, grief; nevers on balance, utility, strategy, therapy-speak, "values" language).

### Traditions without refinements:
- Utilitarian, Libertarian, Communitarian, Pragmatist — base files only, no refinements, no sources.

## The Agent Psychology System

Each agent in a scenario has a `Psychology` dataclass with:
- **Biases**: loss_aversion, status_quo, confirmation, sunk_cost, bandwagon, authority, in_group, availability, anchoring, self_serving
- **Incentives**: primary_goal, public_goal, fears, pressures, reelection, financial_stake, reputation_stake
- **Personality floats** (0.0–1.0): stubbornness, courage, vanity, empathy, crowd_sensitivity

These shape how agents filter their tradition's ideal reasoning during the constrained pass.

## Conventions

- All Python scripts run from the `polis/` directory: `python3 polis.py`, `python3 teach.py socratic`, `python3 knowledge.py ingest socratic`
- When working remotely (Cowork), you need to mount `~/polis` or the specific subdirectory you're working with
- Source texts should be plain `.txt` files in `traditions/sources/{tradition}/`
- The project uses the Anthropic Python SDK for Claude calls and optionally the OpenAI SDK
- Tradition names are lowercase: `socratic`, `aristotelian`, `jesus`, `utilitarian`, `libertarian`, `communitarian`, `pragmatist`
