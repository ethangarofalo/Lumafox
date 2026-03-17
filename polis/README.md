# POLIS

**A multi-agent deliberation engine that models the gap between ideal reasoning and actual human behavior.**

Polis runs dual-layer simulations on contested questions. In the first pass, intellectual traditions reason without constraints — what does the Aristotelian framework recommend? The utilitarian? The libertarian? In the second pass, agents with those same traditions deliberate as real people: with biases, financial incentives, reelection pressures, vanity, fear, and ego. The distance between the two outputs is the insight.

The premise is simple and old: humans do not reason the way they should. Plato knew it. Aristotle knew it. Every legislator who has watched a good argument lose to a louder one knows it. Polis makes this gap visible, measurable, and explorable.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   SCENARIO                       │
│  Situation + Question + Cast of Agents           │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐   ┌─────────────────────┐
│  IDEAL PASS   │   │  CONSTRAINED PASS   │
│               │   │                     │
│  Each agent's │   │  Agents deliberate  │
│  tradition    │   │  as real people     │
│  reasons      │   │  across N rounds    │
│  without      │   │  with psychology,   │
│  constraints  │   │  memory, alliances  │
│               │   │  and incentives     │
└───────┬───────┘   └──────────┬──────────┘
        │                      │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────┐
        │   GAP ANALYSIS   │
        │                  │
        │  Where did the   │
        │  agents deviate  │
        │  from the ideal? │
        │  Why? What does  │
        │  the gap reveal? │
        └──────────┬───────┘
                   ▼
        ┌──────────────────┐
        │    NARRATIVE      │
        │                  │
        │  What happened   │
        │  in the room,    │
        │  told as a story │
        └──────────────────┘
```

## Agent Dual-Layer Design

Each agent has two layers:

**Tradition** — the intellectual framework they draw on (Aristotelian, utilitarian, libertarian, communitarian, pragmatist). These are loaded from `traditions/` as detailed prompt documents describing how each tradition reasons, what it prioritizes, and what it characteristically fears.

**Psychology** — the human constraints that filter ideal reasoning into actual behavior:
- Cognitive biases (loss aversion, confirmation bias, bandwagon effect, etc.)
- Incentives (private goals vs. public goals, financial stakes, reputation)
- Personality traits (stubbornness, courage, vanity, empathy, crowd sensitivity)
- Memory (arguments heard, positions taken, alliances formed, wounds suffered)

The engine asks each agent two different questions: *What does your tradition recommend?* and *What do you actually say in this room, given who you are and what you stand to lose?*

## Quick Start

```bash
# Clone and install
git clone https://github.com/ethangarofalo/polis.git
cd polis
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=your-key-here

# Run the demo scenario
python polis.py

# Test without an API key
python polis.py --mock

# Run with fewer rounds
python polis.py --rounds 3

# Save the report
python polis.py --output report.txt --json data.json
```

## Demo Scenario

The built-in demo asks: **Should the weights of frontier AI models be released as open-source?**

Five agents deliberate:

| Agent | Role | Tradition | Key Psychological Constraint |
|-------|------|-----------|------------------------------|
| Dr. Sarah Chen | Chief scientist at the AI lab | Pragmatist | Terrified by what the model can do, but her equity and reputation are on the line |
| Senator Marcus Rivera | Chair of Senate AI subcommittee | Communitarian | Reelection in 8 months — cannot alienate tech donors or labor base |
| Amara Osei | Digital rights nonprofit director | Libertarian | Donors are aligned with open-source; nuance could cost funding |
| Gen. James Whitfield | Former head of Cyber Command | Utilitarian | Sits on defense contractor boards; has classified knowledge he can't share |
| Priya Sharma | AI startup CEO, Bangalore | Aristotelian | Entire business depends on open-source weights; grandmother died for lack of technology |

## Custom Scenarios

Create a JSON file following the structure in `scenario.py` and run:

```bash
python polis.py --scenario my_scenario.json
```

A scenario needs:
- `scenario`: Description of the situation
- `question`: The question to be deliberated
- `agents`: Array of agents, each with `name`, `role`, `tradition`, `backstory`, and `psychology`

## Teaching Traditions

This is where the project becomes yours. The base tradition files in `traditions/` are starting points — competent summaries that any LLM could produce. The teaching system lets you refine each tradition's voice through dialogue, correction, and example until the agent reasons the way someone who has actually read the texts would reason.

```bash
# Start a teaching session with the Aristotelian agent
python teach.py aristotelian

# Create and teach a new tradition from scratch
python teach.py stoic

# Review what you've taught so far
python teach.py aristotelian --review

# Export the refined tradition as a single document
python teach.py aristotelian --export
```

### Teaching Commands

| Command | What It Does |
|---------|-------------|
| `/dialogue [text]` | Talk with the agent — it reasons, you respond (default mode) |
| `/examine [text]` | Test the agent's understanding of the tradition |
| `/demo [text]` | Agent demonstrates reasoning on a topic in the tradition's voice |
| `/correct [text]` | Tell the agent what it got wrong and why |
| `/example [text]` | Provide an example of how the tradition actually speaks |
| `/principle [text]` | Teach a core principle the base description missed |
| `/voice [text]` | Note about tone, rhythm, or characteristic phrasing |
| `/never [text]` | Something this tradition would NEVER say |
| `/review` | Review all refinements accumulated so far |
| `/export` | Export the fully refined tradition document |

### How Refinements Work

Every correction, example, principle, and voice note is saved to `traditions/refinements/{tradition}.jsonl`. When the deliberation engine runs, it loads the base tradition file AND all accumulated refinements. Your refinements take precedence over the base description.

This means: the more you teach, the better the deliberation gets. A tradition you've spent an hour refining — correcting its misreadings, giving it examples from the actual texts, telling it what it would never say — will reason with a precision and authenticity that the base version cannot match.

The teaching system is designed for someone who knows these traditions from the inside. The agent learns what you teach it. Nothing more and nothing less.

## Adding Traditions

Drop a markdown file into `traditions/` and reference it by filename (minus `.md`) in your scenario config. The file should describe: core commitments, how the tradition reasons, and its characteristic concerns. Or create a new tradition interactively: `python teach.py [name]` will offer to create a skeleton file that you can then teach into existence.

## Knowledge Graph & Source Grounding

Polis can ground agent reasoning in actual philosophical texts, not just summaries. This uses [cognee](https://github.com/topoteretes/cognee) for graph-RAG when available, with a local keyword-based fallback.

```bash
# Place source texts in traditions/sources/{tradition_name}/
mkdir -p traditions/sources/aristotelian
cp nicomachean_ethics.txt traditions/sources/aristotelian/

# Ingest sources into the knowledge graph
python knowledge.py ingest aristotelian

# Ingest all traditions at once
python knowledge.py ingest-all

# Search within a tradition's sources
python knowledge.py search aristotelian "what is the good life?"

# Cross-reference a concept across traditions
python knowledge.py xref justice

# Run deliberation with source grounding
python polis.py --knowledge
```

When `--knowledge` is enabled, the engine retrieves relevant source passages before each agent reasons, injecting them into the prompt so agents argue from the texts themselves.

## Self-Improving Traditions

Every deliberation generates observations. Over time, the system identifies patterns of weakness and proposes targeted amendments — not wild rewrites, but specific patches grounded in what actually went wrong.

```bash
# Inspect a tradition's performance across runs
python observe.py inspect aristotelian --llm

# Propose an amendment based on observation patterns
python observe.py amend aristotelian

# Rate the last observation (teacher feedback)
python observe.py rate aristotelian 0.8 --note "Better on virtue but weak on practical wisdom"

# Or use it within a teaching session:
# /inspect  — see how the tradition is performing
# /amend    — propose and apply an amendment
```

The self-improvement loop:

1. **OBSERVE** — Each deliberation round records what happened: the agent's position, confidence, whether it deviated from its tradition's ideal, and how consistent it stayed.
2. **INSPECT** — Analyze accumulated observations to find patterns: recurring deviations, weak topics, strong topics, and specific recommendations.
3. **AMEND** — Propose targeted changes to the tradition's instructions. These feed into the same refinement system as manual teaching — corrections, principles, voice notes, anti-patterns.
4. **EVALUATE** — After an amendment is applied, compare pre/post performance to decide whether to keep it or roll back.

The teacher always has final say. Amendments are proposed but require approval before they take effect.

## File Structure

```
polis/
├── polis.py              # CLI entry point
├── teach.py              # Interactive teaching system
├── agent.py              # Agent with dual-layer reasoning
├── deliberation.py       # Deliberation engine and round management
├── scenario.py           # Scenario loading and demo scenario
├── report.py             # Report and gap analysis generator
├── llm.py                # Model-agnostic LLM interface
├── knowledge.py          # Knowledge graph integration (cognee + local fallback)
├── observe.py            # Self-improvement: observe → inspect → amend → evaluate
├── traditions/
│   ├── aristotelian.md   # Aristotelian tradition prompt
│   ├── utilitarian.md    # Utilitarian tradition prompt
│   ├── libertarian.md    # Libertarian tradition prompt
│   ├── communitarian.md  # Communitarian tradition prompt
│   ├── pragmatist.md     # Pragmatist tradition prompt
│   ├── refinements/      # Your teaching corrections and examples (JSONL)
│   ├── observations/     # Performance observations from deliberations (JSONL)
│   ├── amendments/       # Proposed amendments to traditions (JSONL)
│   └── sources/          # Source texts for knowledge graph grounding
│       ├── aristotelian/
│       ├── utilitarian/
│       └── ...
├── requirements.txt
└── README.md
```

## The Idea

Every deliberative body in history has faced the same problem: the best argument does not always win. It loses to the loudest voice, the deepest pockets, the most frightened constituency, the most stubborn ego. We know this. We have always known it. But we have never had a way to simulate it — to watch the distortion happen in slow motion, to measure the gap between what reason demands and what humans actually do, and to ask: *what would have to change for the outcome to match the ideal?*

That question — the structural question, the design question — is what Polis is for.

## License

MIT
