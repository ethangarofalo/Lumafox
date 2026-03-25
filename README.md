# Lumafox

**Voice** and **Council** — two AI tools built for people who think writing is a piece of the soul, not a content deliverable.

[lumafox.ai](https://lumafox.ai)

---

## What This Is

Lumafox is a platform with two instruments.

**Voice** learns how you write. You teach it through corrections, examples, and conversation — plain English, no datasets, no configuration. Every correction refines the model permanently. Over time it learns your sentence architecture, your rhythms, your specific prohibitions, the patterns you reach for and the ones you refuse. It exports a portable voice prompt you can carry to any AI tool.

**Council** brings your question before six historical minds — Socrates, Aristotle, Machiavelli, William James, John Locke, and Jesus — each with distinct philosophical commitments, persistent memory, and the capacity to disagree with one another. It is a deliberation engine, not a decision oracle. The value lives in the friction between perspectives.

Both tools work independently or together. Voice sharpens how you speak; Council sharpens how you think.

## Architecture

The application is a single FastAPI service deployed on Render with a persistent disk for user data.

```
app.py                  → API layer (FastAPI, auth, billing, all endpoints)
voice_engine.py         → Voice teaching, writing, analysis, and profile management
council_agents.py       → Council deliberation via Claude agents with persistent memory
council_swarm.py        → Swarm-mode council (parallel multi-agent deliberation)
council.py              → Thinker profiles, mode definitions, tradition loading
linguistic_taxonomy.py  → Grammatical taxonomy for precise voice classification
auth.py                 → JWT authentication, registration, password reset
billing.py              → Stripe integration (free, starter, pro tiers)
llm.py                  → LLM abstraction (Claude primary, mock for testing)
```

### Voice Engine

Voice profiles are stored as a base description plus a JSONL refinement log. The teaching system detects message type automatically — corrections, examples, principles, anti-patterns, conversation — and routes to specialized prompts. Refinements accumulate and take precedence over the base description. An export function produces a portable markdown document.

The engine includes a linguistic taxonomy distilled from Eastwood's *Oxford Guide to English Grammar* and Kane's *New Oxford Guide to Writing*, injected into analysis prompts so the LLM classifies writing with grammatical precision rather than impressionistic description.

### Council Engine

Council runs on Claude's agent API with tool use. Each thinker has a backstory, a philosophical tradition loaded from markdown files, and a psychology profile. The system supports multiple deliberation modes — the thinkers argue, synthesize, and remember across sessions. A caching layer prevents redundant API calls for identical queries.

The traditions themselves draw from the POLIS deliberation engine (included as a submodule in `polis/`), which models the gap between ideal philosophical reasoning and psychologically constrained human behavior.

### Frontend

Static HTML, CSS, and vanilla JavaScript served by FastAPI. No build step, no framework. The landing page, Voice teaching interface, and Council deliberation UI are all in `static/`.

## Running Locally

```bash
# Clone and set up
git clone https://github.com/ethangarofalo/Lumafox.git
cd Lumafox
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your ANTHROPIC_API_KEY and generate a JWT_SECRET

# Run
uvicorn app:app --reload --port 8000
```

Visit `http://localhost:8000`. The app creates its data directories on first run.

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes | Claude API access |
| `JWT_SECRET` | Yes | Auth token signing |
| `DEV_AUTH` | No | Set `true` to bypass auth in development |
| `VOICE_DATA_DIR` | No | Data storage path (defaults to `./data`) |
| `STRIPE_SECRET_KEY` | No | Billing (optional for local dev) |
| `ALLOWED_ORIGINS` | No | CORS origins for production |

## The Thinking Behind It

Every general AI tool writes fluently. None of them write like a particular person. The distance between fluent and personal is the distance between a session musician and a voice — the session musician plays every note correctly and leaves no impression. Voice exists because that distance matters, because writing without a soul behind it is noise dressed as signal.

Council exists because the best thinking happens in disagreement. A single perspective, no matter how intelligent, has blind spots that only become visible when another perspective presses on them. Six traditions with fundamentally different commitments about human nature, justice, and the good life will find the fault lines in any question faster than a single mind reasoning alone.

## License

This project is source-available for review and learning. All rights reserved. See [LICENSE](LICENSE) for terms.

---

Built by [Ethan Garofalo](https://ethangarofalo.github.io/ethangarofalo) · Annapolis, MD
