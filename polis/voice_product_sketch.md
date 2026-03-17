# Voice Authoring Platform — Product Sketch

## The Philosophy

Every individual has a unique voice inside them. Most people go through their entire education without ever being taught to discover or cultivate it. What is most themselves — their particular way of seeing, their natural rhythm of thought, the images and convictions that are theirs alone — gets buried under years of expected output. School teaches you to write the way school expects. Work teaches you to write the way work expects. AI, left to its defaults, teaches you to write the way a language model writes: competent, generic, interchangeable with ten million other outputs.

The result is a world of incredibly boring, colorless writing, produced by people who are not boring or colorless at all. The voice is in there. It just hasn't been found.

This product helps people uncover what is already within them. For individuals, it enhances the voice or cleans it off if it's dusty or dirty — excavates what was always there but never had the right conditions to emerge. For companies, it ensures that every output matches the tone the company wants to sell and the image it wants to portray.

We are not installing voices. We are uncovering them.

## What the Product Does

You teach an AI your voice through dialogue, not configuration. Not by filling out a form or pasting a paragraph into custom instructions. Through conversation. Through correction. Through the slow, accumulating process of showing the machine what you would say and what you would never say, until it knows the difference.

For people who already have a developed voice, this means preservation — the AI learns to write the way they write, not the way a language model writes by default.

For people who don't yet know their voice, this means discovery — the AI offers starting points drawn from curated traditions and styles, and through the act of correcting what doesn't feel right, the person discovers what does. They find themselves by subtraction. Each "no, not like that" is a small act of self-knowledge. Each "yes, that's closer" is a step toward the voice that was always in them.

## Three Entry Points, One Engine

### 1. Bring Your Voice (Preservation)

For writers, founders, consultants, and anyone with an established voice who wants to maintain it when working with AI.

- Upload samples of your writing
- The system analyzes your patterns: rhythm, diction, metaphor, structure, what you reach for and what you avoid
- Interactive teaching sessions: the AI writes in your voice, you correct it, corrections accumulate
- The result is a Voice Profile — a living document that encodes how you think and write

**The user**: "I've spent years developing this voice. Every AI tool tries to replace it with a generic one. This is the only tool that learns mine."

### 2. Find Your Voice (Discovery)

For people who know what they admire but haven't found their own register yet.

- Paste three to five pieces of writing you wish you had written — essays, articles, posts, anything that moves you
- The system identifies the patterns in what you're drawn to: what kind of rhythm, what kind of imagery, what relationship between thought and feeling
- A starter Voice Profile is assembled from those patterns — not a copy of the source writers, but a distillation of what draws you
- You write with this starter profile. Every correction you make — "less formal," "I wouldn't use that word," "this feels too cold" — moves the voice away from the template and toward you
- Over time, the voice that emerges is yours. Not because you invented it from nothing, but because you discovered it through the oldest method there is: apprenticing to what you love, and finding out where you diverge

**The user**: "I never knew how to describe what I wanted my writing to sound like. Now I do, because I've been correcting toward it for three months."

### 3. Choose a Voice (Professional)

For professionals and companies who need a polished, consistent register immediately.

- Browse a curated library of professional voice profiles: The Strategist, The Analyst, The Essayist, The Executive, The Founder, The Advisor
- Each profile has been built with the same refinement system that powers the entire platform — not a prompt, but dozens of accumulated corrections, principles, examples, and anti-patterns
- Select a profile, customize it ("more direct," "less jargon," "warmer"), and the system learns your preferences as you work
- For companies: the company Voice Profile becomes the living style guide. Every team member writes with it. New hires are on-voice from day one.

**The user**: "I needed to sound professional yesterday. I started with The Strategist and adjusted it until it sounded like me on my best day."

**The company**: "Our brand voice used to be a PDF nobody read. Now it's built into the tool every writer on the team uses."

## The Core Loop

### TEACH Mode

The heart of the product. An interactive conversation where you teach the AI your voice through correction and example.

**Teaching commands:**
- `/example [text]` — show the agent what your writing actually looks like
- `/correct [text]` — tell it what it got wrong and why
- `/principle [text]` — teach a core principle of your voice ("I always ground abstractions in physical images")
- `/voice [text]` — note about tone, rhythm, phrasing ("Semicolons accumulate; they never contrast")
- `/never [text]` — something you would NEVER write ("I would never say 'moreover' or 'it's worth noting'")
- `/try [topic]` — have the agent attempt a passage on a topic in your voice
- `/review` — review all accumulated refinements
- `/export` — export your Voice Profile as a portable document

Each interaction produces a refinement. Refinements accumulate. The agent gets better with every session.

### WRITE Mode

Draft, extend, revise, or complete text using your trained Voice Profile.

- Draft a blog post from an outline
- Extend a paragraph into a full section
- Rewrite a passage that doesn't sound right
- Generate variations
- Adapt a piece for a different format (essay → tweet thread, report → email)

This is the daily-use mode. Teach mode is the investment; write mode is the return.

### ANALYZE Mode

Evaluate any piece of writing against the Voice Profile.

- Where does the voice break? Where does it lapse into generic register?
- Which sentences contradict the principles you've taught?
- Where does it use patterns marked as "never"?
- For teams: submit any draft for voice-consistency analysis before publishing

## The Voice Profile

The Voice Profile is the core artifact. It is a portable, exportable, machine-readable encoding of how a person or organization thinks and writes.

**What it contains:**
- **Principles**: the rules that govern the voice ("Extended metaphors sustained across passages, not used as quick illustrations")
- **Anti-patterns**: what the voice would never do ("Never use 'moreover,' 'furthermore,' or transitional machinery")
- **Voice notes**: observations about tone, rhythm, phrasing ("Sentences build through repetition and variation, not through logical connectives")
- **Examples**: actual samples of the voice in action, with context
- **Corrections**: specific moments where the AI got it wrong and what the right version looks like

**Format:** JSONL (one refinement per line), with a structured schema that makes profiles portable and interoperable.

**Export:** A Voice Profile can be exported as a structured markdown document usable in any AI tool — Claude Projects, system prompts, custom GPTs, or any LLM interface. The export is the distribution mechanism: every exported profile is a demonstration of what the platform can do.

## Who Pays

### Individuals ($15–30/month)

Anyone who produces written work and wants it to sound like a human being — specifically, like them.

- **Writers and essayists**: drafts that match their published voice
- **Consultants and advisors**: consistent professional voice across all output
- **Founders**: investor updates, blog posts, hiring pages that sound like a person runs the company
- **Academics**: first drafts that match their intellectual voice
- **Content creators**: maintain voice across high-volume output
- **Aspirational writers**: people who don't yet have a voice but want to develop one

### Teams ($50–200/month per seat)

Organizations that need voice consistency across multiple writers.

- **Brand voice**: teach the company voice once, give every writer access
- **Editorial teams**: house style enforced by the tool, not a PDF
- **Agencies**: manage multiple client voices, switch between them
- **Law firms, consulting firms**: client-facing documents in a consistent register
- **Marketing teams**: every piece of content on-brand regardless of who drafted it

### Enterprise (custom pricing)

Organizations that need the full POLIS deliberation engine — multi-tradition reasoning for policy analysis, strategic planning, ethical review — in addition to voice authoring.

Voice authoring is the door. Deliberation is what's behind it.

## Technical Architecture

### Existing Technology (from POLIS)

The core engine already works. It was built for teaching philosophical traditions to AI agents, and the mechanism is identical to teaching a personal voice:

| Component | What It Does | Adaptation |
|-----------|-------------|------------|
| teach.py | Interactive teaching REPL with /commands | Wrap as API endpoints |
| Refinement JSONL | Accumulated corrections in structured format | Add user scoping |
| Refinement types | principle, voice_note, anti_pattern, correction, example | Already complete |
| LLM abstraction | Provider-agnostic (Claude, OpenAI, mock) | Minimal changes |
| Prompt assembly | Combines base description + refinements into agent prompt | Adapt for Voice Profiles |
| Knowledge graph | Ingest and retrieve source texts | Repurpose for writing samples |

### What Needs to Be Built

**Backend (FastAPI + Python):**
- User authentication (email/password + OAuth)
- Voice Profile CRUD
- Teaching session management
- Write mode endpoint (generate text with profile)
- Analyze mode endpoint (evaluate text against profile)
- Writing sample ingestion and analysis
- Usage tracking + Stripe billing
- Profile export

**Frontend:**
- Teaching interface: chat UI with /commands
- Writing interface: editor with AI assistance
- Profile dashboard: view refinements, manage profiles
- Sample upload: drag-and-drop writing samples
- Voice library: browse and select starter profiles

**Infrastructure:**
- Hosting: Railway or Fly.io
- Database: PostgreSQL (users, profiles, sessions)
- Storage: S3 or filesystem (refinement files, samples)
- LLM: Claude API (default), provider-agnostic
- Payments: Stripe

## MVP — What Ships First

### Week 1–2: Backend
- FastAPI app with auth and profile CRUD
- Teaching endpoint (adapt teach.py logic)
- Writing sample ingestion and pattern analysis
- Write mode endpoint
- Claude API integration

### Week 3: Frontend
- Single-page app: Teach, Write, Profile tabs
- Chat interface for teaching sessions
- Simple editor for write mode
- Profile view showing accumulated refinements

### Week 4: Polish + Launch
- Stripe integration
- Free tier: 1 profile, 10 interactions/day
- Paid tier: unlimited profiles and interactions
- Voice Profile export
- Landing page
- 3–5 demo Voice Profiles (curated starter voices)

### What MVP Does NOT Include
- Team features (v2)
- Analyze mode (v2)
- Voice library browsing (v2 — launch with starter profiles only)
- Multiple LLM providers (Claude only for v1)
- Deliberation engine integration (enterprise, later)
- Mobile app (web only)

## The Moat

**1. Accumulated refinements.** Once a user has spent hours teaching the system their voice, they won't rebuild on another platform. The switching cost is the teaching itself.

**2. Curated voice traditions.** The starter profiles in the library are not prompts — they're the product of careful philosophical and literary attention. The Socratic voice was built from Strauss's seminars. The essayist voice from years of studying prose rhythm. This quality of curation cannot be replicated by accident.

**3. The refinement format.** If Voice Profiles become the standard way people encode their voice for AI, the platform that generates and hosts them wins.

**4. Network effects (v2+).** When teams share profiles, when agencies manage client voices, when exported profiles circulate — each use reinforces the platform.

## The Rhetoric

The product IS the marketing. Every piece of content produced using a Voice Profile is proof that the product works. Every essay that sounds like a human being wrote it is a demonstration.

**The line:** "AI gave everyone the ability to write. We give you the ability to sound like yourself."

**The deeper line:** "Voice is the last scarcity. When the machine can produce anything, the only question left is: who is speaking?"

**The conviction beneath everything:** Every person has a unique voice inside them. Most have never been taught to find it. We built the tool that helps them look.
