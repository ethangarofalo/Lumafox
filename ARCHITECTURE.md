# Architecture Decisions

This document explains *why* the system is built the way it is. The README covers what it does; this covers the thinking behind the choices.

---

## Voice: Why Auto-Mode Instead of Explicit Commands

The first version of the teaching system used explicit commands — `/correct`, `/example`, `/principle`, `/voice`, `/never` — borrowed directly from the POLIS deliberation engine's `teach.py`. The user typed a command prefix, the system knew what kind of input it was receiving, and the refinement was saved. Clean, unambiguous, easy to implement.

It was also wrong for this product. The explicit command system assumed the user would always know, in the moment, whether they were correcting, teaching a principle, or providing an example. In practice, the best teaching happens when the user is just *talking* — when they say "one thing is, I'd never start a sentence that way" in the middle of a conversation and don't realize they've just taught an anti-pattern. Requiring them to stop, categorize their input, and prefix it with a command interrupts the very flow that produces the most valuable refinements.

Auto-mode solves this with a priority-ordered detection cascade:

```
1. Rephrase request?     → Route to rewrite prompt
2. Writing example?      → Route to example analysis prompt
3. In conversation +     → Route to correction prompt (rewrite incorporating feedback)
   correction signals?
4. In conversation +     → Check for offer acceptance, else conversation prompt
   no correction?
5. Default               → Fallback prompt with TEACH tag classification
```

Each branch uses different detection strategies. Rephrase detection looks for phrases like "say this a different way" or "render this." Correction detection scans for signals like "I'd use," "too formal," "instead of," "not quite." Example detection is the most interesting — it combines explicit markers ("here's my writing:") with an implicit heuristic: if the message is longer than 200 characters, we're not in an active writing exchange, there are no correction signals, and the first 50 characters don't contain command words, the system infers the user is sharing a writing sample.

The TEACH tag system handles what auto-detection misses. Every prompt instructs the LLM to append a classification tag (`TEACH:correction`, `TEACH:principle`, `TEACH:example`, `TEACH:voice`, `TEACH:never`, or `TEACH:none`) to its response. The tag is parsed out before the response reaches the user, and if it indicates a refinement-worthy interaction, the system saves it automatically. The user never sees the machinery. They just talk, and the voice learns.

The explicit commands still exist for power users who want direct control, but the default path is conversation.

---

## Voice: The Synthesis Problem

The refinement system stores each teaching interaction as a line in a JSONL file — a type, a content string, an optional context, and a timestamp. After 50 corrections, examples, and principles, this produces a long bulleted list that gets appended to the voice description and injected into every prompt.

The problem: 50 bullet points don't build understanding. They create a checklist that the LLM satisfices against — it checks enough boxes to feel compliant and then falls back on its default register. A correction that says "never use the antithetical formula" is a rule. But without the reasoning behind the rule — *why* the construction is mechanical, what it looks like across different phrasings, what to do instead — the LLM will avoid the specific example it was shown and produce the same structural pattern in different words.

The `synthesize_voice_document` function addresses this. Every 10 refinements, the system calls the LLM to rewrite the entire voice description as a coherent narrative, folding all accumulated refinements into a single document where each observation builds on the last. The prompt frames this as biography, not aggregation: "Think of yourself as a biographer revising a chapter after learning something new about your subject."

The synthesized document replaces the base description plus flat refinement list. Any refinements added *after* the last synthesis are appended as a temporary list until the next synthesis cycle incorporates them. This means the voice description is a living document that gets richer and more precise over time, not a document with an ever-growing appendix.

The design tension here is between fidelity and coherence. The flat list preserves every refinement exactly as taught. The synthesized document may lose nuance in the rewriting. The system keeps both — the JSONL file is the canonical record, the synthesized document is the working prompt — so nothing is ever lost.

---

## Council: Named Thinkers vs. Population Swarm

Council has two deliberation modes that reflect fundamentally different theories about what makes disagreement productive.

**Named Council** uses six historical figures — Socrates, Aristotle, Machiavelli, and others — each implemented as a Claude agent with a tradition file (a markdown document describing their philosophical commitments), a backstory, and persistent memory. They deliberate in two rounds: independent positions first, then responses to each other. The output is individual thinker cards, a tension/alliance map, and a synthesis.

The value of named thinkers is *depth of tradition*. Socrates doesn't just "ask questions" — his tradition file encodes specific Socratic commitments about the incompleteness of philosophy, the tension between the philosopher and the city, the distinction between the just and the legal. These traditions can be refined over time through the same teaching system that Voice uses (inherited from POLIS). The thinker's responses improve as the tradition files deepen.

**Swarm mode** takes a different approach entirely. Instead of six experts, it runs 40 agents drawn from a weighted population distribution calibrated to represent a range of perspectives — pragmatists, traditionalists, religious thinkers, postmodernists, synthesizers, devil's advocates. Each agent is shaped by a tradition but has a *drift parameter* — a float between 0 and 1 that represents how much cultural context has pulled them from their tradition's pure form. A "Christian with 0.7 drift" isn't a theologian; they're someone raised in the tradition who has absorbed significant secular influence. Their reasoning reflects that tension.

The population weights shift based on the question's domain. A question about religious ethics boosts theological traditions. A question about technology boosts digital-native and pragmatist weights. The distribution is renormalized after boosting so the total always sums to 1.

The output of swarm mode is aggregate, not individual — position clusters, tradition breakdowns, opinion shift maps. No individual agent card matters because the agents are interchangeable. What matters is the *distribution* of where 40 differently-situated minds landed, and how that distribution shifted between rounds.

The design question that produced these two modes: is disagreement more productive when it comes from deep expertise (named thinkers who have studied a tradition for centuries) or from demographic diversity (a population that reflects the actual range of how people think)? The answer is both, depending on the question. Council gives the user the choice.

---

## Council: Persistent Memory

Every thinker in Named Council maintains a global memory — a JSONL log of every question they've been asked, every position they've taken, every argument that moved them, and a private thought field (what they believed but didn't say). This memory persists across all users and all sessions. Socrates remembers every question anyone has ever brought before him.

The retrieval mechanism is deliberately simple: keyword overlap scoring, not embeddings. When a thinker encounters a new question, they can call a `recall_memory` tool to search their history. The search scores each memory entry by word overlap with the query and returns the top matches.

Why keyword search instead of embeddings? Three reasons. First, the memory entries are already structured — they have a question field, a position field, and an argument field, so the search space is semantically organized by construction. Second, the thinkers are asking for their *own* memories, not searching a foreign corpus; the vocabulary overlap between a question about justice and a previous answer about justice is high enough that keyword matching works. Third, embedding-based retrieval would require either an external vector store (added infrastructure and cost) or in-memory embeddings (added latency on every deliberation). The keyword approach is fast, stateless, and sufficient.

The memory is append-only and write-safe — entries are appended to the JSONL file, never modified or deleted. This means the memory is also an audit log of every deliberation the system has ever run.

In Swarm mode, memory is per-tradition rather than per-agent. Christianity remembers every question Christianity has addressed, shared across all Christian-tradition agents in every swarm run. This reflects the design principle that in swarm mode, the individual agent is ephemeral but the tradition persists.

---

## Voice: The Anti-Slop Architecture

The hardest problem in voice replication is not teaching the LLM what to do — it's preventing it from doing what it does by default. LLMs have a powerful prior toward "good AI writing": balanced antithetical constructions ("It's not X — it's Y"), formulaic parallel phrases, stacked poetic devices, and smooth transitional machinery. These patterns are genuinely good writing in most contexts, which is why the model defaults to them. But for a specific voice that avoids these patterns, the default is the enemy.

The system attacks this at three levels:

**Level 1: Banned phrase patterns.** A `BANNED_AI_PATTERNS` constant lists specific constructions that are generic AI writing regardless of voice — "There's something almost [adjective] about...", "The question becomes...", "Yes, exactly — and..." This is injected into every writing prompt.

**Level 2: Per-voice anti-patterns.** The refinement system stores anti-patterns taught by the user — structural prohibitions specific to their voice. "Never use the antithetical formula" is different from "never say 'The question becomes'" because it targets a *class* of construction, not a specific phrase.

**Level 3: The synthesis document.** When the voice description is synthesized into a coherent narrative, the anti-patterns are woven into the reasoning rather than listed as rules. Instead of "Don't use X," the synthesized document explains *why* X fails for this voice and *what to do instead*. This gives the LLM contrastive understanding rather than a checklist.

The three levels compound: banned phrases catch the obvious AI-isms, per-voice anti-patterns catch the voice-specific prohibitions, and the synthesis document provides the reasoning that makes both stick.

---

## Billing: Credit Economics

Council deliberations cost real money — each thinker is an API call, and swarm mode runs 40 of them. The billing system uses weekly credits rather than per-call pricing to keep the mental model simple.

The credit costs reflect actual API expense ratios: a lite council (3 thinkers, 1 round) costs 1 credit, a full council (6 thinkers, 2 rounds) costs 3, and a swarm (40 agents, 2 rounds) costs 10. The 3:1 ratio between full and lite roughly mirrors the API cost difference. The 10:3 ratio for swarm is subsidized — 40 agents should cost more, but the per-agent prompts in swarm mode are lighter than named thinker prompts.

Swarm runs are dispatched as background jobs with polling because they exceed Render's 30-second request timeout. The client receives a job ID immediately and polls `/council/job/{job_id}` until completion. Credits are deducted after the run succeeds, not before, so failed runs don't cost the user.

Voice teaching is rate-limited by monthly interaction count rather than credits, because the per-interaction API cost is low and the value of teaching compounds — a user who has taught 200 corrections has a meaningfully better voice model than one who has taught 20. Limiting teaches too aggressively would cap the product's core value proposition.

---

## What's Not Here (Yet)

**Embedding-based voice matching.** The current system relies on the LLM interpreting a text description of the voice. A future version could embed the user's writing samples and retrieve the closest examples at generation time, giving the LLM concrete reference points rather than abstract descriptions.

**Cross-tradition knowledge retrieval.** The POLIS knowledge graph supports ingesting source texts (Strauss lecture transcripts, Aristotle's works, red-letter Gospel text) and feeding relevant passages to agents during deliberation. This is wired into the POLIS submodule but not yet exposed through the Council API.

**Voice-aware Council synthesis.** Currently, Council's synthesis is written in a neutral register. A natural extension: run the synthesis through the user's Voice profile so the final output sounds like them, not like a generic summarizer.
