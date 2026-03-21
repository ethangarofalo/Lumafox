"""
Linguistic taxonomy for philological voice analysis.

Distilled from the Oxford Guide to English Grammar (Eastwood, 1994)
and The New Oxford Guide to Writing (Kane, 1988).

This module provides a compact, structured reference that gets injected
into voice analysis prompts so the LLM can classify writing with real
grammatical precision rather than impressionistic description.
"""

# ── Compact taxonomy for prompt injection ────────────────────────────────────
# This is the version that actually gets embedded in LLM prompts.
# It must stay under ~2500 tokens to leave room for the writing sample.

LINGUISTIC_TAXONOMY = """## LINGUISTIC TAXONOMY FOR VOICE CLASSIFICATION

Use the following categories to analyze writing with grammatical precision.
Identify WHICH patterns the writer uses, which they AVOID, and at what FREQUENCY.

### SENTENCE ARCHITECTURE

**Sentence Styles (classify the dominant pattern):**
- SEGREGATING: Short, independent, syntactically simple sentences in sequence. Staccato, observational.
- FREIGHT-TRAIN: Independent clauses coupled by "and" / "but" / "so". Rolling, additive, stream-of-consciousness.
- TRIADIC: Three-unit freight-train. The third unit carries the punch or resolution.
- CUMULATIVE (LOOSE): Main clause first, then modifying phrases accumulate after it. The dominant modern English pattern.
- PERIODIC: Main clause delayed to the end; subordinate material builds suspense before it.
- CENTERED: Main clause sits in the middle, framed by subordinate material on both sides.
- CONVOLUTED: Main clause interrupted mid-stream by parenthetical material.
- BALANCED: Two roughly equal halves divided by a central pause (often semicolon). Antithesis is the sharp variant.
- PARALLEL: Two or more grammatically identical constructions occupying the same position. Creates rhythm and linkage.
- FRAGMENT: Deliberately incomplete sentence. Emphasis, mimics speech, isolates an idea.

**Clause Patterns:**
- COORDINATION ratio vs. SUBORDINATION ratio (paratactic vs. hypotactic)
- Preferred subordinate clause types: temporal, causal, concessive, conditional, relative
- Relative clause preference: restrictive vs. non-restrictive, full vs. participle-reduced

**Information Structure:**
- FRONTING: Moving objects, complements, or adverbials before the subject
- CLEFT SENTENCES: "It was X that..." or "What she wanted was..."
- END-FOCUS: Most important element placed last (natural English stress)
- INVERSION: Subject-verb order reversed for emphasis
- THERE-EXISTENTIAL: "There was a silence." Introduces new entities.
- ELLIPSIS: Which elements the writer omits for compression

### RHETORICAL DEVICES

**Emphasis:**
- Announcement (preliminary signal before the key point)
- Short sentence amid long (sudden force through brevity)
- Negative-positive restatement ("not X but Y")
- Rhetorical question
- Interrupted sentence (parenthetical suspension)

**Repetition Patterns:**
- POLYSYNDETON: Many conjunctions ("and...and...and"). Slows pace, adds weight.
- ASYNDETON: Omitted conjunctions. Speeds pace, creates urgency.
- ANAPHORA: Same word/phrase at the beginning of successive clauses.
- EPISTROPHE: Same word/phrase at the end of successive clauses.
- CHIASMUS: ABBA reversal of structure. Mirror effect.
- TAUTOLOGIA: Restating an idea in different words.

**Concision Devices:**
- Participle phrases replacing full clauses
- Predicate adjectives replacing relative clauses
- Colon/dash replacing "that is" or "namely"
- Parallelism as factoring (shared elements appear once)

### DICTION

**Register:** Formal / General / Informal / Mixed (deliberate register-shifting)
**Etymology:** Anglo-Saxon (short, concrete, physical) vs. Latinate (longer, abstract, intellectual)
**Specificity:** General ("vehicle") vs. Specific ("rusted Ford pickup")
**Abstraction:** Concrete (sensory, tangible) vs. Abstract (conceptual, idea-based)
**Unusual choices:** Archaisms, neologisms, nonce compounds, foreign terms, transferred epithets
**Collocations:** Conventional vs. surprising/unusual word pairings

### FIGURATIVE LANGUAGE

**Comparison:** Simile (explicit, "like/as") vs. Metaphor (implicit, asserts identity)
**Extension:** Single-use vs. Extended/sustained across a passage
**Source domains:** Where metaphors draw from — body, nature, architecture, religion, warfare, technology, domestic life
**Substitution:** Metonymy, synecdoche, personification
**Scale:** Hyperbole (overstatement) vs. Litotes (understatement)
**Wordplay:** Irony (verbal/structural), paradox, oxymoron, zeugma, pun
**Allusion density:** Frequency of references to other texts, myths, history

### IMAGERY (SENSORY CHANNELS)

Classify which channels the writer favors: Visual, Auditory, Tactile, Kinesthetic, Olfactory.
Note density (how often imagery appears) and vividness (how specific and fresh).

### PUNCTUATION AS STYLE

- PERIOD patterns: Heavy use = segregating; light use = flowing
- SEMICOLON: Paratactic linkage (implies relationship reader must infer) vs. absent entirely
- COLON: Specification/announcement vs. unused
- DASH: Interruption, emphasis, afterthought — frequency and function
- COMMA: Oxford comma or not; comma splices (deliberate or absent); series handling
- Series style: Polysyndetic ("A and B and C") vs. Asyndetic ("A, B, C") vs. Standard ("A, B, and C")

### PARAGRAPH ARCHITECTURE

**Development:** Deductive (topic sentence first) vs. Inductive (point at end) vs. Pivoting (turn with "but") vs. Accumulative (piling without thesis)
**Length pattern:** Short (1-3 sentences), medium, long (8+), or varied
**Transitions:** Explicit connectives ("However...") vs. Implicit/semantic vs. Absent (juxtaposition)
**Unity:** Tight (one idea per paragraph) vs. Loose (associative drift)

### HOW TO USE THIS TAXONOMY

1. Identify DOMINANT patterns — which categories recur most
2. Identify ABSENCES — what the writer never does is as diagnostic as what they do
3. Track RATIOS — coordination vs. subordination, concrete vs. abstract, short vs. long
4. Note SIGNATURE COMBINATIONS — voice = the intersection of choices across categories
5. Be SPECIFIC — "cumulative sentences with Latinate diction and semicolons" not "flowing style"
"""


# ── Structured feature categories for programmatic use ───────────────────────

SENTENCE_STYLES = [
    "segregating", "freight_train", "triadic", "cumulative",
    "periodic", "centered", "convoluted", "balanced",
    "parallel", "fragment",
]

RHETORICAL_DEVICES = [
    "polysyndeton", "asyndeton", "anaphora", "epistrophe",
    "chiasmus", "tautologia", "negative_positive",
    "rhetorical_question", "announcement", "interruption",
]

FIGURATIVE_TYPES = [
    "simile", "metaphor", "extended_metaphor", "metonymy",
    "synecdoche", "personification", "hyperbole", "litotes",
    "oxymoron", "paradox", "irony_verbal", "irony_structural",
    "allusion", "zeugma",
]

DICTION_AXES = [
    ("register", ["formal", "general", "informal", "mixed"]),
    ("etymology", ["anglo_saxon", "latinate", "mixed"]),
    ("specificity", ["general", "specific", "mixed"]),
    ("abstraction", ["concrete", "abstract", "mixed"]),
]

IMAGERY_CHANNELS = [
    "visual", "auditory", "tactile", "kinesthetic", "olfactory",
]

PUNCTUATION_FEATURES = [
    "heavy_periods", "semicolons_paratactic", "semicolons_absent",
    "colons_announcement", "dashes_frequent", "dashes_rare",
    "oxford_comma", "no_oxford_comma", "comma_splices_deliberate",
    "polysyndetic_series", "asyndetic_series", "standard_series",
]

PARAGRAPH_PATTERNS = [
    ("development", ["deductive", "inductive", "pivoting", "accumulative"]),
    ("length", ["short", "medium", "long", "varied"]),
    ("transitions", ["explicit", "implicit", "absent"]),
    ("unity", ["tight", "loose"]),
]
