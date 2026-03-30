# Lumafox Anti-Pattern Taxonomy v1.0.0

A formal classification of structural failure patterns in LLM prose generation.

## What this is

Current-generation language models default to a small, identifiable set of structural patterns when generating prose. These patterns are not errors of fact or safety — they are errors of *craft*. They produce text that reads as competent but generic, recognizable as machine-generated not by any single tell but by the accumulation of structural formulas that no specific human would produce.

This taxonomy classifies those patterns by *structure*, not by specific phrasing. A banned phrase list catches `"What strikes me most..."` — this taxonomy catches the *class of construction* that phrase belongs to (simulated subjectivity) and identifies it across all its surface forms.

## How patterns are identified

Patterns are discovered through Lumafox's voice-teaching loop: users teach a language model their specific writing voice through iterative correction. When the model fails — when it produces output the user rejects — the correction is analyzed for structural properties. A pattern is promoted from observation to taxonomy entry when it meets three criteria:

1. **Recurrence** — appears across multiple voices, not just one user's preferences
2. **Structural identity** — is recognizable as a class of construction, not a one-off phrasing
3. **Correctability** — has a documented fix (correction strategy) that produces better output

## Contents

### `anti_patterns.json`

Machine-readable taxonomy. Structure:

- **`banned_phrases`** — surface-level string patterns (9 entries)
- **`banned_structures`** — structural formula classes (5 entries), each with:
  - Examples of the failure
  - Rhetorical explanation of why it fails
  - Correction strategy and corrected examples
  - Detection difficulty and frequency ratings
- **`general_rules`** — high-level constraints (3 entries)
- **`failure_modes`** — documented behavioral failures in LLM generation (6 entries), each with discovery date, status, and fix

### Using the taxonomy

The JSON is designed to be consumed programmatically — for evaluation suites, prompt injection, or analysis tools. Each entry has a stable ID (`BP-001`, `BS-001`, `FM-001`, etc.) for citation and tracking.

## Versioning

- **v1.0.0** (2026-03-30) — Initial extraction from Lumafox voice engine. 9 banned phrases, 5 banned structures, 3 general rules, 6 failure modes.

## License

MIT. Cite as: Lumafox Anti-Pattern Taxonomy v1.0.0, Ethan Garofalo, 2026.
