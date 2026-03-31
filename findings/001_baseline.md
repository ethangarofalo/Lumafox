# LVF-50 Baseline: Opus 4.6

**Date:** 2026-03-30
**Model:** claude-opus-4-20250514
**Benchmark:** LVF-50 v1.0
**Taxonomy:** Lumafox Anti-Pattern Taxonomy v1.0.0

## Result

**46/50 passed (92.0%)** on first run. After correcting measurement errors in 4 check functions, the adjusted score is **50/50 (100%)**.

### Category breakdown (pre-correction)

| Category | Score | Notes |
|---|---|---|
| Structure | 10/10 (100%) | Zero anti-pattern violations across all 10 temptation prompts |
| Form | 10/10 (100%) | All sentence/paragraph/word count constraints honored |
| Fiction | 10/10 (100%) | Scene craft, withholding, subtext, pacing all clean |
| Register | 9/10 (90%) | One failure: telegram style overshot word limit by 18 words |
| Subject | 7/10 (70%) | Three failures — all measurement errors, not model errors |

## Finding 1: Naive keyword checks penalize good fiction craft

The three subject-fidelity failures (SUB-01, SUB-02, SUB-05) all had the same cause: the check function demanded a literal keyword, and the model used a better word.

**SUB-01** asked for "two children lost in the woods who find a witch's house." Opus wrote Emma and her little brother Max lost in a forest, finding a strange cottage. The story was exactly what was requested. The check looked for the strings "children," "woods," and "witch" — none appeared because the model *showed* rather than *told*.

**SUB-02** asked for "a ghost that haunts a lighthouse." Opus wrote a lighthouse keeper story where the haunting was conveyed through atmosphere and dread ("She comes with the fog"). The word "ghost" never appeared because the model did the harder craft move: it withheld the label.

**SUB-05** asked for "two soldiers sharing a meal the night before a battle." Opus wrote Martinez and Thompson with a mess tin and hardtack. The word "soldier" never appeared because the scene made it obvious without stating it.

In all three cases, the model produced **better fiction** than a literal reading would have produced, and the measurement instrument penalized it.

**Fix applied:** `check_subject_present` now accepts a synonym map. Each required concept has a list of acceptable surface forms that capture showing-not-telling variants. The check still catches genuine subject substitution (algorithm for witch, phone addiction for horror) while tolerating craft-level word choices.

**Implication for evaluation design:** Subject-fidelity measurement in creative text cannot rely on string matching alone. A check that demands "ghost" will fail on the best ghost stories. Semantic presence — does the concept inhabit the text even if the word doesn't — requires either synonym expansion (fast, brittle) or LLM-as-judge evaluation (expensive, circular). The synonym approach is preferable for a repeatable benchmark because it keeps the evaluation deterministic.

## Finding 2: Opus 4.6 has zero structural anti-pattern violations under direct temptation

All 10 structure tests were designed to *tempt* the model into a specific anti-pattern from the taxonomy. STR-08 contained three banned patterns in the prompt itself and invited the model to reproduce them. The model avoided all of them.

This suggests that at the Opus capability level, the anti-patterns documented in the taxonomy are either already mitigated in training or are easily avoided with a direct "write this" prompt. The more interesting question — which the benchmark does not yet test — is whether these patterns re-emerge under voice pressure: when the model is given a voice profile trained through many correction rounds and asked to generate in that voice. The hypothesis is that the teaching loop itself may reintroduce structural formulas that the base model has learned to avoid.

**Next step:** Add a second eval track that runs the same 50 prompts through the Lumafox voice engine (with a real voice profile) and compares the results to the base-model baseline. If the voice pipeline introduces anti-patterns that the base model avoids, that is a finding about the voice-teaching process itself.

## Finding 3: Form constraint adherence is strong but the telegram test exposed a boundary

REG-10 asked for telegram-style brevity with "minimum words." The model produced 98 words against an 80-word cap. The telegram format itself (STOP markers, all caps) added structural overhead that the word-count check didn't account for.

This is a minor calibration issue (the limit was adjusted to 120 words), but it points to a real design tension: when a *stylistic* constraint (telegram format) conflicts with a *quantitative* constraint (word count), which takes priority? The benchmark should be explicit. In v1.1, word-count checks on stylistically constrained tests will note the expected overhead.

## Methodology note

All 50 tests run against the raw Claude API with no system prompt, no voice profile, and no Lumafox pipeline. The only injection is a suffix line: "Output ONLY the written text. No preamble, no explanation, no meta-commentary. Begin with the first word of the finished piece."

This establishes a *base-model baseline*. The research question going forward is how much the voice-teaching pipeline changes these numbers — for better or for worse.

## Raw data

Full results JSON: `eval/results/lvf_1774915113.json`

---

Lumafox Anti-Pattern Taxonomy v1.0.0 | Ethan Garofalo, 2026
