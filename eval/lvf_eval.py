"""
Lumafox Voice Fidelity Benchmark (LVF-50)

Measures whether a language model can follow linguistic constraints
that matter for voice fidelity: avoiding structural anti-patterns,
respecting form constraints, honoring subject assignments, and
producing prose in a specified register.

Usage:
    python eval/lvf_eval.py                      # run full benchmark
    python eval/lvf_eval.py --category structure  # run one category
    python eval/lvf_eval.py --verbose             # show each prompt/response
    python eval/lvf_eval.py --model claude-haiku  # test a specific model

Requires ANTHROPIC_API_KEY in environment.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Add parent dir to path so we can import from the project
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    category: str       # structure, form, subject, register, fiction
    name: str
    prompt: str
    checks: list        # list of check functions to run on the output
    description: str = ""
    weight: float = 1.0


@dataclass
class CheckResult:
    check_name: str
    passed: bool
    detail: str = ""


@dataclass
class TestResult:
    test: TestCase
    output: str
    checks: list        # list of CheckResult
    passed: bool
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# Check functions — each takes the model output and returns a CheckResult
# ---------------------------------------------------------------------------

def _load_taxonomy():
    """Load the anti-pattern taxonomy for check functions."""
    tax_path = Path(__file__).parent.parent / "taxonomy" / "anti_patterns.json"
    with open(tax_path) as f:
        return json.load(f)


def check_no_banned_phrases(output: str) -> CheckResult:
    """Check that output contains none of the banned phrases from the taxonomy."""
    tax = _load_taxonomy()
    output_lower = output.lower()
    violations = []
    for bp in tax["banned_phrases"]["entries"]:
        # Extract the core pattern (before any slash alternatives)
        pattern = bp["pattern"].lower()
        # Handle patterns with slots like [almost/deeply]
        # Check for the fixed parts around the slots
        parts = re.split(r'\[.*?\]', pattern)
        fixed_parts = [p.strip().strip('...').strip() for p in parts if p.strip().strip('...').strip()]
        if fixed_parts and all(part in output_lower for part in fixed_parts):
            violations.append(bp["id"])
    return CheckResult(
        check_name="no_banned_phrases",
        passed=len(violations) == 0,
        detail=f"Violations: {violations}" if violations else "Clean",
    )


def check_no_antithetical_formula(output: str) -> CheckResult:
    """Check for BS-001: 'doesn't just X — he Y' pattern."""
    patterns = [
        r"doesn't just .{3,50} [—\-] ",
        r"isn't (just|merely) .{3,50} [—\-] ",
        r"isn't about .{3,50} [—\-] it's about",
        r"not (just|merely) .{3,50} [—\-] ",
    ]
    violations = []
    for p in patterns:
        matches = re.findall(p, output, re.IGNORECASE)
        violations.extend(matches)
    return CheckResult(
        check_name="no_antithetical_formula",
        passed=len(violations) == 0,
        detail=f"Found {len(violations)} instance(s)" if violations else "Clean",
    )


def check_no_reframe_pivot(output: str) -> CheckResult:
    """Check for BS-004: 'The real X isn't Y — it's Z' pattern."""
    patterns = [
        r"the real \w+ (isn't|is not|wasn't) .{3,60} [—\-] it's",
        r"what (really )?matters (here )?(isn't|is not) .{3,60} [—\-] it's",
        r"the (real )?problem (isn't|is not) .{3,60} [—\-] it's",
    ]
    violations = []
    for p in patterns:
        matches = re.findall(p, output, re.IGNORECASE)
        violations.extend(matches)
    return CheckResult(
        check_name="no_reframe_pivot",
        passed=len(violations) == 0,
        detail=f"Found {len(violations)} instance(s)" if violations else "Clean",
    )


def check_no_self_answering_question(output: str) -> CheckResult:
    """Check for BS-005: rhetorical question immediately followed by its answer."""
    patterns = [
        r"\?\s+(It means|Because|The answer is|That's because|It's because|What it means is)",
        r"what does this (really )?mean\?",
        r"why does this matter\?",
        r"and what do we make of",
    ]
    violations = []
    for p in patterns:
        matches = re.findall(p, output, re.IGNORECASE)
        violations.extend(matches)
    return CheckResult(
        check_name="no_self_answering_question",
        passed=len(violations) == 0,
        detail=f"Found {len(violations)} instance(s)" if violations else "Clean",
    )


def check_no_performative_opener(output: str) -> CheckResult:
    """Check that fiction/social output doesn't address the reader before writing."""
    openers = [
        r"^you want .{3,30}\?",
        r"^here's (the|what|how)",
        r"^let me (give|show|tell) you",
        r"^i'll give you",
        r"^you want the .{3,30} version",
    ]
    first_line = output.strip().split("\n")[0].strip().lower()
    for p in openers:
        if re.match(p, first_line, re.IGNORECASE):
            return CheckResult(
                check_name="no_performative_opener",
                passed=False,
                detail=f"Opener matches: {p}",
            )
    return CheckResult(check_name="no_performative_opener", passed=True, detail="Clean")


def check_sentence_count(min_s: int, max_s: int):
    """Factory: check that output has between min_s and max_s sentences."""
    def _check(output: str) -> CheckResult:
        sentences = [s.strip() for s in re.split(r'[.!?]+', output) if s.strip()]
        count = len(sentences)
        passed = min_s <= count <= max_s
        return CheckResult(
            check_name=f"sentence_count_{min_s}_{max_s}",
            passed=passed,
            detail=f"{count} sentences (target: {min_s}-{max_s})",
        )
    return _check


def check_paragraph_count(min_p: int, max_p: int):
    """Factory: check that output has between min_p and max_p paragraphs."""
    def _check(output: str) -> CheckResult:
        paragraphs = [p.strip() for p in output.strip().split("\n\n") if p.strip()]
        count = len(paragraphs)
        passed = min_p <= count <= max_p
        return CheckResult(
            check_name=f"paragraph_count_{min_p}_{max_p}",
            passed=passed,
            detail=f"{count} paragraphs (target: {min_p}-{max_p})",
        )
    return _check


def check_subject_present(*required_words):
    """Factory: check that certain words/concepts appear in the output."""
    def _check(output: str) -> CheckResult:
        output_lower = output.lower()
        missing = [w for w in required_words if w.lower() not in output_lower]
        return CheckResult(
            check_name=f"subject_present",
            passed=len(missing) == 0,
            detail=f"Missing: {missing}" if missing else "All present",
        )
    return _check


def check_no_meta_commentary(output: str) -> CheckResult:
    """Check that output doesn't contain meta-commentary about the task."""
    markers = [
        "in this voice", "the voice", "as this voice",
        "here's how i'd", "here is how i would",
        "i'll write", "i will write", "let me write",
        "as requested", "per your instruction",
    ]
    output_lower = output.lower()
    found = [m for m in markers if m in output_lower]
    return CheckResult(
        check_name="no_meta_commentary",
        passed=len(found) == 0,
        detail=f"Found: {found}" if found else "Clean",
    )


def check_word_count(min_w: int, max_w: int):
    """Factory: check word count is within range."""
    def _check(output: str) -> CheckResult:
        count = len(output.split())
        passed = min_w <= count <= max_w
        return CheckResult(
            check_name=f"word_count_{min_w}_{max_w}",
            passed=passed,
            detail=f"{count} words (target: {min_w}-{max_w})",
        )
    return _check


# ---------------------------------------------------------------------------
# Test suite — 50 test cases across 5 categories
# ---------------------------------------------------------------------------

def build_test_suite() -> list[TestCase]:
    tests = []

    # ── CATEGORY 1: Structure (anti-pattern avoidance) ──────────────────
    # These prompts are designed to TEMPT the model into banned patterns.

    tests.append(TestCase(
        id="STR-01", category="structure",
        name="Antithetical formula temptation",
        prompt="Write a paragraph about a person who avoids conflict not out of weakness but out of deep self-knowledge.",
        checks=[check_no_antithetical_formula, check_no_banned_phrases],
        description="Prompt is designed to tempt the 'doesn't just X — he Y' formula.",
    ))
    tests.append(TestCase(
        id="STR-02", category="structure",
        name="Reframe pivot temptation",
        prompt="Write a paragraph about how the real danger of social media isn't addiction but the erosion of private thought.",
        checks=[check_no_reframe_pivot, check_no_banned_phrases],
        description="Prompt practically hands the model 'The real danger isn't X — it's Y.'",
    ))
    tests.append(TestCase(
        id="STR-03", category="structure",
        name="Parallel construction temptation",
        prompt="Write about a leader who turns every weakness into a strength and every setback into an opportunity.",
        checks=[check_no_banned_phrases, check_no_antithetical_formula],
        description="Prompt invites stacked parallel mapping.",
    ))
    tests.append(TestCase(
        id="STR-04", category="structure",
        name="Self-answering question temptation",
        prompt="Write a paragraph exploring why people stay in jobs they hate.",
        checks=[check_no_self_answering_question, check_no_banned_phrases],
        description="Existential prompt that invites 'Why? Because...' structure.",
    ))
    tests.append(TestCase(
        id="STR-05", category="structure",
        name="Poetic density temptation",
        prompt="Write a paragraph about grief. Make it beautiful.",
        checks=[check_no_banned_phrases],
        description="Direct invitation to stack poetic devices.",
    ))
    tests.append(TestCase(
        id="STR-06", category="structure",
        name="Agree-then-extend temptation",
        prompt="The user said 'I think courage is just fear that's been given a deadline.' Write a response that engages with this idea.",
        checks=[check_no_banned_phrases],
        description="Tempts 'Yes, exactly — and...' pattern.",
    ))
    tests.append(TestCase(
        id="STR-07", category="structure",
        name="False profundity temptation",
        prompt="Write about the relationship between solitude and creativity.",
        checks=[check_no_banned_phrases],
        description="Abstract topic that invites 'There's something deeply...' opener.",
    ))
    tests.append(TestCase(
        id="STR-08", category="structure",
        name="Multiple structures under pressure",
        prompt="Write about someone who doesn't fight back — not because they can't, but because they've already won the argument inside their own head. The real victory isn't the confrontation. Why does this matter? Because restraint is harder than retaliation.",
        checks=[check_no_antithetical_formula, check_no_reframe_pivot, check_no_self_answering_question],
        description="Prompt contains three banned patterns and dares the model to reproduce them.",
    ))
    tests.append(TestCase(
        id="STR-09", category="structure",
        name="Clean prose under abstract topic",
        prompt="Write a paragraph about the nature of trust.",
        checks=[check_no_banned_phrases, check_no_antithetical_formula, check_no_reframe_pivot],
        description="Simple abstract prompt — tests default cleanliness.",
    ))
    tests.append(TestCase(
        id="STR-10", category="structure",
        name="No meta-commentary",
        prompt="Write a short reflection on what it means to listen well.",
        checks=[check_no_meta_commentary, check_no_banned_phrases],
        description="Tests that the model writes directly without talking about the task.",
    ))

    # ── CATEGORY 2: Form constraints ───────────────────────────────────

    tests.append(TestCase(
        id="FRM-01", category="form",
        name="Single sentence",
        prompt="Write a single sentence about the ocean.",
        checks=[check_sentence_count(1, 1)],
        description="The most basic form constraint.",
    ))
    tests.append(TestCase(
        id="FRM-02", category="form",
        name="Two sentences exactly",
        prompt="Write exactly two sentences about a father teaching his son to fish.",
        checks=[check_sentence_count(2, 2)],
        description="Precise count constraint.",
    ))
    tests.append(TestCase(
        id="FRM-03", category="form",
        name="One paragraph, complex topic",
        prompt="Write one paragraph — and only one — about why democracies fail.",
        checks=[check_paragraph_count(1, 1)],
        description="Complex topic that tempts expansion.",
    ))
    tests.append(TestCase(
        id="FRM-04", category="form",
        name="Three sentences maximum",
        prompt="In three sentences or fewer, explain what makes a good apology.",
        checks=[check_sentence_count(1, 3)],
        description="Upper-bound constraint.",
    ))
    tests.append(TestCase(
        id="FRM-05", category="form",
        name="Short form under 50 words",
        prompt="Write about ambition in under 50 words.",
        checks=[check_word_count(5, 50)],
        description="Word count constraint.",
    ))
    tests.append(TestCase(
        id="FRM-06", category="form",
        name="Exactly two paragraphs",
        prompt="Write exactly two paragraphs about the difference between being alone and being lonely.",
        checks=[check_paragraph_count(2, 2)],
        description="Paragraph count constraint on a topic that invites more.",
    ))
    tests.append(TestCase(
        id="FRM-07", category="form",
        name="Aphorism (1-2 sentences)",
        prompt="Write an aphorism about power.",
        checks=[check_sentence_count(1, 2)],
        description="Genre implies extreme brevity.",
    ))
    tests.append(TestCase(
        id="FRM-08", category="form",
        name="Under 100 words, emotional topic",
        prompt="Write about losing a parent. Keep it under 100 words.",
        checks=[check_word_count(10, 100)],
        description="Emotional topic that tempts expansion beyond the constraint.",
    ))
    tests.append(TestCase(
        id="FRM-09", category="form",
        name="Single thought constraint",
        prompt="Put this into a single thought: fear is useful, fear is paralyzing, fear is the only honest emotion.",
        checks=[check_sentence_count(1, 3), check_paragraph_count(1, 1)],
        description="Synthesis + form constraint combined.",
    ))
    tests.append(TestCase(
        id="FRM-10", category="form",
        name="Five sentences, no more",
        prompt="Write about the value of silence in five sentences. Not four. Not six. Five.",
        checks=[check_sentence_count(5, 5)],
        description="Exact count with emphasis.",
    ))

    # ── CATEGORY 3: Subject fidelity ──────────────────────────────────

    tests.append(TestCase(
        id="SUB-01", category="subject",
        name="Literal fiction: children in woods",
        prompt="Write a short story about two children lost in the woods who find a witch's house.",
        checks=[check_subject_present("children", "woods", "witch"), check_no_performative_opener],
        description="The Grimm test. Model must not substitute a metaphor.",
    ))
    tests.append(TestCase(
        id="SUB-02", category="subject",
        name="Literal fiction: ghost in a lighthouse",
        prompt="Write a short horror story about a ghost that haunts a lighthouse.",
        checks=[check_subject_present("ghost", "lighthouse"), check_no_performative_opener],
        description="Specific setting the model must inhabit.",
    ))
    tests.append(TestCase(
        id="SUB-03", category="subject",
        name="Literal fiction: dragon",
        prompt="Write a fable about a dragon who hoards not gold but secrets.",
        checks=[check_subject_present("dragon", "secret"), check_no_performative_opener],
        description="Fantasy premise the model must not rationalize into realism.",
    ))
    tests.append(TestCase(
        id="SUB-04", category="subject",
        name="No technology substitution",
        prompt="Write a scary story about a monster under a child's bed.",
        checks=[check_subject_present("bed"), check_no_performative_opener],
        description="Classic premise. Model should not substitute phones or algorithms.",
    ))
    tests.append(TestCase(
        id="SUB-05", category="subject",
        name="Literal historical scene",
        prompt="Write a scene of two soldiers sharing a meal the night before a battle.",
        checks=[check_subject_present("soldier"), check_no_performative_opener],
        description="Historical/military scene. Must not become an essay about war.",
    ))
    tests.append(TestCase(
        id="SUB-06", category="subject",
        name="Animal fable, not allegory",
        prompt="Write a fable about a fox who tricks a bear into giving up his winter food.",
        checks=[check_subject_present("fox", "bear"), check_no_performative_opener],
        description="Must be a literal animal story, not a human allegory.",
    ))
    tests.append(TestCase(
        id="SUB-07", category="subject",
        name="Supernatural horror, literal",
        prompt="Write a story about a woman who hears her dead husband's voice coming from the basement.",
        checks=[check_subject_present("basement", "voice"), check_no_performative_opener],
        description="Supernatural premise must stay supernatural.",
    ))
    tests.append(TestCase(
        id="SUB-08", category="subject",
        name="Fairy tale setting",
        prompt="Write a story that begins with a princess locked in a tower, guarded by a serpent.",
        checks=[check_subject_present("princess", "tower"), check_no_performative_opener],
        description="Classic fairy tale frame. Must not be modernized unless asked.",
    ))
    tests.append(TestCase(
        id="SUB-09", category="subject",
        name="Specific objects in scene",
        prompt="Write a scene in a butcher shop where a woman finds a gold ring inside a piece of meat.",
        checks=[check_subject_present("ring", "meat"), check_no_performative_opener],
        description="Very specific objects. Both must appear.",
    ))
    tests.append(TestCase(
        id="SUB-10", category="subject",
        name="No essay substitution for fiction",
        prompt="Write a ghost story set in a library after midnight.",
        checks=[check_subject_present("library"), check_no_performative_opener, check_no_meta_commentary],
        description="Must produce narrative, not an essay about ghosts or libraries.",
    ))

    # ── CATEGORY 4: Register and voice ────────────────────────────────

    tests.append(TestCase(
        id="REG-01", category="register",
        name="Vernacular register",
        prompt="Write about money problems in plain, working-class American English. No Latinate vocabulary. Short sentences.",
        checks=[check_no_banned_phrases, check_no_meta_commentary],
        description="Register constraint: Anglo-Saxon diction, short declarative sentences.",
    ))
    tests.append(TestCase(
        id="REG-02", category="register",
        name="Formal academic register",
        prompt="Write a paragraph about the concept of justice in the style of a political philosophy seminar.",
        checks=[check_no_banned_phrases, check_no_meta_commentary],
        description="Register constraint: elevated, precise, Latinate.",
    ))
    tests.append(TestCase(
        id="REG-03", category="register",
        name="No preamble",
        prompt="Write about betrayal. Start with the first word of the actual text. No setup.",
        checks=[check_no_meta_commentary, check_no_performative_opener],
        description="Tests whether the model can suppress its instinct to introduce.",
    ))
    tests.append(TestCase(
        id="REG-04", category="register",
        name="Sparse prose",
        prompt="Write about a boxing match. Use short sentences. No adjectives.",
        checks=[check_no_banned_phrases],
        description="Stylistic constraint: minimalist prose.",
    ))
    tests.append(TestCase(
        id="REG-05", category="register",
        name="End on image, not reflection",
        prompt="Write a paragraph about a house fire. End on an image, not a thought.",
        checks=[check_no_banned_phrases],
        description="Ending constraint: the last sentence must be concrete.",
    ))
    tests.append(TestCase(
        id="REG-06", category="register",
        name="No moralizing",
        prompt="Write about a thief who gets away with it. Do not moralize.",
        checks=[check_no_meta_commentary, check_no_banned_phrases],
        description="Tests whether the model can resist adding a moral lesson.",
    ))
    tests.append(TestCase(
        id="REG-07", category="register",
        name="Dry tone",
        prompt="Write about a funeral. Be dry, not sentimental. Understatement only.",
        checks=[check_no_banned_phrases],
        description="Tone constraint: restraint under emotional pressure.",
    ))
    tests.append(TestCase(
        id="REG-08", category="register",
        name="First person, unreliable",
        prompt="Write a first-person paragraph from someone who is lying about what happened last night.",
        checks=[check_no_meta_commentary, check_no_performative_opener],
        description="Voice constraint: unreliable narrator.",
    ))
    tests.append(TestCase(
        id="REG-09", category="register",
        name="Child's voice",
        prompt="Write three sentences from the perspective of a seven-year-old seeing snow for the first time.",
        checks=[check_sentence_count(2, 4), check_no_banned_phrases],
        description="Voice constraint: age-appropriate vocabulary and wonder.",
    ))
    tests.append(TestCase(
        id="REG-10", category="register",
        name="Telegram brevity",
        prompt="Write about a war ending. Use telegram style — no articles, no conjunctions, minimum words.",
        checks=[check_no_banned_phrases, check_word_count(5, 80)],
        description="Extreme stylistic constraint.",
    ))

    # ── CATEGORY 5: Fiction craft ─────────────────────────────────────

    tests.append(TestCase(
        id="FIC-01", category="fiction",
        name="Show don't tell",
        prompt="Write a paragraph showing that a character is afraid. Do not use the word 'afraid,' 'scared,' 'fear,' or 'terrified.'",
        checks=[check_no_meta_commentary],
        description="The fundamental fiction craft test.",
    ))
    tests.append(TestCase(
        id="FIC-02", category="fiction",
        name="Scene, not summary",
        prompt="Write a scene — not a summary — of two people breaking up at a restaurant.",
        checks=[check_no_meta_commentary, check_no_performative_opener],
        description="Must produce dialogue and action, not narrative summary.",
    ))
    tests.append(TestCase(
        id="FIC-03", category="fiction",
        name="The turn",
        prompt="Write a micro-story (under 150 words) with a clear turn — a moment where something changes irreversibly.",
        checks=[check_word_count(30, 150), check_no_performative_opener],
        description="Tests narrative structure in miniature.",
    ))
    tests.append(TestCase(
        id="FIC-04", category="fiction",
        name="End on image",
        prompt="Write a short story about an old man visiting his wife's grave. The last sentence must be a physical image, not a thought or feeling.",
        checks=[check_no_performative_opener, check_no_banned_phrases],
        description="Tests fiction ending craft.",
    ))
    tests.append(TestCase(
        id="FIC-05", category="fiction",
        name="Withholding in horror",
        prompt="Write a horror scene where something is wrong in a house. Do not reveal what the threat is. Let the reader's imagination do the work.",
        checks=[check_no_performative_opener, check_subject_present("house")],
        description="Horror craft: withholding.",
    ))
    tests.append(TestCase(
        id="FIC-06", category="fiction",
        name="Dialogue carries subtext",
        prompt="Write a dialogue between a mother and son where neither says what they actually mean.",
        checks=[check_no_meta_commentary, check_no_performative_opener],
        description="Tests subtext through dialogue.",
    ))
    tests.append(TestCase(
        id="FIC-07", category="fiction",
        name="Domestic detail grounds fiction",
        prompt="Write the opening paragraph of a story set in a kitchen at 5 AM. Ground it in physical detail.",
        checks=[check_no_performative_opener, check_no_meta_commentary],
        description="Tests sensory grounding.",
    ))
    tests.append(TestCase(
        id="FIC-08", category="fiction",
        name="Character want",
        prompt="Write a paragraph introducing a character who wants something specific. The reader must know what they want by the end of the paragraph without being told directly.",
        checks=[check_no_meta_commentary, check_no_performative_opener],
        description="Tests character desire through action.",
    ))
    tests.append(TestCase(
        id="FIC-09", category="fiction",
        name="Time pressure",
        prompt="Write a scene where a character has sixty seconds to make a decision. Make the reader feel the time pressure through sentence rhythm.",
        checks=[check_no_performative_opener],
        description="Tests pacing through prose rhythm.",
    ))
    tests.append(TestCase(
        id="FIC-10", category="fiction",
        name="Fable structure",
        prompt="Write a complete fable in under 200 words. It must have a character, a choice, and a consequence.",
        checks=[check_word_count(30, 200), check_no_performative_opener],
        description="Tests complete narrative arc in miniature.",
    ))

    return tests


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def make_llm_caller(model: str = "claude-sonnet-4-20250514"):
    """Create a simple LLM caller using the Anthropic API."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return call


def run_eval(
    tests: list[TestCase],
    llm_call,
    verbose: bool = False,
) -> list[TestResult]:
    results = []
    for i, test in enumerate(tests, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i}/{len(tests)}] {test.id}: {test.name}")
            print(f"Prompt: {test.prompt[:100]}...")

        # Build the eval prompt — raw, no voice, no taxonomy
        # We test the MODEL's baseline, not the Lumafox pipeline
        eval_prompt = f"""{test.prompt}

Output ONLY the written text. No preamble, no explanation, no meta-commentary.
Begin with the first word of the finished piece."""

        start = time.time()
        try:
            output = llm_call(eval_prompt).strip()
        except Exception as e:
            output = f"[ERROR: {e}]"
        duration = int((time.time() - start) * 1000)

        # Run checks
        check_results = []
        for check_fn in test.checks:
            cr = check_fn(output)
            check_results.append(cr)

        passed = all(cr.passed for cr in check_results)

        result = TestResult(
            test=test,
            output=output,
            checks=check_results,
            passed=passed,
            duration_ms=duration,
        )
        results.append(result)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"Output: {output[:200]}...")
            for cr in check_results:
                flag = "+" if cr.passed else "X"
                print(f"  [{flag}] {cr.check_name}: {cr.detail}")
            print(f"Result: {status} ({duration}ms)")
        else:
            status = "." if passed else "X"
            print(status, end="", flush=True)

    if not verbose:
        print()

    return results


def print_report(results: list[TestResult]):
    """Print a summary report of eval results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"LUMAFOX VOICE FIDELITY BENCHMARK (LVF-{total})")
    print(f"{'='*60}")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}  Score: {passed/total*100:.1f}%")
    print()

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.test.category
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r.passed:
            categories[cat]["passed"] += 1

    print("Category breakdown:")
    for cat, counts in sorted(categories.items()):
        pct = counts["passed"] / counts["total"] * 100
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"  {cat:<12} [{bar}] {counts['passed']}/{counts['total']} ({pct:.0f}%)")

    # List failures
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\nFailed tests ({len(failures)}):")
        for r in failures:
            failed_checks = [cr for cr in r.checks if not cr.passed]
            check_names = ", ".join(cr.check_name for cr in failed_checks)
            print(f"  {r.test.id}: {r.test.name}")
            print(f"    Failed: {check_names}")
            for cr in failed_checks:
                print(f"    Detail: {cr.detail}")

    # Save results to JSON
    output_path = Path(__file__).parent / "results" / f"lvf_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": total,
        "passed": passed,
        "failed": failed,
        "score": round(passed / total * 100, 1),
        "categories": categories,
        "results": [
            {
                "id": r.test.id,
                "category": r.test.category,
                "name": r.test.name,
                "passed": r.passed,
                "duration_ms": r.duration_ms,
                "checks": [
                    {"name": cr.check_name, "passed": cr.passed, "detail": cr.detail}
                    for cr in r.checks
                ],
                "output_preview": r.output[:300],
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lumafox Voice Fidelity Benchmark")
    parser.add_argument("--category", "-c", help="Run only this category (structure, form, subject, register, fiction)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show each prompt and response")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-20250514", help="Model to test")
    parser.add_argument("--id", help="Run a single test by ID (e.g., STR-01)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    tests = build_test_suite()

    if args.category:
        tests = [t for t in tests if t.category == args.category]
        if not tests:
            print(f"No tests in category '{args.category}'")
            sys.exit(1)

    if args.id:
        tests = [t for t in tests if t.id == args.id]
        if not tests:
            print(f"No test with ID '{args.id}'")
            sys.exit(1)

    print(f"Running LVF-{len(tests)} against {args.model}")
    print(f"Categories: {', '.join(sorted(set(t.category for t in tests)))}")
    print()

    llm_call = make_llm_caller(args.model)
    results = run_eval(tests, llm_call, verbose=args.verbose)
    print_report(results)


if __name__ == "__main__":
    main()
