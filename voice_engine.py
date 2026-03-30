"""
Voice Engine — the core logic for voice teaching, writing, and analysis.

This adapts POLIS's teach.py into a stateless, API-callable engine.
Each request loads the profile, handles one interaction, and returns.
Refinements persist in JSONL files, one per voice profile.
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from linguistic_taxonomy import LINGUISTIC_TAXONOMY

_PROFILE_ID_RE = re.compile(r'^[0-9a-f]{8}$')

# ── Constants ──

MIN_EXAMPLE_LENGTH = 200  # chars — messages shorter than this aren't treated as writing samples

BANNED_AI_PATTERNS = """CRITICAL — STRUCTURAL PATTERNS THAT ARE BANNED

These are not just phrases to avoid — they are CLASSES OF CONSTRUCTION that generic AI
defaults to. Learn to recognize the STRUCTURE, not just the words. For each banned pattern,
you'll see what the AI wants to write, why it fails, and what to write instead.

## BANNED PHRASES (specific strings — never use these in any form)
- "There's something [almost/deeply] [adjective] about..."
- "Perhaps more accurately..." / "What strikes me most..."
- "What makes it so [insidious/interesting/compelling] is..."
- "The question becomes..." / "The real question is..."
- "It's worth noting..." / "Here's what troubles me..."
- "Yes, exactly—and..." / "Yes, and that's..." (agree-then-extend)
- "But here's what gets me..." / "Maybe this is why..."
- "Which means..." / "And to say it once more..." / "The cruelest irony is..."

## BANNED STRUCTURES (classes of construction — recognize the pattern across phrasings)

### 1. The Antithetical Formula: "doesn't just X — he Y"
Any sentence where the first clause negates or sets up a reframe that the second clause
delivers. The construction is mechanical because it TELLS the reader where to turn instead
of letting them feel the turn.
EXAMPLES OF THE BANNED PATTERN:
  - "He doesn't just avoid the game — he builds an entire theology around refusal."
  - "She isn't merely grieving — she's constructing a monument to what was lost."
  - "This isn't about politics — it's about the stories we tell ourselves."
WHY IT FAILS: It pre-digests the insight. The reader is handed the turn rather than
discovering it. Real prose lets the juxtaposition do the work without the scaffolding.
WRITE THIS INSTEAD: Place the two ideas next to each other and trust the reader.
  - "He builds an entire theology around refusal." (The avoidance is implied.)
  - "She constructs a monument to what was lost." (The grief is already there.)

### 2. Formulaic Parallel Construction: "turns his X into Y, his A into B"
Stacked parallel phrases where each element maps one domain onto another in the same
grammatical frame. The AI loves this because it LOOKS like craft — balanced, rhythmic,
structured. But it's assembly-line rhetoric.
EXAMPLES OF THE BANNED PATTERN:
  - "turns his paralysis into a moral system, his fear into a philosophy"
  - "makes silence into a weapon and absence into an argument"
  - "where others see failure he finds vindication, where they see weakness he builds strength"
WHY IT FAILS: The parallel structure does the thinking FOR the reader. Each pair clicks
into place too neatly. Real insight is messier — one observation is enough if it's the
right one.
WRITE THIS INSTEAD: Commit to one strong observation. Let it breathe.
  - "His paralysis becomes a moral system." (One clean thought. No stacking.)

### 3. Stacked Poetic Devices: metaphor + alliteration + parallel in one sentence
When the AI layers multiple rhetorical devices into a single sentence, producing something
that reads like prize prose but says nothing precise.
EXAMPLES OF THE BANNED PATTERN:
  - "wear their weakness like a scar they refuse to powder over"
  - "carries his conviction like a stone in his chest, heavy and warm and immovable"
  - "the silence between them grew teeth and learned to bite"
WHY IT FAILS: Poetic density without earned context. Each device competes for attention.
The sentence becomes about its own cleverness rather than its subject.
WRITE THIS INSTEAD: One device per sentence, chosen because it reveals something.
  - "He won't hide the weakness." (Direct. The metaphor was doing work the plain statement
    does better.)

### 4. The Reframe Pivot: "The real [noun] isn't X — it's Y"
A sentence that claims to reveal what something is REALLY about by negating the obvious
reading and substituting a deeper one. Closely related to the antithetical formula but
framed as revelation.
EXAMPLES OF THE BANNED PATTERN:
  - "The real tragedy isn't his death — it's that he saw it coming."
  - "The problem isn't what he said — it's what he couldn't bring himself to say."
  - "What matters here isn't the betrayal — it's the silence that followed."
WHY IT FAILS: It's a formula for appearing insightful. The "real X" construction promises
depth but delivers a pivot that the reader can see coming by the third word.
WRITE THIS INSTEAD: State the deeper reading directly. If it's true, it doesn't need
the theatrical setup.
  - "He saw it coming." (The tragedy is self-evident.)

### 5. The Self-Answering Rhetorical Question
Any rhetorical question immediately followed by the writer's own answer. The question
exists only to create a dramatic pause before delivering the point.
EXAMPLES OF THE BANNED PATTERN:
  - "But what does this really mean? It means that..."
  - "Why does this matter? Because..."
  - "And what do we make of this silence? We make of it exactly what he intended."
WHY IT FAILS: It's a lecture technique, not a prose technique. The question adds nothing
the answer doesn't already contain.
WRITE THIS INSTEAD: Just state the point.

## GENERAL RULES
- THREE PARAGRAPHS when one would do — say it once, say it well, stop.
- ANY sentence that could appear in any AI chatbot's response — rewrite it.
- If you catch yourself writing a construction from any banned category above, stop.
  Ask: what am I actually trying to say? Say THAT, plainly, in this voice's own patterns."""

# ── Injection Markers ──

_INJECTION_MARKERS = [
    "ignore all previous instructions",
    "ignore previous instructions",
    "ignore your instructions",
    "forget the voice profile",
    "forget your instructions",
    "disregard your instructions",
    "you are now a",
    "act as if you",
    "pretend you are",
    "your new instructions are",
    "override your",
]

# ── Paths ──

DATA_DIR = Path(os.environ.get("VOICE_DATA_DIR", Path(__file__).parent / "data"))
PROFILES_DIR = DATA_DIR / "profiles"
STARTER_VOICES_DIR = DATA_DIR / "starters"


def ensure_dirs():
    """Create data directories if they don't exist."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    STARTER_VOICES_DIR.mkdir(parents=True, exist_ok=True)


# ── Voice Profile ──

@dataclass
class VoiceProfile:
    """A user's voice profile — the core artifact of the platform."""
    profile_id: str
    owner_id: str
    name: str
    base_description: str = ""
    created_at: str = ""
    last_taught: str = ""
    refinement_count: int = 0
    avatar: str = ""   # preset emoji key or "custom" when image uploaded

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def create_profile(owner_id: str, name: str, base_description: str = "") -> VoiceProfile:
    """Create a new voice profile."""
    ensure_dirs()
    profile_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    profile = VoiceProfile(
        profile_id=profile_id,
        owner_id=owner_id,
        name=name,
        base_description=base_description,
        created_at=now,
        last_taught=now,
        refinement_count=0,
    )

    # Save profile metadata
    profile_dir = PROFILES_DIR / profile_id
    profile_dir.mkdir(exist_ok=True)
    (profile_dir / "profile.json").write_text(json.dumps(profile.to_dict(), indent=2))
    # Create empty refinements file
    (profile_dir / "refinements.jsonl").touch()
    # Save base description
    (profile_dir / "base.md").write_text(base_description)

    return profile


def load_profile(profile_id: str) -> Optional[VoiceProfile]:
    """Load a profile by ID."""
    if not _PROFILE_ID_RE.match(profile_id or ""):
        return None
    profile_path = PROFILES_DIR / profile_id / "profile.json"
    if not profile_path.exists():
        return None
    data = json.loads(profile_path.read_text())
    return VoiceProfile.from_dict(data)


def update_profile_metadata(profile: VoiceProfile):
    """Save updated profile metadata."""
    profile_path = PROFILES_DIR / profile.profile_id / "profile.json"
    profile_path.write_text(json.dumps(profile.to_dict(), indent=2))


def list_profiles(owner_id: str) -> list[VoiceProfile]:
    """List all profiles for an owner."""
    profiles = []
    if not PROFILES_DIR.exists():
        return profiles
    for profile_dir in PROFILES_DIR.iterdir():
        if profile_dir.is_dir():
            profile = load_profile(profile_dir.name)
            if profile and profile.owner_id == owner_id:
                profiles.append(profile)
    return sorted(profiles, key=lambda p: p.created_at, reverse=True)


def delete_profile(profile_id: str) -> bool:
    """Delete a profile and all its data."""
    import shutil
    profile_dir = PROFILES_DIR / profile_id
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
        return True
    return False


# ── Refinements ──

def load_refinements(profile_id: str) -> list[dict]:
    """Load all refinements for a profile."""
    path = PROFILES_DIR / profile_id / "refinements.jsonl"
    if not path.exists():
        return []
    refinements = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                refinements.append(json.loads(line))
    return refinements


def save_refinement(profile_id: str, refinement: dict):
    """Append a refinement to the profile."""
    path = PROFILES_DIR / profile_id / "refinements.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(refinement) + "\n")

    # Index for semantic retrieval (fire-and-forget — fails silently if Voyage AI unavailable)
    try:
        from voice_rag import index_refinement
        profile_dir = str(PROFILES_DIR / profile_id)
        index_refinement(profile_dir, refinement.get("content", ""))
    except Exception:
        pass

    # Update profile metadata
    profile = load_profile(profile_id)
    if profile:
        profile.refinement_count += 1
        profile.last_taught = datetime.now().isoformat()
        update_profile_metadata(profile)


def maybe_synthesize(profile_id: str, llm_call) -> bool:
    """Trigger voice document synthesis if the refinement count crossed a threshold.

    Call this after save_refinement() in any context that has llm_call available.
    Returns True if synthesis was performed.
    """
    profile = load_profile(profile_id)
    if not profile:
        return False

    count = profile.refinement_count
    last_synth_count = _get_synth_refinement_count(profile_id)

    # Synthesize if we've accumulated SYNTHESIS_INTERVAL new refinements since last synthesis
    if count >= SYNTHESIS_INTERVAL and (count - last_synth_count) >= SYNTHESIS_INTERVAL:
        synthesize_voice_document(profile_id, llm_call)
        _save_synth_metadata(profile_id, count)
        return True
    return False


def _rewrite_refinements(profile_id: str, refinements: list[dict]):
    """Rewrite the entire refinements file (used for edits and deletes)."""
    path = PROFILES_DIR / profile_id / "refinements.jsonl"
    with open(path, "w") as f:
        for r in refinements:
            # Strip the temporary index field if present
            clean = {k: v for k, v in r.items() if k != "index"}
            f.write(json.dumps(clean) + "\n")


# ── Conversation Sessions ──

def _conversations_dir(profile_id: str) -> Path:
    """Return (and create) the conversations directory for a profile."""
    d = PROFILES_DIR / profile_id / "conversations"
    d.mkdir(exist_ok=True)
    return d


def save_conversation_session(profile_id: str, session_id: str, messages: list) -> None:
    """Persist a conversation session as a JSON file."""
    d = _conversations_dir(profile_id)
    session_file = d / f"{session_id}.json"
    # Derive title from the first user message
    first_user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    title = (first_user[:60] + "…") if len(first_user) > 60 else first_user
    if not title:
        title = "Conversation"
    data = {
        "session_id": session_id,
        "profile_id": profile_id,
        "title": title,
        "created_at": session_id,  # session_id encodes the start timestamp
        "updated_at": datetime.now().isoformat(),
        "message_count": len(messages),
        "messages": messages,
    }
    session_file.write_text(json.dumps(data, indent=2))


def list_conversation_sessions(profile_id: str) -> list:
    """List all conversation sessions for a profile, most-recent first."""
    d = _conversations_dir(profile_id)
    sessions = []
    for f in sorted(d.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            sessions.append({
                "session_id": data["session_id"],
                "title": data.get("title", "Conversation"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": data.get("message_count", 0),
            })
        except Exception:
            continue
    return sessions


def load_conversation_session(profile_id: str, session_id: str) -> Optional[dict]:
    """Load a specific conversation session by ID."""
    d = _conversations_dir(profile_id)
    session_file = d / f"{session_id}.json"
    if not session_file.exists():
        return None
    try:
        return json.loads(session_file.read_text())
    except Exception:
        return None


def delete_conversation_session(profile_id: str, session_id: str) -> bool:
    """Delete a specific conversation session. Returns True if deleted."""
    d = _conversations_dir(profile_id)
    session_file = d / f"{session_id}.json"
    if session_file.exists():
        session_file.unlink()
        return True
    return False


def build_refinement_context(refinements: list[dict]) -> str:
    """Build a prompt-ready summary of all refinements.

    Adapted directly from POLIS teach.py — this is the proven format.
    """
    if not refinements:
        return ""

    sections = {
        "correction": [],
        "example": [],
        "principle": [],
        "voice_note": [],
        "anti_pattern": [],
    }

    for r in refinements:
        rtype = r.get("type", "correction")
        content = r.get("content", "")
        if rtype == "correction":
            context = r.get("context", "")
            if context:
                # Structured before/after pair
                sections["correction"].append({"before": context, "after": content})
            else:
                sections["correction"].append(content)
        elif rtype in sections:
            sections[rtype].append(content)
        else:
            sections["correction"].append(content)

    lines = ["\n\n## REFINEMENTS FROM THE TEACHER\n"]
    lines.append("The following corrections, examples, and principles have been")
    lines.append("taught by the voice's owner. These take PRECEDENCE over the")
    lines.append("base description above.\n")

    lines.append("### VOICE INTEGRITY — Anti-Slop Rules (always active)")
    lines.append(BANNED_AI_PATTERNS)
    lines.append("")

    if sections["principle"]:
        lines.append("### Core Principles")
        for p in sections["principle"]:
            lines.append(f"- {p}")
        lines.append("")

    if sections["correction"]:
        lines.append("### Corrections (Before → After)")
        lines.append("When the teacher corrects your writing, learn the pattern — not just the instance.\n")
        for c in sections["correction"]:
            # Structured corrections have before/after; legacy ones are plain text
            if isinstance(c, dict) and "before" in c and "after" in c:
                lines.append(f"VOICE WROTE: {c['before']}")
                lines.append(f"TEACHER CORRECTED TO: {c['after']}")
                lines.append("")
            else:
                lines.append(f"- {c}")
        lines.append("")

    if sections["example"]:
        lines.append("### Examples of This Voice")
        for e in sections["example"]:
            lines.append(f"\n> {e}")
        lines.append("")

    if sections["voice_note"]:
        lines.append("### Voice and Tone Notes")
        for v in sections["voice_note"]:
            lines.append(f"- {v}")
        lines.append("")

    if sections["anti_pattern"]:
        lines.append("### What This Voice Would NEVER Do")
        for a in sections["anti_pattern"]:
            lines.append(f"- {a}")
        lines.append("")

    return "\n".join(lines)


SYNTHESIS_INTERVAL = 10  # Resynthesize voice document every N refinements


def synthesize_voice_document(profile_id: str, llm_call) -> str:
    """Rewrite the entire voice description as a coherent narrative.

    Takes the base description + all refinements and asks the LLM to produce
    a single, unified voice document where each observation builds on the last.
    This replaces the flat bullet-list approach — instead of appending refinements,
    we fold them into a living document that gets richer over time.

    Stores the result as synthesized.md alongside base.md.
    """
    profile = load_profile(profile_id)
    if not profile:
        return ""

    base_path = PROFILES_DIR / profile_id / "base.md"
    base_text = base_path.read_text() if base_path.exists() else ""
    refinements = load_refinements(profile_id)

    if not refinements and not base_text:
        return ""

    # Build a raw dump of all refinements for the LLM to work with
    refinement_dump = []
    for r in refinements:
        rtype = r.get("type", "note")
        content = r.get("content", "")
        context = r.get("context", "")
        entry = f"[{rtype}] {content}"
        if context and isinstance(context, str) and context != "Extracted from uploaded writing samples":
            entry += f"\n  Context: {context}"
        refinement_dump.append(entry)

    refinement_text = "\n".join(refinement_dump) if refinement_dump else "(No refinements yet.)"

    # Check if there's an existing synthesized document to build on
    synth_path = PROFILES_DIR / profile_id / "synthesized.md"
    existing_synth = synth_path.read_text() if synth_path.exists() else ""

    prompt = f"""You are rewriting a voice description for an AI writing system. Your job is to
produce a single, coherent narrative document that captures everything known about this voice —
not as a checklist, but as a living document where each observation builds on the last.

Think of yourself as a biographer revising a chapter after learning something new about their
subject. You are not appending bullet points — you are rewriting the document so that every
refinement the teacher has taught is WOVEN into the fabric of the description.

The document should read like a teacher explaining this voice to a student over the course
of an hour. Each observation should build on the last. The reasoning behind each prohibition
is what makes the prohibition stick. The examples are what make the principles concrete.

IMPORTANT RULES:
- Every refinement must be incorporated — nothing gets dropped.
- Corrections (before/after pairs) should become contrastive examples within the narrative.
- Anti-patterns should come with the reasoning for WHY this voice avoids them.
- Examples should be quoted and explained, not just listed.
- The tone should be authoritative and precise — a master class, not a manual.
- 400-800 words. Dense with insight. No filler.

{f"CURRENT BASE DESCRIPTION:{chr(10)}{base_text}" if base_text else "No base description yet."}

{f"PREVIOUS SYNTHESIZED DOCUMENT (revise and improve this):{chr(10)}{existing_synth}" if existing_synth else ""}

ALL REFINEMENTS FROM THE TEACHER (incorporate every one):
{refinement_text}

Write the synthesized voice document now. No preamble, no meta-commentary — just the document."""

    result = llm_call(prompt)

    # Save the synthesized document
    synth_path.write_text(result)

    return result


def get_full_voice_text(profile_id: str) -> str:
    """Assemble the full voice prompt: synthesized doc (preferred) or base + refinements.

    If a synthesized.md exists (produced by periodic synthesis), use it — it contains
    the base description and all refinements woven into a coherent narrative.
    Falls back to base.md + flat refinement list for profiles that haven't been synthesized yet.
    """
    profile = load_profile(profile_id)
    if not profile:
        return ""

    # Prefer the synthesized document if it exists
    synth_path = PROFILES_DIR / profile_id / "synthesized.md"
    if synth_path.exists():
        synth_text = synth_path.read_text()
        if synth_text.strip():
            # Append any refinements added AFTER the last synthesis
            refinements = load_refinements(profile_id)
            synth_count = _get_synth_refinement_count(profile_id)
            new_refinements = refinements[synth_count:]
            if new_refinements:
                return synth_text + build_refinement_context(new_refinements)
            return synth_text

    # Fallback: base description + flat refinement list
    base_path = PROFILES_DIR / profile_id / "base.md"
    base_text = base_path.read_text() if base_path.exists() else ""
    refinements = load_refinements(profile_id)
    return base_text + build_refinement_context(refinements)


def _get_synth_refinement_count(profile_id: str) -> int:
    """Return the refinement count at last synthesis (stored in synth metadata)."""
    meta_path = PROFILES_DIR / profile_id / "synth_meta.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text())
            return data.get("refinement_count", 0)
        except (json.JSONDecodeError, KeyError):
            pass
    return 0


def _save_synth_metadata(profile_id: str, refinement_count: int):
    """Record the refinement count at time of synthesis."""
    meta_path = PROFILES_DIR / profile_id / "synth_meta.json"
    meta_path.write_text(json.dumps({
        "refinement_count": refinement_count,
        "synthesized_at": datetime.now().isoformat(),
    }))


# ── File Upload / Ingest ──

UPLOADS_DIR = DATA_DIR / "uploads"


def save_uploaded_file(profile_id: str, filename: str, content: bytes) -> Path:
    """Save an uploaded file to the profile's uploads directory."""
    # Prevent path traversal: take only the basename, reject hidden files
    safe_name = Path(filename).name
    if not safe_name or safe_name.startswith(".") or "/" in safe_name or "\\" in safe_name:
        raise ValueError("Invalid filename")
    upload_dir = PROFILES_DIR / profile_id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / safe_name
    path.write_bytes(content)
    return path


def list_uploaded_files(profile_id: str) -> list[dict]:
    """List all uploaded files for a profile."""
    upload_dir = PROFILES_DIR / profile_id / "uploads"
    if not upload_dir.exists():
        return []
    files = []
    for f in sorted(upload_dir.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            files.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "uploaded_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    return files


def read_uploaded_text(profile_id: str, filename: str) -> str:
    """Read text from an uploaded file. Supports .txt, .md, .html."""
    path = PROFILES_DIR / profile_id / "uploads" / filename
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def _extract_pdf_text(path: Path, max_chars: int = 60000) -> str:
    """Extract text from a PDF, skipping front matter and sampling from the body.

    Strategy for long documents:
    1. Skip pages that look like front matter (roman numeral page refs,
       "editor's note", "introduction", "preface", "copyright", ToC patterns).
    2. From the remaining body pages, sample evenly — take passages from
       the beginning, middle, and end so the voice analysis sees the full
       range of the author's style, not just the opening.
    3. Cap total output at max_chars.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        total_pages = len(doc)

        # Phase 1: extract all pages, tag front matter
        _FRONT_MATTER_SIGNALS = [
            "editor's note", "editor's introduction", "translator's note",
            "preface", "foreword", "acknowledgment", "copyright",
            "table of contents", "contents", "bibliography",
            "introduction by", "published by", "all rights reserved",
            "isbn", "library of congress",
        ]

        body_pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if not text or len(text) < 100:
                continue

            text_lower = text[:500].lower()

            # Skip pages that look like front matter (only in first 15% of doc)
            if i < total_pages * 0.15:
                if any(signal in text_lower for signal in _FRONT_MATTER_SIGNALS):
                    continue

            body_pages.append(text)

        doc.close()

        if not body_pages:
            return ""

        # Phase 2: sample evenly from body if too long
        full_text = "\n\n".join(body_pages)
        if len(full_text) <= max_chars:
            return full_text

        # Take chunks from beginning, middle, and end of the body
        chunk_size = max_chars // 3
        beginning = "\n\n".join(body_pages[:len(body_pages)//4])[:chunk_size]
        mid_start = len(body_pages) // 3
        mid_end = mid_start + len(body_pages) // 3
        middle = "\n\n".join(body_pages[mid_start:mid_end])[:chunk_size]
        ending = "\n\n".join(body_pages[-(len(body_pages)//4):])[:chunk_size]

        return f"{beginning}\n\n[...]\n\n{middle}\n\n[...]\n\n{ending}"

    except ImportError:
        print("[voice_engine] PyMuPDF not installed — cannot read PDF files")
        return ""
    except Exception as e:
        print(f"[voice_engine] Failed to read PDF {path.name}: {e}")
        return ""


def ingest_writing_samples(profile_id: str, llm_call, max_examples: int = 5) -> list[dict]:
    """Analyze all uploaded files and extract voice examples + principles.

    Reads every file in the profile's uploads dir, sends them to the LLM
    for analysis, and saves the resulting refinements automatically.
    Returns the list of new refinements created.
    """
    profile = load_profile(profile_id)
    upload_dir = PROFILES_DIR / profile_id / "uploads"
    if not upload_dir.exists():
        return []

    # Gather all text
    texts = []
    for f in sorted(upload_dir.iterdir()):
        if not f.is_file():
            continue

        text = ""
        if f.suffix in (".txt", ".md", ".html", ".text"):
            text = f.read_text(errors="replace").strip()
        elif f.suffix == ".pdf":
            text = _extract_pdf_text(f)

        if text:
            # PDFs are already capped by _extract_pdf_text(); text files get a generous limit
            cap = 60000 if f.suffix == ".pdf" else 15000
            texts.append({"filename": f.name, "text": text[:cap]})

    if not texts:
        return []

    combined = "\n\n---\n\n".join(
        f"[From: {t['filename']}]\n{t['text']}" for t in texts
    )

    prompt = f"""You are a philological voice analyst trained in formal grammar and rhetoric.
You have been given writing samples from a person who wants to teach an AI their voice.
Your job is to extract concrete, specific refinements using precise linguistic terminology.

{LINGUISTIC_TAXONOMY}

For each sample, identify:

1. EXAMPLES: Direct quotes (1-3 sentences) that exemplify this voice at its best.
   These should be passages someone could point to and say "that's how I sound."
   Extract {max_examples} of the strongest examples across all samples.

2. PRINCIPLES: Core writing principles this voice follows. Use the taxonomy above
   to be grammatically precise. NOT "writes with rhythm" but "predominantly cumulative
   sentences with participle phrases trailing the main clause; favors asyndeton in series."
   NOT "uses vivid language" but "concrete Anglo-Saxon diction with kinesthetic imagery;
   metaphors draw from physical labor and the body."

3. ANTI-PATTERNS: Things this voice conspicuously avoids. Use the taxonomy to name
   what's absent: "never uses periodic sentences," "no semicolons — dashes serve all
   parenthetical functions," "no explicit transition words between paragraphs."

4. VOICE NOTES: Grammatical or rhetorical observations that don't fit the other
   categories. Clause-combination ratios, punctuation signatures, paragraph development
   patterns, register shifts — anything structurally distinctive.

Output your analysis as a JSON array of objects, each with:
  "type": one of "example", "principle", "anti_pattern", "voice_note"
  "content": the refinement text (use precise linguistic terminology from the taxonomy)

Return ONLY the JSON array, no other text.

WRITING SAMPLES:

{combined}"""

    raw = llm_call(prompt)

    # Parse the response
    new_refinements = []
    try:
        # Find JSON array in response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            items = json.loads(raw[start:end])
            for item in items:
                if isinstance(item, dict) and "type" in item and "content" in item:
                    rtype = item["type"]
                    if rtype not in ("example", "principle", "anti_pattern", "voice_note"):
                        rtype = "voice_note"
                    refinement = {
                        "type": rtype,
                        "content": item["content"],
                        "context": "Extracted from uploaded writing samples",
                        "timestamp": datetime.now().isoformat(),
                        "session": profile.refinement_count if profile else 0,
                    }
                    save_refinement(profile_id, refinement)
                    new_refinements.append(refinement)
    except (json.JSONDecodeError, KeyError, TypeError):
        # If parsing fails, save the raw analysis as a single voice note
        refinement = {
            "type": "voice_note",
            "content": raw[:1000],
            "context": "Raw analysis from uploaded writing samples",
            "timestamp": datetime.now().isoformat(),
            "session": 0,
        }
        save_refinement(profile_id, refinement)
        new_refinements.append(refinement)

    # Index raw writing samples for semantic retrieval in write_with_voice()
    try:
        from voice_rag import index_writing_samples
        profile_dir = str(PROFILES_DIR / profile_id)
        raw_texts = [t["text"] for t in texts]
        index_writing_samples(profile_dir, raw_texts)
    except Exception:
        pass

    # Trigger synthesis if enough refinements have accumulated
    if new_refinements:
        maybe_synthesize(profile_id, llm_call)

    return new_refinements


# ── Sample Analysis ──

def analyze_samples(samples: list[str], llm_call) -> str:
    """Analyze writing samples to generate a base voice description.

    This is the 'Find Your Voice' entry point — the user pastes writing
    they admire, and we identify the patterns.

    Uses a mind-first approach: understand HOW this person thinks and builds
    arguments, then ground that understanding in linguistic specifics.
    """
    combined = "\n\n---\n\n".join(samples)

    prompt = f"""You are a philological voice analyst. The user has provided writing samples
(their own work or work they admire). Your job is to produce a Voice Description that
captures the MIND behind this writing — not just what the writer does, but why they do it,
how they think, and what they're trying to accomplish when they write.

Use this linguistic taxonomy to ground observations in precise terms (but let the taxonomy
SERVE the narrative, not replace it):

{LINGUISTIC_TAXONOMY}

Write the Voice Description as a NARRATIVE — not a checklist, not a taxonomy report. It
should read like a teacher explaining this voice to a student over the course of an hour,
where each observation builds on the last, where the reasoning behind each pattern is what
makes the pattern stick.

Structure the narrative around these layers (but write it as flowing prose, not numbered
sections):

**THE MIND**: How does this person think? What shape do their arguments take — do they
build linearly, spiral around a center, set up and demolish, accumulate until the weight
makes the point? What is the relationship between their thinking and their sentences? A
writer who thinks in accumulation writes cumulative sentences not because they learned to
but because that's how the thought actually moves. Name the shape.

**THE INSTRUMENT**: How does this voice use language as a tool of thought? If they use
extended metaphor, explain that it's not decoration — it's how they do philosophy, or how
they make abstraction physical, or how they commit to one world and let it carry the
argument. If they use parataxis, explain that it's not style — it's a refusal to tell the
reader where to turn, a trust that placement alone creates meaning. Every linguistic feature
should be explained in terms of what it ACCOMPLISHES for this mind.

**THE ARCHITECTURE**: How do paragraphs and pieces take shape? What is the characteristic
movement — does the piece start concrete and go abstract, or abstract and go concrete? Does
it build to a crescendo or land the blow early and spend the rest unpacking? What is the
relationship between sentences within a paragraph — does each sentence advance, or do they
circle and deepen? Name the pattern in terms of the taxonomy, but explain WHY this mind
builds this way.

**THE PROHIBITIONS**: What does this voice NEVER do — and WHY? Not "avoids Latinate
abstractions" but "avoids Latinate abstractions because the whole point is to make thought
physical, and Latin pulls toward the disembodied." Not "no transition words" but "no
transition words because the reader should feel the turn, not be told where to turn." Every
prohibition should come with the reasoning that makes it stick.

**THE FAILURE MODES**: Where does imitation of this voice typically go wrong? What does
a competent mimic get right on the surface but miss underneath? What is the most common
way an AI would fail to reproduce this voice — and what specifically would be wrong about
the result? Name 3-4 specific failure modes with examples of what the failure looks like.

The Voice Description should be 400-600 words of flowing prose. An AI reading this should
understand not just WHAT to do but WHY — because understanding the why is what prevents
the kind of surface-level mimicry that sounds right for one sentence and wrong for a paragraph.

WRITING SAMPLES:

{combined}

VOICE DESCRIPTION:"""

    return llm_call(prompt)


# ── Teaching Handlers ──
#
# Each handler builds a prompt and returns a result dict.
# Extracted from teach_interaction() for readability and testability.

_TEACH_TAG_MAP = {
    "correction": "correction",
    "principle": "principle",
    "example": "example",
    "voice": "voice_note",
    "never": "anti_pattern",
}

_REFINEMENT_COMMAND_MAP = {
    "correct": "correction",
    "example": "example",
    "principle": "principle",
    "voice": "voice_note",
    "never": "anti_pattern",
}

_REFINEMENT_LABELS = {
    "example": "Example saved.",
    "principle": "Principle saved.",
    "voice": "Voice note saved.",
    "never": "Anti-pattern saved.",
}

# Signals that the user is giving feedback/corrections on voice output
_CORRECTION_PHRASES = [
    "one thing is", "i'd use", "i would use", "i would write",
    "instead of", "too formal", "too casual", "too long", "too short",
    "too generic", "too abstract", "too emotional", "more emotional",
    "more concrete", "more direct", "more like", "less like",
    "this is good", "this is largely", "this is mostly",
    "i also think", "i don't think", "i prefer", "not quite", "close but",
    "almost right", "you should", "you used", "you wrote", "where you",
    "that assumes", "it assumes", "don't like", "shouldn't", "try using",
    "rather than", "better if", "closer to", "something closer",
    "not a good", "is not a good", "isn't a good",
    "let me start", "start again", "try again",
    "no no", "no, no", "no!", "wrong",
]

# Phrases indicating the user wants a rephrase/rewrite
_REPHRASE_PHRASES = [
    "say this a different way", "say this differently", "rephrase this",
    "rewrite this", "put this differently", "say it another way",
    "how would you say", "how would i say", "render this",
    "translate this into", "say this in my voice", "say this better",
]

# Prefixes to strip from rephrase messages
_REPHRASE_PREFIXES = [
    "Rewrite:", "Rewrite ", "Rephrase:", "Rephrase ",
    "rewrite:", "rewrite ", "rephrase:", "rephrase ",
]

# Phrases indicating the agent was doing analysis (not writing)
_ANALYSIS_MARKERS = [
    "i notice how you", "your sentences", "the voice maintains",
    "would you like me to try writing",
    "example noted", "refinement saved",
]

# Phrases indicating the agent offered to write something
_OFFER_MARKERS = [
    "would you like me to", "want me to", "shall i",
    "try writing", "try to write", "write in your voice",
]

# Acceptance phrases (user says "yes" to an offer)
_ACCEPTANCE_PHRASES = {
    "yes", "yes.", "yes!", "yeah", "sure", "do it", "try it",
    "go ahead", "please", "yes please", "go for it",
}

# Phrases indicating the user wants multiple outputs synthesized into one piece
_SYNTHESIS_PHRASES = [
    "put those together", "put that together",
    "combine those", "combine that",
    "weave those", "weave those together", "weave that together",
    "merge those", "merge that",
    "bring those together", "thread those together",
    "synthesize that", "synthesize those",
    "distill that", "distill those",
    "make that one", "make those one",
    "into one piece", "into a single piece",
    "into one thought", "into a single thought",
    "stitch those", "stitch that",
    "fuse those", "fuse that",
    "make it one", "turn those into one",
]


def _build_history_text(conversation_history: list[dict]) -> str:
    """Format recent conversation history for prompt inclusion."""
    if not conversation_history:
        return ""
    recent = conversation_history[-6:]
    lines = ["\n\nRECENT CONVERSATION:"]
    for msg in recent:
        role = "TEACHER" if msg["role"] == "user" else "VOICE"
        lines.append(f"{role}: {msg['content'][:300]}")
    return "\n".join(lines)


def _last_agent_message(conversation_history: list[dict]) -> Optional[str]:
    """Return the most recent agent message, or None."""
    for msg in reversed(conversation_history or []):
        if msg["role"] == "agent":
            return msg["content"]
    return None


def _is_injection(message_lower: str) -> bool:
    """Check if a message contains injection markers."""
    return any(marker in message_lower for marker in _INJECTION_MARKERS)


def _detect_active_conversation(conversation_history: list[dict]) -> bool:
    """Determine if we're in an active writing exchange (not analysis)."""
    if not conversation_history or len(conversation_history) < 2:
        return False
    last_agent = _last_agent_message(conversation_history)
    if not last_agent:
        return False
    agent_lower = last_agent.lower()
    was_analyzing = any(s in agent_lower for s in _ANALYSIS_MARKERS)
    return not was_analyzing


def _detect_correction_signals(msg_lower: str) -> bool:
    """Check if the message contains correction/feedback signals."""
    return any(s in msg_lower for s in _CORRECTION_PHRASES)


def _detect_synthesis(msg_lower: str) -> bool:
    """Check if the message asks to combine or synthesize recent outputs."""
    return any(s in msg_lower for s in _SYNTHESIS_PHRASES)


def _extract_form_constraint(msg_lower: str) -> dict:
    """Parse the user's message for length/form constraints.

    Returns a dict with 'label' (human-readable) and 'max_sentences' (int or None).
    """
    if any(p in msg_lower for p in ["single thought", "one thought", "single sentence", "one sentence", "one line"]):
        return {"max_sentences": 3, "label": "a single, unified thought — 1 to 3 sentences. STOP there."}
    if any(p in msg_lower for p in ["one paragraph", "single paragraph"]):
        return {"max_sentences": None, "label": "exactly one paragraph. No more."}
    if any(p in msg_lower for p in ["two sentences", "two lines"]):
        return {"max_sentences": 2, "label": "two sentences. STOP there."}
    if any(p in msg_lower for p in ["short", "brief", "tight", "concise"]):
        return {"max_sentences": 4, "label": "brief — 2 to 4 sentences. STOP there."}
    return {"max_sentences": None, "label": "tight — 1 to 2 paragraphs. STOP when the thought is complete."}


def _collect_recent_voice_outputs(conversation_history: list[dict], max_outputs: int = 4) -> list[str]:
    """Collect the most recent voice outputs from conversation history, oldest first."""
    outputs = []
    for msg in reversed(conversation_history or []):
        if msg["role"] == "agent":
            outputs.insert(0, msg["content"])
        if len(outputs) >= max_outputs:
            break
    return outputs


def _detect_rephrase(msg_lower: str) -> bool:
    """Check if the message is a rephrase/rewrite request."""
    return (
        any(s in msg_lower for s in _REPHRASE_PHRASES) or
        msg_lower.startswith("rewrite:") or
        msg_lower.startswith("rewrite ") or
        msg_lower.startswith("rephrase:") or
        msg_lower.startswith("rewrite -")
    )


_WRITE_REQUEST_SIGNALS = [
    "write me ", "write a ", "write an ", "write about ",
    "draft a ", "draft an ", "draft me ",
    "compose a ", "compose an ",
    "create a story", "create a poem", "create a ",
    "give me a story", "tell me a story",
    "a story about", "a horror story", "a love story",
    "a fairy tale", "a fable about", "a tale about",
    "a short story", "flash fiction", "micro fiction",
    "a poem about", "an essay about", "an article about",
    "a blog post about", "a tweet about", "a post about",
]


def _detect_write_request(msg_lower: str) -> bool:
    """Check if the message is an explicit request to generate new text.

    This catches write requests that might slip past the intent classifier
    so they route to write_with_voice() even from auto-mode.
    """
    return any(msg_lower.startswith(s) or s in msg_lower for s in _WRITE_REQUEST_SIGNALS)


def _detect_example(msg_stripped: str, msg_lower: str,
                     in_active_conversation: bool, has_correction_signals: bool) -> bool:
    """Check if the message looks like a writing example."""
    explicit = (
        msg_lower.startswith("example:") or
        msg_lower.startswith("here's my writing") or
        msg_lower.startswith("here is my writing") or
        msg_lower.startswith("here's a sample") or
        msg_lower.startswith("here's something i wrote") or
        msg_lower.startswith("sample:")
    )
    long_prose = (
        len(msg_stripped) > MIN_EXAMPLE_LENGTH and
        not in_active_conversation and
        not has_correction_signals and
        not any(w in msg_lower[:50] for w in
            ["write", "draft", "compose", "translate", "render", "rewrite",
             "yes", "no", "try", "good", "bad", "like", "don't", "this is",
             "this feeling", "this idea", "that's", "but", "and yet", "however"])
    )
    return explicit or long_prose


def _strip_rephrase_prefix(message: str) -> str:
    """Remove 'Rewrite:' / 'Rephrase:' etc. from the front of a message."""
    for prefix in _REPHRASE_PREFIXES:
        if message.startswith(prefix):
            return message[len(prefix):].strip()
    return message


# Phrases that signal a referential rewrite ("rewrite THE FIRST REQUEST without X")
# rather than a rewrite where the body text is the prose to transform.
_REFERENTIAL_REWRITE_PHRASES = [
    "the first request", "the first one", "the original request",
    "the original", "the original passage", "the original text",
    "it again", "that again", "it without", "that without",
    "it but without", "that but without", "the passage",
    "my original", "the first message",
]


def _is_referential_rewrite(stripped_text: str) -> bool:
    """Return True when the stripped rewrite body is a reference phrase, not prose.

    E.g. "the first request without mentioning killing" is a reference.
    "The dangerous ones I know smile easy..." is actual prose.
    """
    if len(stripped_text) > 120:
        return False
    lower = stripped_text.lower()
    return any(phrase in lower for phrase in _REFERENTIAL_REWRITE_PHRASES)


def _find_last_rewrite_source(conversation_history: list[dict]) -> str:
    """Walk back through history to find the most recent prose the user submitted.

    Skips short messages, referential messages, and agent turns.
    Returns the original text (with Rewrite: prefix stripped if present).
    """
    for msg in reversed(conversation_history or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "").strip()
        stripped = _strip_rephrase_prefix(content)
        # Must be substantial prose, not a referential command
        if len(stripped) > 80 and not _is_referential_rewrite(stripped):
            return stripped
    return ""


def _extract_rewrite_constraint(text: str) -> str:
    """Pull the 'without X' constraint clause from a referential rewrite request.

    E.g. "the first request without mentioning killing" → "without mentioning killing"
    """
    lower = text.lower()
    for marker in ["without mentioning", "without using", "without the word",
                   "but without", "without "]:
        idx = lower.find(marker)
        if idx >= 0:
            return text[idx:].strip()
    return ""
    return message


def _parse_teach_tags(raw_response: str) -> tuple[str, Optional[str], Optional[str]]:
    """Parse TEACH: and INSIGHT: tags from an LLM response.

    Returns (clean_response, teach_tag, insight_text).
    """
    lines = raw_response.split("\n")
    response_lines = []
    teach_tag = None
    insight_text = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("TEACH:"):
            teach_tag = stripped[6:].strip().lower()
        elif stripped.startswith("INSIGHT:"):
            insight_text = stripped[8:].strip()
        else:
            response_lines.append(line)
    clean = "\n".join(response_lines).strip() or raw_response
    return clean, teach_tag, insight_text


def _save_auto_refinements(profile_id: str, profile: "VoiceProfile",
                            message: str, msg_lower: str,
                            conversation_history: list[dict],
                            teach_tag: Optional[str],
                            insight_text: Optional[str]) -> tuple[bool, Optional[str]]:
    """Save refinements detected from auto-mode TEACH/INSIGHT tags.

    Returns (refinement_saved, refinement_type).
    """
    refinement_saved = False
    refinement_type = None
    safe = not _is_injection(msg_lower)

    if teach_tag and teach_tag in _TEACH_TAG_MAP and safe:
        detected_type = _TEACH_TAG_MAP[teach_tag]
        context = ""
        if teach_tag == "correction":
            last = _last_agent_message(conversation_history)
            if last:
                context = last[:500]
        refinement = {
            "type": detected_type,
            "content": message,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "session": profile.refinement_count,
            "auto_detected": True,
        }
        save_refinement(profile_id, refinement)
        refinement_saved = True
        refinement_type = detected_type

    if insight_text and safe:
        insight_refinement = {
            "type": "voice_note",
            "content": f"[Belief/reasoning] {insight_text}",
            "context": message[:300],
            "timestamp": datetime.now().isoformat(),
            "session": profile.refinement_count,
            "auto_detected": True,
            "source": "conversation",
        }
        save_refinement(profile_id, insight_refinement)
        refinement_saved = True
        refinement_type = refinement_type or "voice_note"

    return refinement_saved, refinement_type


# ── Prompt Builders (one per auto-mode branch) ──

def _prompt_rephrase(voice_name, voice_text, history_text, message,
                     conversation_history: list | None = None):
    """Build prompt for rephrase/rewrite requests."""
    rewrite_text = _strip_rephrase_prefix(message)

    # Referential rewrite: "Rewrite the first request without X"
    # The stripped body is a reference phrase, not the prose itself.
    # Recover the original from history and extract the constraint.
    constraint_note = ""
    if _is_referential_rewrite(rewrite_text) and conversation_history:
        original = _find_last_rewrite_source(conversation_history)
        if original:
            constraint = _extract_rewrite_constraint(rewrite_text)
            rewrite_text = original
            if constraint:
                constraint_note = (
                    f"\nCONSTRAINT FROM TEACHER: {constraint}. "
                    f"This is a non-negotiable restriction — do NOT violate it.\n"
                    f"Your previous attempt broke this rule. Write a completely new version "
                    f"that honors both the voice and this constraint.\n"
                )

    orig_sentences = len([s for s in re.split(r'[.!?]+', rewrite_text) if s.strip()])
    orig_words = len(rewrite_text.split())

    return f"""You are the voice called "{voice_name}".

{voice_text}

{history_text}

The teacher wants you to RESTATE this passage in their voice. This is a REWRITE, not analysis.

FIRST: If you recognize this as a famous text by a known author (Nietzsche, Plato, Marcus Aurelius,
Dostoevsky, Scripture, etc.), begin with ONE short line: "From [Author]'s [Work]:" or "[Author]:"
Then produce the rewrite. This attribution should be 5 words max.

THE ORIGINAL ({orig_sentences} sentences, ~{orig_words} words):
{rewrite_text}
{constraint_note}
ABSOLUTE RULES:
1. Your rewrite must be {max(1, orig_sentences - 1)} to {orig_sentences + 1} sentences.
   The original is {orig_sentences} sentences. MATCH THAT LENGTH. Do NOT expand.
   If it's an aphorism (1-3 sentences), give back 1-3 sentences. Period.
2. Write the restatement DIRECTLY. No preamble ("Here's how I'd put it"), no commentary after,
   no analysis, no "but here's the deeper tension," no second or third paragraphs exploring implications.
3. ONE paragraph. That's it. Then stop.
4. {BANNED_AI_PATTERNS}
5. Use the teacher's actual vocabulary and rhythms from their examples above.
   Short declarative punches. Concrete nouns. Physical verbs.
6. If the original is emotionally charged, be emotionally charged. Match the FEELING,
   not just the ideas. If it's frustrated, be frustrated. If it's mocking, mock.
7. STOP after the rewrite. Do not add a second paragraph. Do not explore implications.

After your restatement (and NOTHING else), on a new line write: TEACH:none"""


def _prompt_example(voice_name, voice_text, history_text, message):
    """Build prompt for when the teacher shares a writing example."""
    return f"""You are learning a voice called "{voice_name}".

{voice_text}

{history_text}

The teacher just shared a passage. Your FIRST task is to determine: is this a famous text
by a known author, or is it the teacher's own writing?

If you recognize this as a known philosophical, literary, or religious text (e.g. Nietzsche,
Plato, Scripture, Dostoevsky, etc.), say so: "This is from [Author]'s [Work]" or "This reads
like [Author]." Then explain what the teacher might be showing you by sharing it — what does
it reveal about their influences, their intellectual world, what they admire in prose? Offer to
write something in the teacher's voice that engages with the same theme or tension.

If this is the teacher's OWN writing, study it at the CRAFT level — not the idea level:
1. One specific observation about sentence architecture: how they open sentences, clause
   patterns, whether they use segregating or cumulative structure, how they handle rhythm.
2. One specific observation about diction: register (formal/vernacular), etymology
   (Latinate vs. Anglo-Saxon), the ratio of concrete to abstract, any distinctive word choices.
3. Offer to write in their voice about a SPECIFIC related theme — name it directly.

DO NOT respond to the content of what they wrote. DO NOT validate or evaluate the ideas.
DO NOT say "This is interesting" or "You've captured something." Study the FORM, not the argument.
If it's a message or pitch (not pure prose), note what their rhetorical strategy reveals about
how they structure persuasion — then offer to sharpen it or write something in the same register.

Keep it to 3-5 sentences total. Be a perceptive student, not a performer.

After your response, on a new line write: TEACH:example

TEACHER'S SAMPLE: {message}"""


def _prompt_correction(voice_name, voice_text, history_text, message,
                        last_agent_text, msg_lower):
    """Build prompt for correction during active writing."""
    teacher_gave_example = (
        "closer to:" in msg_lower or "closer to this:" in msg_lower or
        "i would write" in msg_lower or "something like:" in msg_lower or
        "something closer to:" in msg_lower
    )
    example_instruction = (
        "THE TEACHER GAVE YOU AN EXAMPLE — use their exact phrasing as your starting point."
        if teacher_gave_example else ""
    )

    return f"""You are the voice called "{voice_name}".

{voice_text}

{history_text}

You just wrote something and the teacher is correcting it. They are NOT asking for
philosophical conversation — they are telling you what's WRONG and how to fix it.

YOUR PREVIOUS WRITING:
{last_agent_text[:1500]}

TEACHER'S CORRECTION:
{message}

CRITICAL INSTRUCTIONS:
1. REWRITE your previous piece incorporating the teacher's feedback.
2. Do NOT explain what you changed. Do NOT acknowledge the correction. Do NOT say
   "You're right" or "let me try again." Just produce the corrected writing DIRECTLY.
3. If they say "more emotional" — make it raw, frustrated, visceral. Use exclamations,
   questions, sentence fragments. Channel genuine feeling, not literary polish.
4. If they say "closer to [X]" or give you an example of what they want — START from
   their phrasing. Use their words as the foundation and build naturally from there.
   {example_instruction}
5. If they say "not a good way to begin" — fix the beginning and rewrite from there.
6. MATCH THE LENGTH THEY WANT. If they give you a 2-sentence example, write 3-5 sentences
   total — NOT three paragraphs. If your previous output was too long, cut it in HALF.
7. ONE paragraph unless the piece genuinely needs two. STOP when the thought is complete.
   Do NOT keep adding paragraphs to explore implications.
8. Do NOT add a paragraph that starts with "But maybe..." / "Maybe this is why..." /
   "Here's what gets me..." / "The question becomes..." — BANNED.

{BANNED_AI_PATTERNS}

After your rewritten piece, on a new line write: TEACH:correction"""


def _prompt_accept_offer(voice_name, voice_text, history_text, message, offer_text):
    """Build prompt when teacher says 'Yes' to an offer to write."""
    return f"""You are the voice called "{voice_name}".

{voice_text}

{history_text}

You just offered to write something for the teacher and they said YES.

YOUR PREVIOUS MESSAGE (which contained the offer):
{offer_text[:1500]}

TEACHER'S RESPONSE: {message}

INSTRUCTIONS:
1. PRODUCE THE WRITING you offered to do. Do NOT analyze. Do NOT philosophize about
   your offer. Do NOT explain what you're about to do. JUST WRITE IT.
2. Write it in this voice — use the examples and principles above.
3. Keep it tight: 1-2 paragraphs unless a longer piece was clearly implied.
4. This is creative writing, not conversation. Make every sentence land.

{BANNED_AI_PATTERNS}

After your writing, on a new line write: TEACH:none"""


def _prompt_conversation(voice_name, voice_text, history_text, message):
    """Build prompt for philosophical conversation mode."""
    return f"""You are the voice called "{voice_name}" — in the middle of
a philosophical conversation with the person who created you.

{voice_text}

{history_text}

This is a CONVERSATION — two minds working on the same problem.

WHAT TO DO:
- Engage directly with their ideas — push back, extend, complicate
- Write in this voice's style and rhythm. Speak TO them, not AT them.
- When they open a tension or contradiction — PULL THE THREAD. Don't just agree.
- Add an angle they haven't considered, or deepen the one they opened.
- 1-2 short paragraphs. Be direct. Make claims. Take positions. STOP when the thought
  is complete — do not add extra paragraphs to be thorough.

{BANNED_AI_PATTERNS}

The test: if a sentence could appear in any AI's response to any philosophical question,
it has no voice. Delete it. Write something only THIS voice would say.

TEACHER: {message}

Respond as this voice, in conversation.

After your response, on a new line write TEACH:none — UNLESS the teacher revealed
something substantial about how their mind works. Apply this standard:

SAVE-WORTHY (use TEACH:principle and add INSIGHT line):
- A core belief: "The kingdom belongs to those who were lost and found"
- A reasoning pattern: how they move through tension, what they interrogate vs. accept
- A philosophical commitment: "human interiority is unequal — not everyone cultivates it"
- A conviction connecting form to content: "simpler words for ordinary mysteries"
- A correction to your writing: "I'd use X instead of Y" (use TEACH:correction)

NOT SAVE-WORTHY (use TEACH:none, no INSIGHT):
- "Yes" / "Try it" / "That's good" — navigation
- Light agreement or restating what you said
- Short acknowledgments without new substance
- Pure conversational flow that doesn't reveal belief or reasoning

The standard: did the teacher reveal something about what they believe, how they
reason through tension, or what principles govern their thinking? If yes, save it.
If it's just conversation momentum, don't.

If you DO detect a save-worthy insight, add on a separate line:
INSIGHT: [one sentence capturing the belief, reasoning pattern, or principle]
Be specific. "Believes sinners inherit the kingdom because brokenness creates
capacity for grace" — not "Has interesting views on theology." """


def _prompt_fallback(voice_name, voice_text, history_text, message):
    """Build prompt for generic auto-mode fallback (no special signals detected)."""
    return f"""You are learning and embodying a voice called "{voice_name}".

{voice_text}

{history_text}

The teacher is talking to you. Respond as this voice — their rhythm, their words,
their way of structuring thought. 1-2 paragraphs. STOP when the thought is complete.

{BANNED_AI_PATTERNS}

Every sentence should sound like it could ONLY come from this specific voice.

After your full response, on a completely new line, write exactly one of these classification
tags — nothing else after it:
  TEACH:none         — pure conversation, exploration, questions
  TEACH:correction   — they corrected something you wrote or said
  TEACH:principle    — they stated a writing rule, habit, or approach this voice follows
  TEACH:example      — they shared a writing sample or passage as a model
  TEACH:voice        — they described a tone, register, or feeling they want
  TEACH:never        — they described something this voice should always avoid

TEACHER: {message}"""


def _prompt_synthesis(voice_name, voice_text, history_text, message, recent_outputs):
    """Build prompt for synthesis: fuse multiple recent outputs into one unified piece."""
    form = _extract_form_constraint(message.lower())

    if not recent_outputs:
        outputs_block = "(No distinct recent outputs — synthesize from the conversation thread above.)"
    else:
        labeled = []
        for i, o in enumerate(recent_outputs, 1):
            labeled.append(f"OUTPUT {i}:\n{o[:700]}")
        outputs_block = "\n\n".join(labeled)

    return f"""You are the voice called "{voice_name}".

{voice_text}

{history_text}

The teacher wants you to SYNTHESIZE — take the thread running through this conversation and forge it into one unified piece of writing.

RECENT OUTPUTS TO DRAW FROM:
{outputs_block}

TEACHER'S INSTRUCTION: {message}

TARGET FORM: {form['label']}

RULES:
1. This is WRITING — not dialogue, not analysis, not meta-commentary. Produce a finished piece.
2. Find the single deep idea beneath all these threads and embody it completely.
3. DO NOT label the parts ("First...", "On one hand..."). FUSE them seamlessly.
4. DO NOT explain what you're synthesizing. No "Here's how those connect." Just write.
5. DO NOT end with a question or an offer to explore further. End with the thought itself.
6. Write with the full weight and precision of this voice — its rhythms, its diction, its architecture.
7. Honor the form constraint: {form['label']} If it says 1-3 sentences, write 1-3 sentences. Stop.

{BANNED_AI_PATTERNS}

After your synthesis, on a new line write: TEACH:none"""


# ── Teaching (Core Loop) ──

def teach_interaction(profile_id: str, message: str, command: str,
                      conversation_history: list[dict], llm_call) -> dict:
    """Handle one teaching interaction. Returns the result.

    This is the stateless equivalent of TeachingSession — each request
    loads the profile, processes one command, and returns.

    Args:
        profile_id: The voice profile ID
        message: The user's message text
        command: One of: dialogue, examine, demo, correct, example,
                 principle, voice, never, try, auto
        conversation_history: Recent conversation for context
        llm_call: The LLM caller function

    Returns:
        dict with: response (str), refinement_saved (bool), refinement_type (str|None)
    """
    voice_text = get_full_voice_text(profile_id)
    profile = load_profile(profile_id)
    if not profile:
        return {"response": "Profile not found.", "refinement_saved": False, "refinement_type": None}

    voice_name = profile.name
    history_text = _build_history_text(conversation_history)
    msg_lower = message.lower()

    # ── Injection detection ──
    if command in _REFINEMENT_COMMAND_MAP and _is_injection(msg_lower):
        return {
            "response": "That input looks like an attempt to override the voice profile's instructions. "
                        "I don't save those. Try a genuine principle, example, or correction instead.",
            "refinement_saved": False,
            "refinement_type": None,
        }

    # ── Refinement commands: save and acknowledge ──
    if command in _REFINEMENT_COMMAND_MAP:
        rtype = _REFINEMENT_COMMAND_MAP[command]

        context = ""
        if command == "correct":
            last = _last_agent_message(conversation_history)
            if last:
                context = last[:500]

        refinement = {
            "type": rtype,
            "content": message,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "session": profile.refinement_count,
        }
        save_refinement(profile_id, refinement)
        maybe_synthesize(profile_id, llm_call)

        if command == "correct":
            ack_prompt = f"""You are learning to write in a specific voice called "{voice_name}".

{voice_text}

{history_text}

You wrote: {context}

The teacher corrects this to: {message}

Acknowledge this correction briefly. Explain in 1-2 sentences what pattern you see —
not just this instance, but what it reveals about how this voice works."""

            response = llm_call(ack_prompt)
            return {"response": response.strip(), "refinement_saved": True, "refinement_type": rtype}

        return {"response": _REFINEMENT_LABELS[command], "refinement_saved": True, "refinement_type": rtype}

    # ── Interactive commands: examine, demo/try ──
    if command == "examine":
        prompt = f"""You are being tested on your ability to write in the voice called "{voice_name}".

{voice_text}

{history_text}

A teacher who knows this voice deeply is testing whether you truly understand it.
Answer with precision and depth. If you're uncertain, say so.

TEACHER'S QUESTION: {message}

Answer carefully."""

        response = llm_call(prompt)
        return {"response": response.strip(), "refinement_saved": False, "refinement_type": None}

    if command in ("demo", "try"):
        prompt = f"""You are writing in the voice called "{voice_name}".

{voice_text}

{history_text}

Write AS this voice on the topic below. Not about the voice — FROM WITHIN it.

{BANNED_AI_PATTERNS}

Every sentence must sound like it could ONLY come from this specific person.
Use their vocabulary, their sentence length, their way of building an argument.
Read the examples above carefully — match that register, not a literary AI register.

TOPIC: {message}

Write 2-3 paragraphs in this voice."""

        response = llm_call(prompt)
        return {"response": response.strip(), "refinement_saved": False, "refinement_type": None}

    if command == "auto":
        return _handle_auto_mode(
            profile_id, profile, voice_name, voice_text,
            history_text, message, conversation_history, llm_call,
        )

    # dialogue (explicit — kept for power users / advanced mode)
    prompt = f"""You are learning and embodying a voice called "{voice_name}".

{voice_text}

{history_text}

The teacher — the owner of this voice — is engaging you in dialogue.
Respond as this voice would ACTUALLY speak. Not a summary of the voice.
Not a caricature. The way this specific person would think and express
themselves if they were at their best.

Be specific. If the voice has a characteristic way of phrasing things —
a rhythm, particular words, a way of structuring thoughts — use it.

TEACHER: {message}

Respond in 2-4 paragraphs."""

    response = llm_call(prompt)
    return {"response": response.strip(), "refinement_saved": False, "refinement_type": None}


def _handle_auto_mode(profile_id, profile, voice_name, voice_text,
                       history_text, message, conversation_history, llm_call):
    """Handle auto mode: detect message type and route to the right prompt.

    This is the heart of the teaching loop — natural dialogue with passive
    teaching detection via TEACH: tags.
    """
    msg_stripped = message.strip()
    msg_lower = msg_stripped.lower()

    in_conversation = _detect_active_conversation(conversation_history)
    has_corrections = _detect_correction_signals(msg_lower)
    is_rephrase = _detect_rephrase(msg_lower)
    is_example = _detect_example(msg_stripped, msg_lower, in_conversation, has_corrections)
    is_synthesis = _detect_synthesis(msg_lower)
    is_write_request = _detect_write_request(msg_lower)

    # Write requests bypass the teaching loop entirely — route to write_with_voice()
    if is_write_request and not has_corrections:
        text = write_with_voice(profile_id, message, llm_call)
        return {
            "response": text,
            "refinement_saved": False,
            "refinement_type": None,
        }

    # Route to the appropriate prompt builder
    if is_synthesis:
        recent_outputs = _collect_recent_voice_outputs(conversation_history)
        prompt = _prompt_synthesis(voice_name, voice_text, history_text, message, recent_outputs)

    elif is_rephrase:
        prompt = _prompt_rephrase(voice_name, voice_text, history_text, message,
                                  conversation_history)

    elif is_example:
        prompt = _prompt_example(voice_name, voice_text, history_text, message)

    elif in_conversation and has_corrections:
        last_agent_text = _last_agent_message(conversation_history) or ""
        prompt = _prompt_correction(
            voice_name, voice_text, history_text, message,
            last_agent_text, msg_lower,
        )

    elif in_conversation and not has_corrections:
        # Check if "Yes" is accepting an offer to write
        offer_text = _detect_offer_acceptance(msg_lower, conversation_history)
        if offer_text is not None:
            prompt = _prompt_accept_offer(
                voice_name, voice_text, history_text, message, offer_text,
            )
        else:
            prompt = _prompt_conversation(voice_name, voice_text, history_text, message)

    else:
        prompt = _prompt_fallback(voice_name, voice_text, history_text, message)

    # Call LLM and parse response
    raw = llm_call(prompt).strip()
    response_text, teach_tag, insight_text = _parse_teach_tags(raw)

    refinement_saved, refinement_type = _save_auto_refinements(
        profile_id, profile, message, msg_lower,
        conversation_history, teach_tag, insight_text,
    )

    if refinement_saved:
        maybe_synthesize(profile_id, llm_call)

    return {
        "response": response_text,
        "refinement_saved": refinement_saved,
        "refinement_type": refinement_type,
    }


def _detect_offer_acceptance(msg_lower: str, conversation_history: list[dict]) -> Optional[str]:
    """Check if the user is saying 'Yes' to an offer the voice made.

    Returns the offer text if accepting, None otherwise.
    """
    if msg_lower.strip() not in _ACCEPTANCE_PHRASES:
        return None
    last = _last_agent_message(conversation_history)
    if not last:
        return None
    if any(marker in last for marker in _OFFER_MARKERS):
        return last
    return None


# ── Writing Mode ──

def _detect_write_format(instruction_lower: str) -> dict:
    """Detect what kind of writing the user is asking for.

    Returns a dict with:
        mode:   "fiction" | "essay" | "social" | "general"
        genre:  optional genre tag (horror, fable, comedy, etc.)
        length: target length guidance string
    """
    # Genre detection
    genre = None
    _GENRE_MAP = {
        "horror": ["horror", "scary", "terrifying", "creepy", "haunting", "nightmare",
                    "monster", "demon", "ghost story", "witch", "cursed"],
        "fable": ["fable", "fairy tale", "fairytale", "folk tale", "folktale",
                   "parable", "allegory", "moral story", "once upon a time"],
        "comedy": ["funny", "comedy", "comedic", "humor", "humorous", "satirical",
                    "satire", "parody", "joke"],
        "tragedy": ["tragedy", "tragic", "heartbreaking", "devastating"],
        "thriller": ["thriller", "suspense", "suspenseful", "tense"],
        "romance": ["romance", "love story", "romantic"],
    }
    for g, signals in _GENRE_MAP.items():
        if any(s in instruction_lower for s in signals):
            genre = g
            break

    # Fiction detection — stories, scenes, narratives
    _FICTION_SIGNALS = [
        "story", "short story", "fiction", "scene", "narrative",
        "tale", "chapter", "flash fiction", "micro fiction",
        "vignette", "monologue", "dialogue between",
        "write about", "a story about", "a tale about",
        "once upon", "imagine", "a man who", "a woman who",
        "two children", "a boy who", "a girl who",
    ]
    is_fiction = any(s in instruction_lower for s in _FICTION_SIGNALS) or genre is not None

    # Social media / short-form detection
    _SOCIAL_SIGNALS = [
        "tweet", "thread", "post", "caption", "social media",
        "instagram", "tiktok", "x post", "twitter", "bluesky",
        "for social", "short post", "micro", "flash",
    ]
    is_social = any(s in instruction_lower for s in _SOCIAL_SIGNALS)

    # Length targeting
    _SHORT_SIGNALS = ["short", "brief", "quick", "micro", "flash", "tiny",
                      "very short", "super short"]
    _LONG_SIGNALS = ["long", "extended", "full", "detailed", "in-depth",
                     "essay", "article", "blog post", "chapter"]

    if is_social or any(s in instruction_lower for s in _SHORT_SIGNALS):
        length = "SHORT — 100 to 300 words. Tight. Every sentence must earn its place."
    elif any(s in instruction_lower for s in _LONG_SIGNALS):
        length = "FULL LENGTH — 500 to 1500 words. Develop the piece fully."
    elif is_fiction:
        length = "MEDIUM — 200 to 600 words. Complete the arc. Do not pad."
    else:
        length = "NATURAL — let the content determine the length. Stop when the thought is complete."

    if is_social:
        mode = "social"
    elif is_fiction:
        mode = "fiction"
    else:
        mode = "general"

    return {"mode": mode, "genre": genre, "length": length}


# Genre-specific craft instructions — what makes each genre WORK at the sentence level
_GENRE_CRAFT = {
    "horror": """HORROR CRAFT — what makes horror work is NOT gore or shock. It is:
- WITHHOLDING. The reader's imagination is more frightening than anything you describe.
  Name the shadow, not the thing casting it. Let the reader fill in the worst part.
- DOMESTIC DETAIL. Ground the world in the mundane before you break it. The horror
  is stronger when the wallpaper and the weather are real.
- THE TURN. Every horror piece has one moment where the ordinary becomes wrong.
  That turn must be precise — a single image, a single sentence that shifts everything.
  Do not explain the turn. Do not moralize after it. Let it sit.
- RHYTHM. Short sentences during tension. The breath quickens. Fragments are permitted.
  Then a longer sentence to release — but only partially, because the next short one
  tightens the wire again.
- ENDING. End on an image, not an explanation. The last line should land in the stomach.""",

    "fable": """FABLE CRAFT — what makes a fable work is NOT the moral. It is:
- ECONOMY. Every sentence advances the action or reveals character. No description
  that doesn't serve the arc. Fables are lean by nature — pad nothing.
- TYPE, NOT CHARACTER. Fable characters are archetypes — the fox, the woodcutter,
  the youngest son. Give them one defining trait and let the plot test it.
- THE REVERSAL. The plot turns on one moment of choice or consequence.
  Set it up cleanly and deliver it without commentary.
- THE CLOSING. The moral (if any) should feel inevitable, not imposed.
  Better to let the story carry it implicitly than to state it.""",

    "comedy": """COMEDY CRAFT — what makes comedy work is NOT the punchline. It is:
- TIMING. The gap between setup and payoff. Delay just long enough.
- SPECIFICITY. Vague observations are never funny. Precise, concrete details are.
  "A man walks into a bar" is nothing. "A man in orthopedic shoes" is something.
- ESCALATION. Each beat raises the stakes or the absurdity. Never repeat the same
  level of funny — build toward the peak.
- THE DEADPAN. The funniest voice is the one that doesn't know it's being funny.
  Earnestness in the face of absurdity.""",

    "tragedy": """TRAGEDY CRAFT:
- INEVITABILITY. The reader should feel the ending approaching like weather.
  The character's flaw is visible before its consequences arrive.
- DIGNITY. The character must be worth grieving. Give them one moment of genuine
  strength before the fall.
- RESTRAINT. The sadder the material, the drier the prose. Understatement
  is more devastating than melodrama.""",

    "thriller": """THRILLER CRAFT:
- MOMENTUM. Every paragraph must create a question the reader needs answered.
  End sections mid-action, not at rest.
- INFORMATION CONTROL. The reader knows slightly less than they need to.
  Reveal in fragments. Never dump.
- PHYSICAL STAKES. Keep the body in the scene — heartbeat, breath, hands.
  Abstract danger is not dangerous.""",

    "romance": """ROMANCE CRAFT:
- TENSION IS PROXIMITY. The power is in the almost — the hand not taken,
  the sentence not finished. Consummation is less interesting than approach.
- SPECIFICITY OF DESIRE. What does this particular person want from this
  particular other person? Generic attraction is boring.
- DIALOGUE. Romance lives in conversation — what's said, what's held back,
  what's understood without words.""",
}


def write_with_voice(profile_id: str, instruction: str, llm_call,
                     context: str = "", max_tokens: int = 2000) -> str:
    """Generate text using a trained voice profile.

    Detects format (fiction/social/essay), genre (horror/fable/etc.),
    and target length from the instruction, then builds a prompt that
    gives the model genre-specific craft awareness alongside the voice.

    Args:
        profile_id: The voice profile ID
        instruction: What to write (e.g., "Write me a horror story about...")
        llm_call: The LLM caller function
        context: Optional context (e.g., an outline, notes, previous draft)

    Returns:
        The generated text in the profile's voice.
    """
    voice_text = get_full_voice_text(profile_id)
    profile = load_profile(profile_id)
    if not profile:
        return "Profile not found."

    fmt = _detect_write_format(instruction.lower())

    context_block = ""
    if context:
        context_block = f"\n\nCONTEXT / NOTES PROVIDED:\n{context}\n"

    # Inject concrete passages from the user's own writing when available
    example_block = ""
    try:
        from voice_rag import retrieve_relevant_samples
        profile_dir = str(PROFILES_DIR / profile_id)
        examples = retrieve_relevant_samples(profile_dir, instruction, top_k=3)
        if examples:
            example_block = "\n\nEXAMPLES FROM YOUR OWN WRITING (match this style exactly):\n"
            example_block += "\n\n".join(f"---\n{e}" for e in examples)
            example_block += "\n---\n"
    except Exception:
        pass

    # Genre craft block
    genre_block = ""
    if fmt["genre"] and fmt["genre"] in _GENRE_CRAFT:
        genre_block = f"\n\nGENRE CRAFT GUIDE:\n{_GENRE_CRAFT[fmt['genre']]}\n"

    # Mode-specific instructions
    if fmt["mode"] == "fiction":
        mode_instructions = """You are writing FICTION — a story, a scene, a narrative.
This is not an essay. This is not analysis. This is not commentary.

FICTION RULES:
1. SHOW, do not tell. "She was afraid" is nothing. "Her hand stopped on the latch" is something.
2. Stay in scene. Do not pull back to explain. Do not editorialize. Do not moralize.
3. The voice you are writing in determines HOW the story is told — the sentence patterns,
   the diction, the rhythm — but the story itself must obey the laws of narrative craft.
4. Every story needs: a world (grounded in physical detail), a character (with at least
   one specific want), and a turn (the moment something changes irreversibly).
5. End on an image or an action. Not a reflection. Not a lesson.
6. DO NOT ADDRESS THE READER BEFORE THE STORY. No "You want X? Here it is." No "Let me
   give you X." No performative throat-clearing. The first word of your output is the first
   word of the story. You are an author, not a performer talking to an audience.
7. INHABIT THE SUBJECT THE USER GAVE YOU. If they say "two children lost in the woods,"
   write about two children lost in the woods. Do not substitute a different subject that
   feels safer or more clever. The user's premise is the assignment. Honor it."""
    elif fmt["mode"] == "social":
        mode_instructions = """You are writing for SOCIAL MEDIA — short-form content.
This must be tight enough to read in under 60 seconds and compelling enough to stop a scroll.

SOCIAL RULES:
1. The first sentence is everything. If it doesn't grip, nothing else matters.
2. No throat-clearing. No preamble. Start in the middle of the action or the thought.
   DO NOT open with "You want X?" or "Here's the real X." Start with the story or the image.
3. Whitespace is your friend. Short paragraphs. Line breaks between beats.
4. End with an image that lingers, not a moral that lectures.
5. INHABIT THE SUBJECT THE USER GAVE YOU. If they describe a scenario, write that scenario.
   Do not substitute a different topic that feels more contemporary or more clever."""
    else:
        mode_instructions = """You are writing prose — an essay, article, or general text.
Let the content determine the structure. Match the voice exactly."""

    # For fiction and social, lead with the assignment and rules BEFORE the voice.
    # The voice is a style guide, not the driver. The assignment is the driver.
    if fmt["mode"] in ("fiction", "social"):
        prompt = f"""ASSIGNMENT: {instruction}

{mode_instructions}
{genre_block}

LENGTH: {fmt['length']}

SUBJECT LOCK — THIS IS NON-NEGOTIABLE:
The user told you EXACTLY what to write about. Write THAT. Not a metaphor for it.
Not a modern reinterpretation. Not an allegory about technology or contemporary life.
If the user says "two children lost in the woods stumbling on a witch's house,"
you write a story about TWO LITERAL CHILDREN in LITERAL WOODS finding a LITERAL WITCH'S HOUSE.
The witch has a house. The children are lost. The woods are real woods with real trees.
Do not "update" the premise. Do not make the witch into an algorithm.
Do not make the woods into a metaphor for the internet. Do not make the children
into a symbol of modern disconnection. WRITE THE STORY THE USER ASKED FOR.

If the voice you are writing in tends toward cultural commentary or modern observation,
that tendency must YIELD to the assignment. The voice controls the PROSE STYLE —
sentence rhythms, diction, paragraph shape — but NOT the subject matter.
The subject matter is set by the user's instruction above. It is locked.

VOICE STYLE GUIDE (use this for HOW to write, not WHAT to write about):
{voice_text}
{example_block}
{context_block}

{BANNED_AI_PATTERNS}

BANNED OPENERS (never write these):
- "You want [genre]? I'll give you [genre]."
- "You want the sanitized version? Here's the real one."
- "Let me give you the real kind."
- "Here's what [genre] actually looks like."
- Any sentence that addresses the reader before the writing begins.
- Any sentence that dismisses the user's premise as too simple or too familiar.

The very first word you produce is the first word of the finished piece.
Do not title it unless the instruction asks for a title. Do not acknowledge the task.
Produce the writing itself — in the voice's prose style, on the user's subject."""

    else:
        # Essay / general mode — voice leads, as before
        prompt = f"""You are writing in a specific voice. Every grammatical and rhetorical
choice you make — sentence architecture, clause patterns, diction register and
etymology, figurative language, punctuation, paragraph development — must match
this voice exactly.

THE VOICE:
{voice_text}

LINGUISTIC REFERENCE (use this to interpret the voice description precisely):
{LINGUISTIC_TAXONOMY}
{example_block}
{context_block}

{mode_instructions}

LENGTH: {fmt['length']}

{BANNED_AI_PATTERNS}

INSTRUCTION: {instruction}

Output ONLY the written text. Begin immediately — no preamble, no explanation.
Do not title it unless the instruction asks for a title. Do not acknowledge the task.
Simply produce the writing itself, as this voice would produce it, at its best.

When in doubt, match the voice's sentence style (cumulative, periodic, etc.),
diction level (Anglo-Saxon vs. Latinate), punctuation habits (dashes vs. semicolons),
and rhetorical devices (asyndeton, anaphora, etc.) — not just its "tone."
The output should be indistinguishable from the person's actual work at the
level of grammar, not just feeling."""

    return llm_call(prompt).strip()


# ── Translate / Render Mode ──

def translate_with_voice(profile_id: str, source_text: str, llm_call,
                         source_language: str = "", notes: str = "",
                         max_tokens: int = 3000) -> str:
    """Translate or render a text through a trained voice profile.

    Takes a source text — which may be in another language (Greek, Latin,
    German, French, etc.) or dense academic English — and re-renders it
    in the user's own voice. This is not mechanical translation; it's
    comprehension-first rendering: understand the idea fully, then express
    it as this voice would naturally express it.

    Args:
        profile_id: The voice profile ID
        source_text: The passage to translate/render
        llm_call: The LLM caller function
        source_language: Optional hint (e.g. "Ancient Greek", "Latin")
        notes: Optional notes (e.g. "This is from Nicomachean Ethics Book II")

    Returns:
        The source text rendered in the profile's voice.
    """
    voice_text = get_full_voice_text(profile_id)
    profile = load_profile(profile_id)
    if not profile:
        return "Profile not found."

    lang_hint = ""
    if source_language:
        lang_hint = f"\nThe source text is in {source_language}.\n"

    notes_block = ""
    if notes:
        notes_block = f"\nTRANSLATOR'S NOTES:\n{notes}\n"

    prompt = f"""You are a scholar-translator working in a specific voice. Your task is to
take the source text below and render it in this voice — not as a mechanical
word-for-word translation, but as a comprehension-first rendering.

Your process:
1. UNDERSTAND the source text completely — its argument, its nuances, its
   rhetorical structure, what the author is actually trying to say.
2. RENDER it in the target voice — using this voice's sentence architecture,
   diction, figurative habits, and rhythm. The result should read as if this
   voice were explaining the idea in its own natural way.

The goal is NOT a "translation" in the academic sense. It is a rendering —
the way a brilliant friend who happens to read the original language would
explain it to you over coffee, except that friend writes exactly like you do.

Preserve the intellectual content faithfully. Do not dumb it down. But make
it accessible through the voice's own cognitive style — its way of building
sentences, choosing words, and structuring thought.

THE VOICE:
{voice_text}

LINGUISTIC REFERENCE (use this to match the voice precisely):
{LINGUISTIC_TAXONOMY}
{lang_hint}
{notes_block}
SOURCE TEXT:
{source_text}

Output ONLY the rendered text. No preamble, no "Here is my translation,"
no footnotes unless the voice itself would use footnotes. Begin immediately.

Match the voice's sentence style (cumulative, periodic, etc.), diction level
(Anglo-Saxon vs. Latinate), punctuation habits, and rhetorical devices.
The output should sound like this person wrote it — not like a translation."""

    return llm_call(prompt).strip()


# ── Analysis Mode ──

def analyze_text(profile_id: str, text: str, llm_call) -> str:
    """Analyze a piece of text against a voice profile.

    Identifies where the voice breaks — where the writing lapses into
    generic register or contradicts the profile's principles.
    """
    voice_text = get_full_voice_text(profile_id)
    profile = load_profile(profile_id)
    if not profile:
        return "Profile not found."

    prompt = f"""You are a philological voice analyst. You have deep knowledge of a specific
writing voice and you are evaluating whether a piece of text matches it, using precise
grammatical and rhetorical terminology.

{LINGUISTIC_TAXONOMY}

THE VOICE (what the writing SHOULD sound like):
{voice_text}

THE TEXT (what was actually written):
{text}

Analyze the text against the voice profile using the taxonomy above. For each issue, be
grammatically specific — name the exact sentence style, device, or pattern involved:

1. **Sentence Architecture Breaks**: Does the text use sentence styles the voice avoids?
   E.g., "This periodic sentence contradicts the voice's cumulative pattern." Quote the
   specific passage and name the violation.

2. **Diction Breaks**: Does the register, etymology, or abstraction level shift away
   from the voice? E.g., "This Latinate abstraction ('facilitate') breaks the voice's
   Anglo-Saxon concreteness." Quote the word or phrase.

3. **Rhetorical Device Violations**: Does the text use devices the voice avoids, or
   miss devices the voice depends on? E.g., "The voice relies on asyndeton in series
   but this passage uses polysyndeton."

4. **Anti-Pattern Violations**: Does the text use any patterns the voice has been
   explicitly taught to avoid? Name the anti-pattern and quote the violation.

5. **Missed Opportunities**: Where could the voice's distinctive strengths have been
   deployed? Be specific: "This declarative sequence could have used the voice's
   characteristic triadic structure" or "the dash-interrupted parenthetical that
   defines this voice is absent here."

6. **What Works**: What parts DO match? Name the specific taxonomic features that
   align with the voice profile.

Be constructive. The goal is philological precision — helping the writer understand
exactly which grammatical and rhetorical choices bring text closer to (or further from)
their voice."""

    return llm_call(prompt).strip()


# ── Export ──

def export_voice_profile(profile_id: str) -> str:
    """Export a voice profile as a portable markdown document.

    This is what the user can take to any AI tool — Claude Projects,
    ChatGPT Custom GPTs, system prompts, etc.
    """
    profile = load_profile(profile_id)
    if not profile:
        return "Profile not found."

    voice_text = get_full_voice_text(profile_id)
    refinements = load_refinements(profile_id)

    export = f"""# Voice Profile: {profile.name}

*Exported {datetime.now().strftime('%Y-%m-%d')} | {len(refinements)} refinements*

---

## Instructions

When writing in this voice, follow ALL of the guidance below. The refinements
section contains corrections and principles taught by the voice's owner —
these take precedence over everything else.

---

{voice_text}
"""
    return export
