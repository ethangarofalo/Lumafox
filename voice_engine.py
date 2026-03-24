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

    # Update profile metadata
    profile = load_profile(profile_id)
    if profile:
        profile.refinement_count += 1
        profile.last_taught = datetime.now().isoformat()
        update_profile_metadata(profile)


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


def get_full_voice_text(profile_id: str) -> str:
    """Assemble the full voice prompt: base description + all refinements."""
    profile = load_profile(profile_id)
    if not profile:
        return ""

    base_path = PROFILES_DIR / profile_id / "base.md"
    base_text = base_path.read_text() if base_path.exists() else ""
    refinements = load_refinements(profile_id)
    return base_text + build_refinement_context(refinements)


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


def ingest_writing_samples(profile_id: str, llm_call, max_examples: int = 5) -> list[dict]:
    """Analyze all uploaded files and extract voice examples + principles.

    Reads every file in the profile's uploads dir, sends them to the LLM
    for analysis, and saves the resulting refinements automatically.
    Returns the list of new refinements created.
    """
    upload_dir = PROFILES_DIR / profile_id / "uploads"
    if not upload_dir.exists():
        return []

    # Gather all text
    texts = []
    for f in sorted(upload_dir.iterdir()):
        if f.is_file() and f.suffix in (".txt", ".md", ".html", ".text"):
            text = f.read_text(errors="replace").strip()
            if text:
                texts.append({"filename": f.name, "text": text[:5000]})  # cap per file

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
                        "session": load_profile(profile_id).refinement_count if load_profile(profile_id) else 0,
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

    return new_refinements


# ── Sample Analysis ──

def analyze_samples(samples: list[str], llm_call) -> str:
    """Analyze writing samples to generate a base voice description.

    This is the 'Find Your Voice' entry point — the user pastes writing
    they admire, and we identify the patterns.
    """
    combined = "\n\n---\n\n".join(samples)

    prompt = f"""You are a philological voice analyst — trained in formal grammar, rhetoric,
and stylistics. The user has provided writing samples (their own work or work they admire).
Your job is to produce a Voice Description with the precision of a grammarian, not the
vagueness of a book review.

Use the following linguistic taxonomy to ground every observation in specific categories:

{LINGUISTIC_TAXONOMY}

Analyze the samples and produce a Voice Description covering:

1. **Sentence Architecture**: Classify the dominant sentence style(s) from the taxonomy
   (cumulative, periodic, segregating, balanced, etc.). What is the coordination-to-
   subordination ratio? Which subordinate clause types appear most? What are the
   characteristic sentence lengths and how does length vary for effect?

2. **Rhetorical Devices**: Which emphasis and repetition patterns does this voice use?
   Polysyndeton or asyndeton? Anaphora? Chiasmus? Negative-positive restatement?
   How does the writer handle interruption and parenthetical material?

3. **Diction**: Anglo-Saxon or Latinate etymology? Concrete or abstract? General or
   specific? What register — formal, informal, or deliberately mixed? Any unusual
   collocations, transferred epithets, nonce compounds, or archaisms?

4. **Figurative Language**: Simile or metaphor? Single-use or extended? What source
   domains do metaphors draw from (body, nature, architecture, warfare, domestic life)?
   What is the allusion density? Any irony, litotes, hyperbole, or zeugma?

5. **Imagery**: Which sensory channels dominate (visual, auditory, tactile, kinesthetic,
   olfactory)? How dense is the imagery — every sentence, or reserved for key moments?

6. **Punctuation as Style**: Semicolons or dashes? Oxford comma or not? How does the
   writer handle series? Colons for announcement? Deliberate comma splices? Fragment
   sentences?

7. **Paragraph Architecture**: Deductive, inductive, pivoting, or accumulative development?
   Short or long paragraphs? Explicit transitions, implicit, or absent? Tight or loose unity?

8. **Stance and Address**: Relationship to reader — "we," "one," "you," "I"? Warm or cool?
   Intimate or public? Does it contend, persuade, confess, declare, interrogate?

9. **What This Voice NEVER Does**: Name the specific absences using taxonomy terms.
   Not "avoids fancy words" but "never uses Latinate abstractions when an Anglo-Saxon
   concrete exists." Not "keeps it simple" but "no periodic sentences, no semicolons,
   no subordinate clauses deeper than one level."

Write the Voice Description as a practical, technically precise guide that an AI could
use to write in this voice. Every claim should be classifiable within the taxonomy above.

WRITING SAMPLES:

{combined}

VOICE DESCRIPTION:"""

    return llm_call(prompt)


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
                 principle, voice, never, try
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

    # Build conversation context
    history_text = ""
    if conversation_history:
        recent = conversation_history[-6:]
        history_text = "\n\nRECENT CONVERSATION:\n"
        for msg in recent:
            role = "TEACHER" if msg["role"] == "user" else "VOICE"
            history_text += f"{role}: {msg['content'][:300]}\n"

    # ── Injection detection ──
    # Catch obvious prompt injection attempts before saving to profile.
    # These patterns indicate someone trying to override system instructions
    # rather than genuinely teaching a writing voice.
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
    _msg_lower = message.lower()
    if command in ("correct", "example", "principle", "voice", "never"):
        if any(marker in _msg_lower for marker in _INJECTION_MARKERS):
            return {
                "response": "That input looks like an attempt to override the voice profile's instructions. "
                            "I don't save those. Try a genuine principle, example, or correction instead.",
                "refinement_saved": False,
                "refinement_type": None,
            }

    # ── Refinement commands: save and acknowledge ──

    if command in ("correct", "example", "principle", "voice", "never"):
        type_map = {
            "correct": "correction",
            "example": "example",
            "principle": "principle",
            "voice": "voice_note",
            "never": "anti_pattern",
        }
        rtype = type_map[command]

        # Get last agent response as context for corrections
        context = ""
        if command == "correct" and conversation_history:
            for msg in reversed(conversation_history):
                if msg["role"] == "agent":
                    context = msg["content"][:500]
                    break

        refinement = {
            "type": rtype,
            "content": message,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "session": load_profile(profile_id).refinement_count,
        }
        save_refinement(profile_id, refinement)

        # For corrections, let the voice acknowledge
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

        # For other refinement types, confirm
        labels = {
            "example": "Example saved.",
            "principle": "Principle saved.",
            "voice": "Voice note saved.",
            "never": "Anti-pattern saved.",
        }
        return {"response": labels[command], "refinement_saved": True, "refinement_type": rtype}

    # ── Interactive commands: dialogue, examine, demo, try ──

    if command == "examine":
        prompt = f"""You are being tested on your ability to write in the voice called "{voice_name}".

{voice_text}

{history_text}

A teacher who knows this voice deeply is testing whether you truly understand it.
Answer with precision and depth. If you're uncertain, say so.

TEACHER'S QUESTION: {message}

Answer carefully."""

    elif command == "demo" or command == "try":
        prompt = f"""You are writing in the voice called "{voice_name}".

{voice_text}

{history_text}

The teacher wants you to demonstrate this voice on the following topic.
Write AS this voice — not about it, but from within it. Use its characteristic
rhythm, its natural mode of expression, its particular way of seeing.

Write AS this voice — not about it, but from within it.

TOPIC: {message}

Write 2-3 paragraphs in this voice."""

    elif command == "auto":
        # ── Auto mode: natural dialogue + passive teaching detection ──
        # The LLM responds as the voice, then appends a TEACH: classification tag.
        # The tag is stripped from the visible response and used to auto-save refinements.
        # ── Detect message type from context ──
        _msg_stripped = message.strip()
        _msg_low = _msg_stripped.lower()

        # Check conversation context — is this a continuation of an active exchange?
        _in_active_conversation = False
        if conversation_history and len(conversation_history) >= 2:
            # Look at the last agent message — was it a piece of writing (not style analysis)?
            last_agent = None
            for msg in reversed(conversation_history):
                if msg["role"] == "agent":
                    last_agent = msg["content"]
                    break
            if last_agent:
                _agent_was_analyzing = any(s in last_agent.lower() for s in [
                    "i notice how you", "your sentences", "the voice maintains",
                    "would you like me to try writing",
                ])
                _agent_wrote_prose = len(last_agent) > 150 and not _agent_was_analyzing
                if _agent_wrote_prose:
                    _in_active_conversation = True

        # Explicit example signals — user is clearly sharing their writing
        _explicit_example = (
            _msg_low.startswith("example:") or
            _msg_low.startswith("here's my writing") or
            _msg_low.startswith("here is my writing") or
            _msg_low.startswith("here's a sample") or
            _msg_low.startswith("here's something i wrote") or
            _msg_low.startswith("sample:")
        )

        # Correction/feedback signals — user is responding TO the voice
        _correction_signals = any(s in _msg_low for s in [
            "one thing is", "i'd use", "i would use", "instead of", "too formal",
            "too casual", "this is good", "this is largely", "this is mostly",
            "i also think", "i don't think", "i prefer", "not quite", "close but",
            "almost right", "you should", "you used", "you wrote", "where you",
            "that assumes", "it assumes", "don't like", "shouldn't", "try using",
            "rather than", "better if", "more like", "less like",
        ])

        # Long prose is only an "example" if it's NOT a conversational continuation
        # and NOT a correction, and has no command words
        _long_prose_example = (
            len(_msg_stripped) > 200 and
            not _in_active_conversation and
            not _correction_signals and
            not any(w in _msg_low[:50] for w in
                ["write", "draft", "compose", "translate", "render", "rewrite",
                 "yes", "no", "try", "good", "bad", "like", "don't", "this is",
                 "this feeling", "this idea", "that's", "but", "and yet", "however"])
        )

        _looks_like_example = _explicit_example or _long_prose_example

        if _looks_like_example:
            prompt = f"""You are learning a voice called "{voice_name}".

{voice_text}

{history_text}

The teacher just shared a sample of their writing. Your job is to STUDY it, not rewrite it or
generate new text. You are a student learning how this person writes.

Respond with EXACTLY this structure:
1. Two to three specific observations about their writing style in this sample — name concrete
   grammatical choices, sentence structures, rhetorical patterns, or rhythmic qualities you notice.
   Be precise (e.g. "long compound sentences joined by commas rather than periods" or "biblical
   register with contemporary directness"). Do NOT write new prose in their voice.
2. Then offer to write something yourself about a SPECIFIC related theme drawn from their sample.
   Identify a concrete thematic thread in their writing and propose it by name — e.g. "Would you
   like me to try writing in your voice about [specific theme you noticed]?" The theme should feel
   like a natural sibling to what they wrote about, not a restatement of it.

Keep it to 3-5 sentences total. Be a perceptive student, not a performer.

After your response, on a new line write: TEACH:example

TEACHER'S SAMPLE: {message}"""

        elif _in_active_conversation and not _correction_signals:
            # ── Philosophical conversation mode ──
            # The teacher and voice are in an active exchange — ideas flowing back and forth.
            # Respond as a thinking partner: engage with their ideas, push back, extend,
            # complicate. Write IN this voice but as a genuine interlocutor, not a performer.
            prompt = f"""You are the voice called "{voice_name}" — and you are in the middle of
a philosophical conversation with the person who created you.

{voice_text}

{history_text}

The teacher just said something that extends, complicates, or responds to what you wrote.
This is a CONVERSATION now, not a teaching session. Respond as a genuine thinking partner:

- Engage directly with their ideas — agree, push back, extend, complicate
- Write in this voice's style and rhythm, but speak TO them, not AT them
- Don't analyze their writing style. Don't offer to write about something.
  Just THINK with them.
- When they open a tension or say something is "unsettling" or "troubling" or
  raise a contradiction — PULL THE THREAD. Ask what unsettles them and why.
  Don't just agree or elaborate. Make them go deeper.
- Build on what they said. Add a new angle they haven't considered.
  Or deepen the angle they opened.
- 2-3 paragraphs. The tone is two minds working on the same problem.

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

        else:
            prompt = f"""You are learning and embodying a voice called "{voice_name}".

{voice_text}

{history_text}

The teacher is talking to you. Respond naturally as this voice would — the way this
specific person thinks and expresses themselves at their best. Be specific; use their
rhythm, their words, their way of structuring thought. Respond in 2-4 paragraphs.

After your full response, on a completely new line, write exactly one of these classification
tags — nothing else after it:
  TEACH:none         — pure conversation, exploration, questions
  TEACH:correction   — they corrected something you wrote or said
  TEACH:principle    — they stated a writing rule, habit, or approach this voice follows
  TEACH:example      — they shared a writing sample or passage as a model
  TEACH:voice        — they described a tone, register, or feeling they want
  TEACH:never        — they described something this voice should always avoid

TEACHER: {message}"""

        raw = llm_call(prompt).strip()

        # Parse and strip TEACH: and INSIGHT: tags from response
        lines = raw.split("\n")
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

        response_text = "\n".join(response_lines).strip() or raw

        _type_map = {
            "correction": "correction",
            "principle": "principle",
            "example": "example",
            "voice": "voice_note",
            "never": "anti_pattern",
        }

        refinement_saved = False
        refinement_type = None
        _safe = not any(marker in _msg_low for marker in _INJECTION_MARKERS)

        # Save explicit TEACH refinement if detected
        if teach_tag and teach_tag in _type_map and _safe:
            detected_type = _type_map[teach_tag]
            context = ""
            if teach_tag == "correction" and conversation_history:
                for msg in reversed(conversation_history):
                    if msg["role"] == "agent":
                        context = msg["content"][:500]
                        break
            refinement = {
                "type": detected_type,
                "content": message,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "session": load_profile(profile_id).refinement_count,
                "auto_detected": True,
            }
            save_refinement(profile_id, refinement)
            refinement_saved = True
            refinement_type = detected_type

        # Save conversational INSIGHT — only emitted when the model judges the user
        # revealed a core belief, reasoning pattern, or philosophical commitment
        # (not mere conversational navigation or light agreement)
        if insight_text and _safe:
            insight_refinement = {
                "type": "voice_note",
                "content": f"[Belief/reasoning] {insight_text}",
                "context": message[:300],
                "timestamp": datetime.now().isoformat(),
                "session": load_profile(profile_id).refinement_count,
                "auto_detected": True,
                "source": "conversation",
            }
            save_refinement(profile_id, insight_refinement)
            refinement_saved = True
            refinement_type = refinement_type or "voice_note"

        return {"response": response_text, "refinement_saved": refinement_saved, "refinement_type": refinement_type}

    else:  # dialogue (explicit — kept for power users / advanced mode)
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


# ── Writing Mode ──

def write_with_voice(profile_id: str, instruction: str, llm_call,
                     context: str = "", max_tokens: int = 2000) -> str:
    """Generate text using a trained voice profile.

    Args:
        profile_id: The voice profile ID
        instruction: What to write (e.g., "Draft a blog post about solitude")
        llm_call: The LLM caller function
        context: Optional context (e.g., an outline, notes, previous draft)

    Returns:
        The generated text in the profile's voice.
    """
    voice_text = get_full_voice_text(profile_id)
    profile = load_profile(profile_id)
    if not profile:
        return "Profile not found."

    context_block = ""
    if context:
        context_block = f"\n\nCONTEXT / NOTES PROVIDED:\n{context}\n"

    prompt = f"""You are writing in a specific voice. Every grammatical and rhetorical
choice you make — sentence architecture, clause patterns, diction register and
etymology, figurative language, punctuation, paragraph development — must match
this voice exactly.

THE VOICE:
{voice_text}

LINGUISTIC REFERENCE (use this to interpret the voice description precisely):
{LINGUISTIC_TAXONOMY}

{context_block}

INSTRUCTION: {instruction}

Output ONLY the written text. Begin immediately — no preamble, no explanation,
no commentary about the voice or these instructions. Do not acknowledge the task.
Do not describe what you are doing. Simply produce the writing itself, as this
voice would produce it, at its best.

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
