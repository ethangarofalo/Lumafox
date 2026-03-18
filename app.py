"""
Luxa — FastAPI Application

The API layer over voice_engine.py. Stateless endpoints for
teaching, writing, analyzing, and managing voice profiles.
"""

import os
from datetime import datetime
from typing import Optional

from pathlib import Path
import anthropic as _anthropic

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from auth import register, login, verify_token, create_token, refresh_token, ensure_dirs as ensure_auth_dirs
from billing import (
    get_subscription, get_plan_limits, create_checkout_session,
    create_portal_session, handle_webhook,
    ensure_dirs as ensure_billing_dirs,
)
from voice_engine import (
    create_profile,
    load_profile,
    list_profiles,
    delete_profile,
    load_refinements,
    get_full_voice_text,
    teach_interaction,
    write_with_voice,
    analyze_text,
    analyze_samples,
    export_voice_profile,
    save_uploaded_file,
    list_uploaded_files,
    ingest_writing_samples,
    update_profile_metadata,
    ensure_dirs,
    save_conversation_session,
    list_conversation_sessions,
    load_conversation_session,
)

# ── LLM Setup ──

from llm import make_claude_caller, make_mock_caller
from council import COUNCIL_NAMES, VALID_MODES
from council_agents import run_council_agents
from council_swarm import run_council_swarm


def make_voice_mock_caller():
    """
    A mock LLM caller that returns voice-plausible placeholder text
    instead of POLIS philosophical responses.
    """
    counter = [0]

    MOCK_RESPONSES = [
        "[MOCK WRITE] The words came out short and straight. No decoration. That was the style — lean and deliberate, the way a carpenter marks a board before cutting.",
        "[MOCK WRITE] Something shifted in the way people talked about it. Not loudly. The change was quiet, the way most real changes are.",
        "[MOCK TEACH] Understood. I've noted that preference and will apply it going forward.",
        "[MOCK TEACH] Got it — I'll treat that as a hard rule in this voice.",
        "[MOCK ANALYZE] This text shows a clear voice. A few patterns worth noting: the sentence rhythm is consistent, the diction leans formal, and the transitions rely on logical connectives.",
        "[MOCK ANALYZE] The voice here is trying to do two things at once — explain and impress. The explanation is clear. The impressing gets in the way.",
        "[MOCK WRITE] He sat at the desk for a long time. The cursor blinked. He had taught the machine what he wanted, and now the machine waited.",
        "[MOCK TEACH] Noted. I'll make sure that pattern stays out of any text generated in this voice.",
    ]

    def call(messages, **kwargs):
        idx = counter[0] % len(MOCK_RESPONSES)
        counter[0] += 1
        return MOCK_RESPONSES[idx]

    return call


def get_llm_caller():
    """Get the LLM caller based on environment."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return make_claude_caller()  # reads ANTHROPIC_API_KEY from env
    print("⚠ No ANTHROPIC_API_KEY set — using voice mock LLM caller")
    return make_voice_mock_caller()


LLM_CALL = get_llm_caller()


# ── App ──

app = FastAPI(
    title="Luxa",
    description=(
        "Luxa — Voice and Council. "
        "Teach an AI your voice through dialogue, then summon history's greatest minds on any question."
    ),
    version="0.1.0",
)

# CORS — restrict to your domain in production via ALLOWED_ORIGINS env var
# e.g. ALLOWED_ORIGINS=https://yourapp.com,https://www.yourapp.com
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Simple in-memory rate limiter for auth endpoints ──
import time as _time
from collections import defaultdict

_login_attempts: dict = defaultdict(list)  # ip -> [timestamps]
_LOGIN_MAX = int(os.environ.get("LOGIN_RATE_LIMIT", "10"))   # attempts
_LOGIN_WINDOW = int(os.environ.get("LOGIN_RATE_WINDOW", "300"))  # seconds (5 min)


def _check_login_rate(ip: str):
    """Raise 429 if too many login attempts from this IP."""
    now = _time.time()
    attempts = _login_attempts[ip]
    # Prune old attempts
    _login_attempts[ip] = [t for t in attempts if now - t < _LOGIN_WINDOW]
    if len(_login_attempts[ip]) >= _LOGIN_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Try again in {_LOGIN_WINDOW // 60} minutes.",
        )
    _login_attempts[ip].append(now)


@app.on_event("startup")
async def startup():
    ensure_dirs()
    ensure_auth_dirs()
    ensure_billing_dirs()


# ── Auth ──

security = HTTPBearer(auto_error=False)

# Dev auth (X-User-Id header) is only enabled when explicitly set.
# Set DEV_AUTH=false in production to ensure JWT is always required.
_DEV_AUTH = os.environ.get("DEV_AUTH", "true").lower() == "true"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_user_id: str = Header(default=None),
) -> str:
    """Extract user ID from JWT token, or fall back to header for dev."""
    if credentials:
        payload = verify_token(credentials.credentials)
        if payload:
            return payload["sub"]
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    # Dev fallback: header-based auth (disabled in production via DEV_AUTH=false)
    if _DEV_AUTH and x_user_id:
        return x_user_id
    raise HTTPException(status_code=401, detail="Authentication required")


# ── Auth Endpoints ──

class RegisterRequest(BaseModel):
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")
    name: str = Field(default="", description="Display name")


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


@app.post("/auth/guest")
async def guest_session():
    """Create an anonymous guest session — no sign-up required."""
    import uuid
    guest_id = f"guest-{uuid.uuid4().hex[:12]}"
    token = create_token(guest_id, f"{guest_id}@guest")
    return {
        "token": token,
        "user": {"user_id": guest_id, "name": "Guest", "email": "", "plan": "free"},
    }


def _transfer_guest_profiles(guest_id: str, new_user_id: str):
    """Move any profiles created under a guest session to a real account."""
    if not guest_id or not guest_id.startswith("guest-"):
        return
    for profile in list_profiles(guest_id):
        profile.owner_id = new_user_id
        update_profile_metadata(profile)


@app.post("/auth/register", response_model=AuthResponse)
async def register_user(req: RegisterRequest, request: Request,
                        x_guest_id: str = Header(default=None)):
    """Create a new account, optionally transferring a guest session's profiles."""
    _check_login_rate(request.client.host if request.client else "unknown")
    try:
        user = register(req.email, req.password, req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = login(req.email, req.password)
    _transfer_guest_profiles(x_guest_id, result["user"]["user_id"])
    return AuthResponse(token=result["token"], user=result["user"])


@app.post("/auth/login", response_model=AuthResponse)
async def login_user(req: LoginRequest, request: Request,
                     x_guest_id: str = Header(default=None)):
    """Log in and receive a JWT token, optionally absorbing a guest session."""
    _check_login_rate(request.client.host if request.client else "unknown")
    result = login(req.email, req.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    _transfer_guest_profiles(x_guest_id, result["user"]["user_id"])
    return AuthResponse(token=result["token"], user=result["user"])


@app.get("/auth/me")
async def get_me(user_id: str = Depends(get_current_user)):
    """Get current user info (validates token)."""
    sub = get_subscription(user_id)
    limits = get_plan_limits(sub["plan"])
    return {"user_id": user_id, "plan": sub["plan"], "limits": limits}


@app.post("/auth/refresh")
async def refresh_session(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Silently refresh a valid JWT token, extending the session.

    Called on every page load. If the token is still valid, returns a new one
    with a fresh expiry window. Front-end stores the new token, so the user
    stays logged in indefinitely as long as they visit within the expiry window.
    No auth dependency — the token IS the credential here.
    """
    token = credentials.token if credentials else None
    if not token:
        raise HTTPException(status_code=401, detail="No token provided")
    result = refresh_token(token)
    if not result:
        raise HTTPException(status_code=401, detail="Token invalid or expired")
    return result


# ── Billing Endpoints ──

class CheckoutRequest(BaseModel):
    plan: str = Field(..., description="Plan to subscribe to: starter or pro")


@app.post("/billing/checkout")
async def billing_checkout(
    req: CheckoutRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """Create a Stripe Checkout session for upgrading."""
    if req.plan not in ("starter", "pro"):
        raise HTTPException(status_code=400, detail="Invalid plan")

    base_url = str(request.base_url).rstrip("/")
    url = create_checkout_session(
        user_id=user_id,
        email="",  # Stripe already has it from customer
        plan=req.plan,
        success_url=f"{base_url}/?billing=success",
        cancel_url=f"{base_url}/?billing=cancelled",
    )
    if not url:
        raise HTTPException(status_code=503, detail="Billing not configured. Set STRIPE_SECRET_KEY.")
    return {"checkout_url": url}


@app.post("/billing/portal")
async def billing_portal(
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """Create a Stripe Customer Portal session for managing subscriptions."""
    base_url = str(request.base_url).rstrip("/")
    url = create_portal_session(user_id, return_url=base_url)
    if not url:
        raise HTTPException(status_code=400, detail="No active subscription found")
    return {"portal_url": url}


@app.get("/billing/plan")
async def get_plan(user_id: str = Depends(get_current_user)):
    """Get current plan and limits."""
    sub = get_subscription(user_id)
    limits = get_plan_limits(sub["plan"])
    return {"plan": sub["plan"], "limits": limits}


@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events (no auth — verified by signature)."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        result = handle_webhook(payload, sig)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Request/Response Models ──

class CreateProfileRequest(BaseModel):
    name: str = Field(..., description="Name for this voice profile")
    base_description: str = Field(
        default="",
        description="Initial voice description (or leave blank to build through teaching)",
    )


class ProfileResponse(BaseModel):
    profile_id: str
    owner_id: str
    name: str
    base_description: str
    created_at: str
    last_taught: str
    refinement_count: int


class TeachRequest(BaseModel):
    message: str = Field(..., description="The teaching message")
    command: str = Field(
        default="dialogue",
        description="One of: dialogue, examine, demo, correct, example, principle, voice, never, try",
    )
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Recent conversation messages [{role, content}, ...]",
    )


class TeachResponse(BaseModel):
    response: str
    refinement_saved: bool
    refinement_type: Optional[str] = None
    refinement_count: int


class WriteRequest(BaseModel):
    instruction: str = Field(..., description="What to write")
    context: str = Field(default="", description="Optional context, notes, or outline")


class WriteResponse(BaseModel):
    text: str
    profile_name: str
    refinement_count: int


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="The text to analyze against the voice profile")


class AnalyzeResponse(BaseModel):
    analysis: str
    profile_name: str


class AnalyzeSamplesRequest(BaseModel):
    samples: list[str] = Field(
        ..., description="Writing samples to analyze (3-5 recommended)"
    )


class AnalyzeSamplesResponse(BaseModel):
    voice_description: str


class ExportResponse(BaseModel):
    markdown: str
    profile_name: str
    refinement_count: int


# ── Profile Endpoints ──

@app.post("/profiles", response_model=ProfileResponse)
async def create_voice_profile(
    req: CreateProfileRequest,
    user_id: str = Depends(get_current_user),
):
    """Create a new voice profile."""
    # Check plan limits
    sub = get_subscription(user_id)
    limits = get_plan_limits(sub["plan"])
    existing = list_profiles(user_id)
    if len(existing) >= limits["profiles"]:
        raise HTTPException(
            status_code=403,
            detail=f"Your {sub['plan']} plan allows {limits['profiles']} voice profile(s). Upgrade to create more.",
        )

    profile = create_profile(
        owner_id=user_id,
        name=req.name,
        base_description=req.base_description,
    )
    return ProfileResponse(**profile.to_dict())


@app.get("/profiles", response_model=list[ProfileResponse])
async def list_voice_profiles(user_id: str = Depends(get_current_user)):
    """List all voice profiles for the current user."""
    profiles = list_profiles(user_id)
    return [ProfileResponse(**p.to_dict()) for p in profiles]


@app.get("/profiles/{profile_id}", response_model=ProfileResponse)
async def get_voice_profile(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """Get a specific voice profile."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    return ProfileResponse(**profile.to_dict())


@app.delete("/profiles/{profile_id}")
async def delete_voice_profile(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """Delete a voice profile and all its data."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    delete_profile(profile_id)
    return {"deleted": True, "profile_id": profile_id}


@app.get("/profiles/{profile_id}/refinements")
async def get_refinements(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """Get all refinements for a profile."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    refinements = load_refinements(profile_id)
    return {"profile_id": profile_id, "refinements": refinements, "count": len(refinements)}


# ── Conversation Sessions ──

class ConversationSaveRequest(BaseModel):
    session_id: str
    messages: list


@app.post("/profiles/{profile_id}/conversations")
async def save_conversation(
    profile_id: str,
    req: ConversationSaveRequest,
    user_id: str = Depends(get_current_user),
):
    """Save (or update) a conversation session for a profile."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    save_conversation_session(profile_id, req.session_id, req.messages)
    return {"saved": True, "session_id": req.session_id}


@app.get("/profiles/{profile_id}/conversations")
async def list_conversations(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """List all conversation sessions for a profile."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    sessions = list_conversation_sessions(profile_id)
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/profiles/{profile_id}/conversations/{session_id}")
async def get_conversation(
    profile_id: str,
    session_id: str,
    user_id: str = Depends(get_current_user),
):
    """Load a specific conversation session."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    session = load_conversation_session(profile_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ── File Upload (Bring Your Voice) ──

from fastapi import UploadFile, File

@app.post("/profiles/{profile_id}/upload")
async def upload_writing_sample(
    profile_id: str,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    """Upload a writing sample file (.txt, .md, .html)."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    # Validate file type
    allowed = (".txt", ".md", ".html", ".text")
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail=f"File type not supported. Use: {', '.join(allowed)}")

    # Size limit: 500KB
    content = await file.read()
    if len(content) > 500_000:
        raise HTTPException(status_code=400, detail="File too large (max 500KB)")

    try:
        path = save_uploaded_file(profile_id, file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"filename": path.name, "size": len(content), "saved": True}


@app.get("/profiles/{profile_id}/uploads")
async def get_uploads(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """List uploaded writing samples for a profile."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    files = list_uploaded_files(profile_id)
    return {"profile_id": profile_id, "files": files, "count": len(files)}


@app.post("/profiles/{profile_id}/ingest")
async def ingest_uploads(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """Analyze all uploaded files and extract voice refinements.

    This is the 'Bring Your Voice' automation — upload your writing,
    then hit ingest, and the system extracts examples, principles,
    anti-patterns, and voice notes automatically.
    """
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    new_refinements = ingest_writing_samples(profile_id, LLM_CALL)
    updated = load_profile(profile_id)

    return {
        "profile_id": profile_id,
        "new_refinements": len(new_refinements),
        "total_refinements": updated.refinement_count if updated else 0,
        "refinements": new_refinements,
    }


# ── Teach Endpoint ──

@app.post("/profiles/{profile_id}/teach", response_model=TeachResponse)
async def teach_voice(
    profile_id: str,
    req: TeachRequest,
    user_id: str = Depends(get_current_user),
):
    """Teach a voice profile through dialogue and correction.

    Commands:
    - dialogue: Open conversation with the voice (default)
    - examine: Test the voice's understanding
    - demo / try: Have the voice write on a topic
    - correct: Correct something the voice got wrong
    - example: Provide an example of the voice
    - principle: Teach a core principle
    - voice: Add a voice/tone note
    - never: Mark an anti-pattern
    """
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    valid_commands = {
        "auto", "dialogue", "examine", "demo", "try",
        "correct", "example", "principle", "voice", "never",
    }
    if req.command not in valid_commands:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid command '{req.command}'. Valid: {sorted(valid_commands)}",
        )

    result = teach_interaction(
        profile_id=profile_id,
        message=req.message,
        command=req.command,
        conversation_history=req.conversation_history,
        llm_call=LLM_CALL,
    )

    # Reload profile for updated count
    updated = load_profile(profile_id)
    return TeachResponse(
        response=result["response"],
        refinement_saved=result["refinement_saved"],
        refinement_type=result["refinement_type"],
        refinement_count=updated.refinement_count if updated else 0,
    )


# ── Write Endpoint ──

@app.post("/profiles/{profile_id}/write", response_model=WriteResponse)
async def write_in_voice(
    profile_id: str,
    req: WriteRequest,
    user_id: str = Depends(get_current_user),
):
    """Generate text using a trained voice profile."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    text = write_with_voice(
        profile_id=profile_id,
        instruction=req.instruction,
        llm_call=LLM_CALL,
        context=req.context,
    )

    return WriteResponse(
        text=text,
        profile_name=profile.name,
        refinement_count=profile.refinement_count,
    )


# ── Analyze Endpoint ──

@app.post("/profiles/{profile_id}/analyze", response_model=AnalyzeResponse)
async def analyze_against_voice(
    profile_id: str,
    req: AnalyzeRequest,
    user_id: str = Depends(get_current_user),
):
    """Analyze text against a voice profile for consistency."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    analysis = analyze_text(
        profile_id=profile_id,
        text=req.text,
        llm_call=LLM_CALL,
    )

    return AnalyzeResponse(
        analysis=analysis,
        profile_name=profile.name,
    )


# ── Find Your Voice (Sample Analysis) ──

@app.post("/analyze-samples", response_model=AnalyzeSamplesResponse)
async def find_your_voice(
    req: AnalyzeSamplesRequest,
    user_id: str = Depends(get_current_user),
):
    """Analyze writing samples to discover voice patterns.

    The 'Find Your Voice' entry point — paste writing you admire,
    and the system identifies the patterns to build a starter profile.
    """
    if len(req.samples) < 1:
        raise HTTPException(status_code=400, detail="Provide at least one writing sample")

    description = analyze_samples(
        samples=req.samples,
        llm_call=LLM_CALL,
    )

    return AnalyzeSamplesResponse(voice_description=description)


# ── Export ──

@app.get("/profiles/{profile_id}/export", response_model=ExportResponse)
async def export_profile(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """Export a voice profile as a portable markdown document.

    Take this to any AI tool — Claude Projects, ChatGPT Custom GPTs,
    system prompts, or any LLM interface.
    """
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    markdown = export_voice_profile(profile_id)
    refinements = load_refinements(profile_id)

    return ExportResponse(
        markdown=markdown,
        profile_name=profile.name,
        refinement_count=len(refinements),
    )


async def _synthesize_skill_md(profile, voice_text: str, refinement_count: int) -> str:
    """Synthesize a Claude Code SKILL.md from a voice profile."""
    client = _anthropic.AsyncAnthropic()

    prompt = f"""You are converting a writing voice profile into a Claude Code SKILL.md file.

SKILL.md files tell Claude when and how to apply a skill. The exact format is:

---
description: <one sentence>
---
<skill content>

---
VOICE NAME: {profile.name}
REFINEMENTS: {refinement_count} teaching sessions

FULL VOICE SPECIFICATION:
{voice_text}
---

Produce a complete, self-contained SKILL.md that:

1. Opens with frontmatter (exactly: three dashes, description line, three dashes).
   The description MUST begin: "Write in {profile.name}'s voice."
   Include trigger phrases: write, draft, rewrite, edit prose, make this sound like me.

2. Has a # {profile.name} heading.

3. Has a ## Core Style section — 5–8 bullet points of essential stylistic principles,
   written as direct instructions (e.g. "Use fragments when the rhythm calls for them",
   NOT "This voice uses fragments"). Drawn directly from the principles in the specification.

4. Has a ## What This Voice Avoids section — bullet list of anti-patterns,
   written as direct instructions ("Never use transitional words like 'however' or 'moreover'").

5. Has a ## Voice in Practice section — 2–3 short examples that demonstrate the style.
   Use actual examples from the specification if present; otherwise extrapolate faithfully.

6. Ends with a ## Instructions line: "When asked to write, draft, or edit in this voice,
   follow the style above precisely. Do not soften, modernize, or normalize the style."

Rules:
- Write every bullet as a direct instruction, not a description.
- Be specific. No vague adjectives like "nuanced", "distinctive", "powerful", "sophisticated".
- Do not restate the same principle twice.
- Do not pad with commentary about the export process."""

    response = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1800,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


@app.get("/profiles/{profile_id}/export/skill")
async def export_profile_as_skill(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """Synthesize and export a voice profile as a Claude Code SKILL.md file.

    Unlike the raw markdown export, this uses Claude to distil the voice
    into a concise, directly-instructional skill file designed to be
    dropped into ~/.claude/skills/ or used as a Claude Projects system prompt.
    """
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    if profile.refinement_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Teach the voice at least one refinement before exporting as a skill.",
        )

    voice_text = get_full_voice_text(profile_id)
    refinements = load_refinements(profile_id)

    skill_content = await _synthesize_skill_md(profile, voice_text, len(refinements))

    # Sanitize name for filename
    safe_name = "".join(c if c.isalnum() or c in "- " else "" for c in profile.name)
    safe_name = safe_name.strip().replace(" ", "-").lower() or "voice"
    filename = f"{safe_name}.md"

    return {"filename": filename, "content": skill_content, "profile_name": profile.name}


# ── Voice Text (for debugging / advanced use) ──

@app.get("/profiles/{profile_id}/voice-text")
async def get_voice_text(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """Get the assembled voice prompt (base + refinements).

    Useful for debugging or for users who want to see
    exactly what the AI receives when writing in their voice.
    """
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    voice_text = get_full_voice_text(profile_id)
    return {"profile_id": profile_id, "voice_text": voice_text}


# ── Health ──

# ── Council ──────────────────────────────────────────────────────────────────

class CouncilRequest(BaseModel):
    question: str = Field(..., min_length=4, max_length=2000)
    mode: str = Field("advice")
    thinkers: list[str] = Field(default_factory=lambda: list(COUNCIL_NAMES))
    # For writing mode with prose critique — prepend as context
    prose: str = Field(default="", max_length=4000)
    # Swarm mode: 40-agent philosophical population instead of 6 named thinkers
    swarm: bool = Field(default=False)
    n_agents: int = Field(default=40, ge=10, le=60)


@app.post("/council")
async def convene_council(
    req: CouncilRequest,
    user_id: str = Depends(get_current_user),
):

    if req.mode not in VALID_MODES:
        raise HTTPException(400, f"mode must be one of {sorted(VALID_MODES)}")

    # For writing mode: if prose is provided, fold it into the question
    question = req.question
    if req.mode == "writing" and req.prose.strip():
        question = (
            f"Please critique this passage and advise how to make it truer, clearer, "
            f"and more powerful:\n\n{req.prose.strip()}\n\n"
            f"Additional context from the writer: {req.question}" if req.question.strip()
            else f"Please critique this passage and advise how to make it truer, clearer, "
                 f"and more powerful:\n\n{req.prose.strip()}"
        )

    try:
        if req.swarm:
            # 40-agent philosophical population swarm
            result = await run_council_swarm(
                question=question,
                mode=req.mode,
                n_agents=req.n_agents,
            )
        else:
            # Original 6-thinker named council
            if not (2 <= len(req.thinkers) <= 6):
                raise HTTPException(400, "Select between 2 and 6 thinkers")
            unknown = [n for n in req.thinkers if n not in COUNCIL_NAMES]
            if unknown:
                raise HTTPException(400, f"Unknown thinkers: {unknown}")
            result = await run_council_agents(
                question=question,
                mode=req.mode,
                thinker_names=req.thinkers,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Council deliberation failed: {e}")

    return result


@app.get("/council/thinkers")
async def list_thinkers():
    """Return available council members. No auth required."""
    return {"thinkers": COUNCIL_NAMES, "modes": sorted(VALID_MODES)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "llm_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


# ── Static Files & Frontend ──

STATIC_DIR = Path(__file__).parent / "static"

@app.get("/")
async def serve_landing():
    return FileResponse(STATIC_DIR / "landing.html")

@app.get("/app")
async def serve_app():
    return FileResponse(STATIC_DIR / "index.html")


# Mount static files AFTER all API routes
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Run ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
