"""
Luxa — FastAPI Application

The API layer over voice_engine.py. Stateless endpoints for
teaching, writing, analyzing, and managing voice profiles.
"""

import asyncio
import os
from datetime import datetime
from typing import Optional

from pathlib import Path
import anthropic as _anthropic

from fastapi import FastAPI, HTTPException, Depends, Header, Request, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from auth import register, login, verify_token, create_token, refresh_token, ensure_dirs as ensure_auth_dirs, create_reset_token, consume_reset_token, send_reset_email
from billing import (
    get_subscription, get_plan_limits, create_checkout_session,
    check_council_credits, consume_credits, get_credits_remaining,
    COUNCIL_CREDIT_COSTS,
    create_portal_session, handle_webhook,
    ensure_dirs as ensure_billing_dirs,
)
from council_cache import get_cached_response, store_response as cache_store_response
from voice_engine import (
    create_profile,
    load_profile,
    list_profiles,
    delete_profile,
    load_refinements,
    _rewrite_refinements,
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
    delete_conversation_session,
    translate_with_voice,
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


# ── Password Reset ──

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    password: str

@app.post("/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest, request: Request):
    """Send a password reset email. Always returns 200 to avoid email enumeration."""
    _check_login_rate(request.client.host if request.client else "unknown")
    email = req.email.lower().strip()
    token = create_reset_token(email)
    if token:
        base_url = str(request.base_url).rstrip("/")
        try:
            send_reset_email(email, token, base_url)
        except Exception as e:
            print(f"[SMTP error] {e}")
    return {"ok": True}

@app.post("/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    """Reset password with a valid token."""
    try:
        ok = consume_reset_token(req.token, req.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(400, "Reset link is invalid or has expired.")
    return {"ok": True}


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


class TranslateRequest(BaseModel):
    source_text: str = Field(..., description="The passage to translate/render", max_length=6000)
    source_language: str = Field(default="", description="Source language hint (e.g. 'Ancient Greek', 'Latin')")
    notes: str = Field(default="", description="Optional context (e.g. 'From Nicomachean Ethics Book II')", max_length=1000)


class TranslateResponse(BaseModel):
    text: str
    profile_name: str
    source_language: str


class ConverseRequest(BaseModel):
    message: str = Field(..., description="What the user said", max_length=8000)
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Recent conversation messages [{role, content}, ...]",
    )


class ConverseResponse(BaseModel):
    response: str
    intent: str  # "chat" | "write" | "translate"
    refinement_saved: bool = False
    refinement_type: Optional[str] = None
    refinement_count: int = 0


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


@app.patch("/profiles/{profile_id}/name")
async def rename_profile(
    profile_id: str,
    user_id: str = Depends(get_current_user),
    name: str = Body(..., embed=True),
):
    """Rename a voice profile."""
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")
    if len(name) > 60:
        raise HTTPException(400, "Name too long (max 60 chars)")
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(403, "Not your profile")
    profile.name = name
    update_profile_metadata(profile)
    return {"name": name}


@app.patch("/profiles/{profile_id}/description")
async def update_profile_description(
    profile_id: str,
    user_id: str = Depends(get_current_user),
    description: str = Body(..., embed=True),
):
    """Update a voice profile's base description."""
    description = description.strip()
    if len(description) > 2000:
        raise HTTPException(400, "Description too long (max 2000 chars)")
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(403, "Not your profile")
    profile.base_description = description
    update_profile_metadata(profile)
    # Also update the base.md file that the voice engine reads
    base_path = Path("data/profiles") / profile_id / "base.md"
    base_path.write_text(description)
    return {"description": description}


@app.patch("/profiles/{profile_id}/avatar")
async def set_profile_avatar(
    profile_id: str,
    user_id: str = Depends(get_current_user),
    avatar: str = Body(..., embed=True),
):
    """Set a preset avatar emoji key for a voice profile."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(403, "Not your profile")
    profile.avatar = avatar
    update_profile_metadata(profile)
    return {"avatar": avatar}


@app.post("/profiles/{profile_id}/avatar/upload")
async def upload_profile_avatar(
    profile_id: str,
    user_id: str = Depends(get_current_user),
    file: UploadFile = File(...),
):
    """Upload a custom image avatar for a voice profile."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(403, "Not your profile")
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        raise HTTPException(400, "Must be JPG, PNG, WebP, or GIF")
    data = await file.read()
    if len(data) > 2 * 1024 * 1024:  # 2MB cap
        raise HTTPException(400, "Image must be under 2MB")
    avatars_dir = Path(__file__).parent / "static" / "avatars"
    avatars_dir.mkdir(parents=True, exist_ok=True)
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
    path = avatars_dir / f"{profile_id}.{ext}"
    path.write_bytes(data)
    profile.avatar = f"custom:{ext}"
    update_profile_metadata(profile)
    return {"avatar": profile.avatar, "url": f"/static/avatars/{profile_id}.{ext}"}


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
    # Add index to each refinement so the UI can reference them
    for i, r in enumerate(refinements):
        r["index"] = i
    return {"profile_id": profile_id, "refinements": refinements, "count": len(refinements)}


@app.delete("/profiles/{profile_id}/refinements/{index}")
async def delete_refinement(
    profile_id: str,
    index: int,
    user_id: str = Depends(get_current_user),
):
    """Delete a refinement by index."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    refinements = load_refinements(profile_id)
    if index < 0 or index >= len(refinements):
        raise HTTPException(status_code=404, detail="Refinement not found")
    refinements.pop(index)
    _rewrite_refinements(profile_id, refinements)
    profile.refinement_count = len(refinements)
    update_profile_metadata(profile)
    return {"count": len(refinements)}


class EditRefinementRequest(BaseModel):
    content: str
    type: str | None = None


@app.put("/profiles/{profile_id}/refinements/{index}")
async def edit_refinement(
    profile_id: str,
    index: int,
    req: EditRefinementRequest,
    user_id: str = Depends(get_current_user),
):
    """Edit a refinement's content and optionally its type."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    refinements = load_refinements(profile_id)
    if index < 0 or index >= len(refinements):
        raise HTTPException(status_code=404, detail="Refinement not found")
    refinements[index]["content"] = req.content
    if req.type:
        refinements[index]["type"] = req.type
    _rewrite_refinements(profile_id, refinements)
    return {"refinement": refinements[index], "count": len(refinements)}



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


@app.delete("/profiles/{profile_id}/conversations/{session_id}")
async def delete_conversation(
    profile_id: str,
    session_id: str,
    user_id: str = Depends(get_current_user),
):
    """Delete a specific conversation session."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    deleted = delete_conversation_session(profile_id, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


# ── File Upload (Bring Your Voice) ──

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


# ── Translate / Render Endpoint ──

@app.post("/profiles/{profile_id}/translate", response_model=TranslateResponse)
async def translate_in_voice(
    profile_id: str,
    req: TranslateRequest,
    user_id: str = Depends(get_current_user),
):
    """Translate or render a text through a trained voice profile."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    text = translate_with_voice(
        profile_id=profile_id,
        source_text=req.source_text,
        llm_call=LLM_CALL,
        source_language=req.source_language,
        notes=req.notes,
    )

    return TranslateResponse(
        text=text,
        profile_name=profile.name,
        source_language=req.source_language or "auto-detected",
    )


# ── Unified Converse Endpoint ──

_INTENT_CLASSIFIER = None

def _get_intent_classifier():
    global _INTENT_CLASSIFIER
    if _INTENT_CLASSIFIER is None and os.environ.get("ANTHROPIC_API_KEY"):
        _INTENT_CLASSIFIER = make_claude_caller(
            model="claude-haiku-4-20250414", max_tokens=60, temperature=0.0,
        )
    return _INTENT_CLASSIFIER


_INTENT_PROMPT = """Classify this user message into exactly one category.

WRITE — The user gives you an explicit instruction to generate/draft/compose NEW text for them. Must contain a clear directive like "write", "draft", "compose", or be phrased as a command to produce text.
  Examples: "Write an opening paragraph about grief", "Draft a tweet about...", "An essay on why...", "A short story about..."

TRANSLATE — The user pasted text in another language or dense/archaic text and wants it rendered in plain language or their voice. Look for non-Latin scripts, explicit "translate"/"render" requests, or pasted passages that are clearly source material to be re-expressed.
  Examples: "τὸ γὰρ αὐτὸ νοεῖν ἐστίν τε καὶ εἶναι", "Translate this passage from Heidegger...", "Render this in my voice: [dense academic text]"

CHAT — Everything else: conversation, feedback, corrections, teaching rules, asking questions, discussing preferences. IMPORTANTLY: if the user pastes their own writing as an example, a sample, or to teach you about their style, this is CHAT — they want you to learn from it, not rewrite it. Look for signals like "Example:", "Here's my writing:", "This is how I write:", "Here's a sample:", or simply a block of prose without any instruction to generate new text.
  Examples: "I never use semicolons", "That last piece was too formal", "What do you know about my voice?", "Show me an example", "Example: The morning came without ceremony...", "Here's something I wrote: ...", [a paragraph of prose with no explicit write/draft instruction]

CRITICAL: When in doubt between WRITE and CHAT, choose CHAT. Only classify as WRITE when there is an unambiguous instruction to generate new text.

Respond with exactly one word: WRITE, TRANSLATE, or CHAT"""


@app.post("/profiles/{profile_id}/converse", response_model=ConverseResponse)
async def converse_with_voice(
    profile_id: str,
    req: ConverseRequest,
    user_id: str = Depends(get_current_user),
):
    """Unified voice conversation — classifies intent and routes to write, translate, or chat."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    # ── Classify intent ──
    intent = "chat"  # default fallback
    msg_stripped = req.message.strip()
    msg_lower = msg_stripped.lower()

    # Hard override: explicit writing examples are always CHAT, never WRITE
    _example_signals = msg_lower.startswith(("example:", "here's my writing", "here is my writing",
                                              "here's a sample", "here's something i wrote", "sample:"))

    if _example_signals:
        intent = "chat"
    else:
        classifier = _get_intent_classifier()
        if classifier:
            try:
                classification = classifier([
                    {"role": "user", "content": f"{_INTENT_PROMPT}\n\nUser message:\n{req.message[:2000]}"},
                ]).strip().upper()
                if classification in ("WRITE", "TRANSLATE", "CHAT"):
                    intent = classification.lower()
            except Exception:
                pass  # fall back to chat

    # ── Route based on intent ──
    if intent == "write":
        text = write_with_voice(
            profile_id=profile_id,
            instruction=req.message,
            llm_call=LLM_CALL,
        )
        updated = load_profile(profile_id)
        return ConverseResponse(
            response=text,
            intent="write",
            refinement_count=updated.refinement_count if updated else 0,
        )

    elif intent == "translate":
        text = translate_with_voice(
            profile_id=profile_id,
            source_text=req.message,
            llm_call=LLM_CALL,
        )
        updated = load_profile(profile_id)
        return ConverseResponse(
            response=text,
            intent="translate",
            refinement_count=updated.refinement_count if updated else 0,
        )

    else:  # chat
        result = teach_interaction(
            profile_id=profile_id,
            message=req.message,
            command="auto",
            conversation_history=req.conversation_history,
            llm_call=LLM_CALL,
        )
        updated = load_profile(profile_id)
        return ConverseResponse(
            response=result["response"],
            intent="chat",
            refinement_saved=result["refinement_saved"],
            refinement_type=result["refinement_type"],
            refinement_count=updated.refinement_count if updated else 0,
        )


# ── Voice Dimensions Endpoint ──

class DimensionsResponse(BaseModel):
    dimensions: dict  # {"precision": 8, "lyricism": 6, ...}
    summary: str


_DIMENSIONS_PROMPT = """Analyze this voice profile and score it on exactly these 6 dimensions.
Each score is 1–10 (integer only).

Dimensions:
- precision: Economy of language — how tight and deliberate is every word?
- lyricism: Musicality and rhythm — does the prose have cadence, flow, beauty of sound?
- complexity: Sentence architecture — how layered and structurally varied are the sentences?
- imagery: Sensory and figurative language — how vivid, metaphorical, image-rich?
- formality: Register and diction — how elevated, academic, formal vs conversational?
- irony: Distance and indirection — how much does the voice use irony, understatement, misdirection?

After the scores, write a 1–2 sentence summary of this voice's character.

Respond in EXACTLY this JSON format, nothing else:
{"precision": N, "lyricism": N, "complexity": N, "imagery": N, "formality": N, "irony": N, "summary": "..."}

Voice profile to analyze:
"""


@app.post("/profiles/{profile_id}/dimensions", response_model=DimensionsResponse)
async def get_voice_dimensions(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """Score a voice profile on 6 dimensions and return a summary."""
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")

    voice_text = get_full_voice_text(profile_id)
    if not voice_text or len(voice_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Not enough voice data yet")

    import json as _json
    raw = LLM_CALL([
        {"role": "user", "content": _DIMENSIONS_PROMPT + voice_text[:4000]},
    ])

    try:
        parsed = _json.loads(raw.strip())
        dims = {k: max(1, min(10, int(parsed.get(k, 5)))) for k in
                ["precision", "lyricism", "complexity", "imagery", "formality", "irony"]}
        summary = parsed.get("summary", "")
    except Exception:
        dims = {"precision": 5, "lyricism": 5, "complexity": 5, "imagery": 5, "formality": 5, "irony": 5}
        summary = raw.strip()[:200]

    return DimensionsResponse(dimensions=dims, summary=summary)


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


async def _synthesize_style_guide(profile, voice_text: str, refinement_count: int) -> str:
    """Synthesize a structured AI Style Guide from a voice profile."""
    client = _anthropic.AsyncAnthropic()

    prompt = f"""You are converting a writing voice profile into a structured AI Style Guide —
a portable document the writer can use with any AI tool to make it write like them.

VOICE NAME: {profile.name}
REFINEMENTS: {refinement_count} teaching sessions

FULL VOICE SPECIFICATION:
{voice_text}

---

Produce a complete, well-organized Style Guide with exactly these 8 sections.
For each section, synthesize from the voice specification above. Be concrete and specific —
no vague adjectives like "nuanced" or "distinctive." Every bullet should be actionable.

# AI Style Guide: {profile.name}

## 1. Voice & Tone
How this writing should feel at its best. Describe with tensions and boundaries,
not flattering adjectives. Cover: formality level, emotional temperature, humor,
whether the voice is reportorial / essayistic / intimate / skeptical / lyrical, etc.
Write 5–8 bullets.

## 2. Structure
How pieces in this voice typically move. How do they open? How quickly do they reach
the point? What is the arc — anecdote to argument, tension to resolution, concrete to abstract?
Write 4–6 bullets.

## 3. Sentence-Level Preferences
What makes a sentence sound right in this voice? Sentence length, rhythm,
tolerance for abstraction, diction choices, punctuation tendencies.
Include positive and negative examples where possible. Write 5–8 bullets.

## 4. Signature Moves
What does this voice do especially well? Name 2–4 recurring patterns or techniques
with a brief description of each.

## 5. Anti-Patterns / Blacklist
What the model must avoid. Be specific — name exact patterns and their fixes.
Format as a list of "Pattern → Fix" entries. Include at least 5.

## 6. Positive Examples
2–3 short passages (real or synthesized from the specification) that demonstrate
the voice working well. For each, add one sentence explaining why it works.

## 7. Negative Examples
2–3 short passages showing what this voice should NOT sound like.
For each, explain what feels wrong.

## 8. Revision Checklist
5–8 yes/no questions to evaluate whether a draft captures this voice.
Each question should test for a specific quality from the sections above.

Rules:
- Write every instruction as a direct command, not a description.
- Be ruthlessly specific. If the voice specification names real preferences, use them.
- If the specification is thin on a section, extrapolate faithfully from what IS present,
  but mark extrapolations with "(inferred)" so the writer can verify.
- Do not pad with commentary. Output only the style guide."""

    response = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


@app.get("/profiles/{profile_id}/export/style-guide")
async def export_profile_as_style_guide(
    profile_id: str,
    user_id: str = Depends(get_current_user),
):
    """Synthesize and export a voice profile as a structured AI Style Guide.

    Organizes the voice into 8 sections: voice & tone, structure, sentence preferences,
    signature moves, anti-patterns, positive examples, negative examples, revision checklist.
    Portable to any AI tool.
    """
    profile = load_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not your profile")
    if profile.refinement_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Teach the voice at least one refinement before generating a style guide.",
        )

    voice_text = get_full_voice_text(profile_id)
    refinements = load_refinements(profile_id)

    guide_content = await _synthesize_style_guide(profile, voice_text, len(refinements))

    safe_name = "".join(c if c.isalnum() or c in "- " else "" for c in profile.name)
    safe_name = safe_name.strip().replace(" ", "-").lower() or "voice"
    filename = f"{safe_name}-style-guide.md"

    return {"filename": filename, "content": guide_content, "profile_name": profile.name}


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


class ClarifyRequest(BaseModel):
    question: str = Field(..., min_length=4, max_length=2000)
    mode: str = Field("auto")


_CLARIFY_CALLER = None

def _get_clarify_caller():
    global _CLARIFY_CALLER
    if _CLARIFY_CALLER is None and os.environ.get("ANTHROPIC_API_KEY"):
        _CLARIFY_CALLER = make_claude_caller(
            model="claude-haiku-4-20250414", max_tokens=600, temperature=0.3,
        )
    return _CLARIFY_CALLER


async def _classify_mode(question: str) -> str:
    """Use Haiku to auto-classify a question as advice, predict, or writing."""
    import asyncio as _asyncio
    import json as _json

    caller = _get_clarify_caller()
    if not caller:
        return "advice"  # safe fallback

    prompt = f"""Classify this question into exactly one mode for a philosophical council deliberation.

Question: "{question}"

Modes:
- "advice" — the person is asking what they should DO, how to live, ethical dilemmas, personal decisions, meaning, values, relationships
- "predict" — the person is asking what WILL HAPPEN, forecasting, consequences, trends, geopolitics, future scenarios
- "writing" — the person is asking about the CRAFT of writing, prose style, how to write about something, literary analysis, critique of text

Most questions are "advice". Only use "predict" when the question explicitly asks about future outcomes or consequences. Only use "writing" when the question is specifically about the craft or technique of writing.

Respond with ONLY one word: advice, predict, or writing"""

    try:
        text = await _asyncio.to_thread(caller, prompt)
        mode = text.strip().lower().replace('"', '').replace("'", "")
        if mode in ("advice", "predict", "writing"):
            return mode
    except Exception:
        pass
    return "advice"


@app.post("/council/clarify")
async def clarify_question(
    req: ClarifyRequest,
    user_id: str = Depends(get_current_user),
):
    """Determine if a question needs clarifying context before deliberation.

    Uses a fast Haiku call to detect personal/situational questions and generate
    2-4 follow-up questions to gather the context the council needs to give
    informed, specific advice rather than generic philosophy.
    """
    import json as _json

    caller = _get_clarify_caller()
    if not caller:
        return {"needs_clarification": False, "questions": []}

    # Auto-classify mode if needed
    mode = req.mode
    if mode == "auto":
        mode = await _classify_mode(req.question)

    try:
        prompt = f"""Analyze this question that someone wants to ask a council of philosophers:

"{req.question}"

Mode: {mode}

Determine: does this question describe a PERSONAL SITUATION or SPECIFIC DECISION that requires understanding the person's circumstances to give good advice? Or is it an ABSTRACT/PHILOSOPHICAL question that can be answered without personal context?

Examples of questions that NEED clarification:
- "Should I accept the PhD offer?" (need: current situation, goals, field, alternatives)
- "Is it time to leave my job?" (need: why unhappy, financial situation, what next)
- "Should I forgive my father?" (need: what happened, current relationship, what forgiveness means to them)
- "How do I handle this conflict with my partner?" (need: nature of conflict, relationship context)

Examples that DON'T need clarification:
- "Is democracy the best form of government?" (abstract)
- "What is the meaning of suffering?" (philosophical)
- "Should AI be regulated?" (general policy)
- "Is revenge ever justified?" (abstract ethical)

If the question NEEDS clarification, respond with JSON:
{{"needs_clarification": true, "questions": ["question1", "question2", "question3"]}}

Generate 2-4 questions that are:
- Warm and conversational, not clinical
- Specific to what the council would need to know
- Short (one sentence each)
- Designed to understand the person's actual situation, stakes, and values

If the question does NOT need clarification:
{{"needs_clarification": false, "questions": []}}

Respond with ONLY the JSON object, no other text."""

        import asyncio
        text = await asyncio.to_thread(caller, prompt)
        text = text.strip()
        result = _json.loads(text)
        return {
            "needs_clarification": bool(result.get("needs_clarification", False)),
            "questions": result.get("questions", [])[:4],
        }
    except Exception as exc:
        import traceback; traceback.print_exc()
        # If anything fails, skip clarification — don't block the user
        return {"needs_clarification": False, "questions": []}


class CouncilRequest(BaseModel):
    question: str = Field(..., min_length=4, max_length=2000)
    mode: str = Field("auto")
    thinkers: list[str] = Field(default_factory=lambda: list(COUNCIL_NAMES))
    prose: str = Field(default="", max_length=4000)
    # Clarifying context gathered from intake questions
    context: str = Field(default="", max_length=4000)
    # council_tier: "lite" | "full" | "swarm"
    council_tier: str = Field(default="full")
    # Legacy swarm flag — maps to council_tier="swarm"
    swarm: bool = Field(default=False)
    n_agents: int = Field(default=40, ge=10, le=60)


# ── Swarm job storage (in-memory — fine for single-instance deploy) ──
_swarm_jobs: dict[str, dict] = {}


@app.get("/council/job/{job_id}")
async def poll_swarm_job(job_id: str, user_id: str = Depends(get_current_user)):
    """Poll for swarm job status. Returns result when done."""
    job = _swarm_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "done":
        result = job["result"]
        del _swarm_jobs[job_id]  # cleanup after retrieval
        return result
    if job["status"] == "error":
        error = job["error"]
        del _swarm_jobs[job_id]
        raise HTTPException(500, f"Swarm failed: {error}")
    return {"status": "running", "job_id": job_id}


@app.post("/council")
async def convene_council(
    req: CouncilRequest,
    user_id: str = Depends(get_current_user),
):
    sub = get_subscription(user_id)
    tier = "swarm" if req.swarm else req.council_tier
    if tier not in ("lite", "full", "swarm"):
        tier = "full"

    # ── Credit gate ──
    credit_check = check_council_credits(user_id, sub["plan"], tier)
    if not credit_check["allowed"]:
        is_guest = user_id.startswith("guest-")
        cost = credit_check["cost"]
        remaining = credit_check["credits_remaining"]
        if is_guest:
            msg = "Create a free account to get 3 Council credits per week."
        else:
            msg = (
                f"This run costs {cost} credit{'s' if cost != 1 else ''} "
                f"and you have {remaining} remaining this week. "
                f"Upgrade to unlock more."
            )
        raise HTTPException(
            status_code=429,
            detail={"message": msg, "credits_remaining": remaining,
                    "cost": cost, "is_guest": is_guest},
        )

    # Auto-classify mode if not explicitly set
    if req.mode == "auto":
        req.mode = await _classify_mode(req.question)

    if req.mode not in VALID_MODES:
        raise HTTPException(400, f"mode must be one of {sorted(VALID_MODES)}")

    question = req.question
    # Enrich question with clarifying context if provided
    if req.context.strip():
        question = (
            f"{req.question}\n\n"
            f"--- Additional context from the person asking ---\n"
            f"{req.context.strip()}"
        )
    if req.mode == "writing" and req.prose.strip():
        question = (
            f"Please critique this passage and advise how to make it truer, clearer, "
            f"and more powerful:\n\n{req.prose.strip()}\n\n"
            f"Additional context from the writer: {req.question}" if req.question.strip()
            else f"Please critique this passage and advise how to make it truer, clearer, "
                 f"and more powerful:\n\n{req.prose.strip()}"
        )

    # ── Semantic cache check (skip for swarm — too expensive to cache exactly) ──
    if tier != "swarm":
        cached = get_cached_response(question, req.mode)
        if cached:
            cached["credits_remaining"] = credit_check["credits_remaining"]
            return cached

    try:
        if tier == "swarm":
            # Swarm runs are long (2-5 min) — run as background job
            # to avoid Render's 30s proxy timeout
            import uuid as _uuid
            job_id = str(_uuid.uuid4())[:12]
            _swarm_jobs[job_id] = {"status": "running", "result": None, "error": None}

            async def _run_swarm_job():
                try:
                    r = await run_council_swarm(
                        question=question, mode=req.mode, n_agents=req.n_agents,
                    )
                    cost = COUNCIL_CREDIT_COSTS["swarm"]
                    consume_credits(user_id, cost)
                    remaining_after = get_credits_remaining(user_id, sub["plan"])
                    r["credits_remaining"] = remaining_after
                    r["credits_cost"] = cost
                    r["classified_mode"] = req.mode
                    _swarm_jobs[job_id] = {"status": "done", "result": r, "error": None}
                except Exception as e:
                    _swarm_jobs[job_id] = {"status": "error", "result": None, "error": str(e)}

            asyncio.create_task(_run_swarm_job())
            return {"job_id": job_id, "status": "running", "credits_remaining": credit_check["credits_remaining"]}

        elif tier == "lite":
            result = await run_council_agents(
                question=question, mode=req.mode,
                thinker_names=["Socrates", "Aristotle", "Machiavelli"],
                lite=True,
            )
        else:
            if not (2 <= len(req.thinkers) <= 6):
                raise HTTPException(400, "Select between 2 and 6 thinkers")
            unknown = [n for n in req.thinkers if n not in COUNCIL_NAMES]
            if unknown:
                raise HTTPException(400, f"Unknown thinkers: {unknown}")
            result = await run_council_agents(
                question=question, mode=req.mode, thinker_names=req.thinkers,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Council deliberation failed: {e}")

    # Deduct credits and cache result
    cost = COUNCIL_CREDIT_COSTS[tier]
    consume_credits(user_id, cost)
    remaining_after = get_credits_remaining(user_id, sub["plan"])
    result["credits_remaining"] = remaining_after
    result["credits_cost"] = cost
    result["classified_mode"] = req.mode

    if tier != "swarm":
        cache_store_response(question, req.mode, result)

    return result


@app.get("/council/credits")
async def get_council_credits_info(user_id: str = Depends(get_current_user)):
    """Return credit balance and costs for the current user."""
    sub = get_subscription(user_id)
    from billing import get_credits_remaining, COUNCIL_CREDIT_COSTS, get_plan_limits
    plan = sub["plan"]
    return {
        "plan": plan,
        "credits_remaining": get_credits_remaining(user_id, plan),
        "credits_per_week": get_plan_limits(plan)["council_credits_per_week"],
        "costs": COUNCIL_CREDIT_COSTS,
    }


class InterrogateRequest(BaseModel):
    thinker: str = Field(..., description="Name of the thinker to interrogate")
    question: str = Field(..., description="Follow-up question for this thinker", max_length=2000)
    deliberation_context: dict = Field(
        ...,
        description="The full council result from the original deliberation",
    )


class InterrogateResponse(BaseModel):
    thinker: str
    response: str


@app.post("/council/interrogate", response_model=InterrogateResponse)
async def interrogate_thinker(
    req: InterrogateRequest,
    user_id: str = Depends(get_current_user),
):
    """Ask a follow-up question to a specific thinker after deliberation.

    The thinker responds in character, with full awareness of the deliberation
    that just occurred — what others said, where they agreed and disagreed.
    This turns the council from a one-shot output into a conversation.
    """
    from council_agents import _PROFILES, _load_tradition, _load_memory, _search_memory

    if req.thinker not in _PROFILES:
        raise HTTPException(400, f"Unknown thinker: {req.thinker}. Valid: {COUNCIL_NAMES}")

    profile = _PROFILES[req.thinker]
    tradition_prompt = _load_tradition(profile["tradition"])

    # Build context from the deliberation
    delib = req.deliberation_context
    original_question = delib.get("question", "")
    thinkers = delib.get("thinkers", [])

    # This thinker's own position
    own = next((t for t in thinkers if t["name"] == req.thinker), None)
    own_position = own["final_position"] if own else "You did not participate."
    own_argument = own.get("key_argument", "") if own else ""

    # Other thinkers' positions
    others_summary = "\n".join(
        f"- {t['name']}: {t['final_position']}"
        for t in thinkers if t["name"] != req.thinker
    )

    synthesis = delib.get("synthesis", "")

    system_prompt = f"""You are {req.thinker}.

{profile['backstory']}

YOUR TRADITION:
{tradition_prompt[:2000]}

YOUR PSYCHOLOGY:
{profile['psychology']}

You have just participated in a council deliberation on this question:
"{original_question}"

YOUR POSITION was: {own_position}

YOUR ARGUMENT was: {own_argument[:600]}

THE OTHER THINKERS said:
{others_summary}

THE SYNTHESIS concluded:
{synthesis[:600]}

Now the person who asked the original question wants to ask you something directly.
Respond in character — with the weight of the full deliberation behind you. You know what
the others said. You can reference their arguments, agree or disagree with specific points,
and speak from the position you arrived at.

Keep your response focused and conversational — 2-4 paragraphs. This is a follow-up
dialogue, not a speech. If the question challenges your position, engage honestly.
If it asks you to go deeper, go deeper."""

    client = _anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        temperature=0.7,
        system=system_prompt,
        messages=[{"role": "user", "content": req.question}],
    )

    return InterrogateResponse(thinker=req.thinker, response=response.content[0].text)


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

@app.get("/privacy")
async def serve_privacy():
    return FileResponse(STATIC_DIR / "privacy.html")

@app.get("/app")
async def serve_app():
    return FileResponse(STATIC_DIR / "index.html")


# Mount static files AFTER all API routes
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Run ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
