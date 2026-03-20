"""
Authentication — email/password with JWT tokens.

Simple but real: bcrypt for passwords, JWT for sessions.
Replace with OAuth/SSO when ready for production.
"""

import json
import os
import secrets
import smtplib
import uuid
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import bcrypt
import jwt

# ── Config ──

SECRET_KEY = os.environ.get("JWT_SECRET", "voice-dev-secret-change-in-prod!")  # 32 bytes
TOKEN_EXPIRY_HOURS = int(os.environ.get("TOKEN_EXPIRY_HOURS", "720"))  # 30 days default
ALGORITHM = "HS256"

DATA_DIR = Path(os.environ.get("VOICE_DATA_DIR", Path(__file__).parent / "data"))
USERS_DIR = DATA_DIR / "users"


def ensure_dirs():
    USERS_DIR.mkdir(parents=True, exist_ok=True)


# ── User Storage ──

def _user_path(email: str) -> Path:
    """Safe filename from email."""
    safe = email.lower().strip().replace("@", "_at_").replace(".", "_")
    return USERS_DIR / f"{safe}.json"


def _load_user(email: str) -> Optional[dict]:
    path = _user_path(email)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save_user(user: dict):
    ensure_dirs()
    path = _user_path(user["email"])
    path.write_text(json.dumps(user, indent=2))


# ── Registration ──

def register(email: str, password: str, name: str = "") -> dict:
    """Register a new user. Returns user dict (without password hash)."""
    email = email.lower().strip()

    if _load_user(email):
        raise ValueError("An account with this email already exists.")

    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user_id = str(uuid.uuid4())[:12]

    user = {
        "user_id": user_id,
        "email": email,
        "name": name or email.split("@")[0],
        "password_hash": hashed,
        "created_at": datetime.now().isoformat(),
        "plan": "free",  # free | starter | pro
    }
    _save_user(user)

    return {k: v for k, v in user.items() if k != "password_hash"}


# ── Login ──

def authenticate(email: str, password: str) -> Optional[dict]:
    """Verify credentials. Returns user dict or None."""
    email = email.lower().strip()
    user = _load_user(email)
    if not user:
        return None

    if bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return {k: v for k, v in user.items() if k != "password_hash"}

    return None


# ── JWT Tokens ──

def create_token(user_id: str, email: str) -> str:
    """Create a JWT token."""
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token. Returns payload or None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def refresh_token(token: str) -> Optional[dict]:
    """Issue a fresh token for a still-valid existing token.

    Returns {token, user_id, email} or None if the token is invalid/expired.
    This is called silently on every page load to keep sessions alive indefinitely
    as long as the user visits at least once per TOKEN_EXPIRY_HOURS window.
    """
    payload = verify_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    email = payload.get("email")
    if not user_id or not email:
        return None
    new_token = create_token(user_id, email)
    # Load user record so we can return current profile info
    user = _load_user(email)
    user_public = {k: v for k, v in user.items() if k != "password_hash"} if user else {
        "user_id": user_id, "email": email, "name": email.split("@")[0], "plan": "free"
    }
    return {"token": new_token, "user": user_public}


# ── Password Reset ──

RESET_TOKENS_PATH = DATA_DIR / "reset_tokens.json"
RESET_TOKEN_EXPIRY_MINUTES = 30


def _load_reset_tokens() -> dict:
    if not RESET_TOKENS_PATH.exists():
        return {}
    try:
        return json.loads(RESET_TOKENS_PATH.read_text())
    except Exception:
        return {}


def _save_reset_tokens(tokens: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESET_TOKENS_PATH.write_text(json.dumps(tokens, indent=2))


def create_reset_token(email: str) -> Optional[str]:
    """Create a password-reset token for email. Returns token or None if user not found."""
    email = email.lower().strip()
    if not _load_user(email):
        return None  # Don't reveal whether account exists
    token = secrets.token_urlsafe(32)
    tokens = _load_reset_tokens()
    # Purge expired tokens
    now = datetime.utcnow()
    tokens = {t: v for t, v in tokens.items()
              if datetime.fromisoformat(v["expires_at"]) > now}
    tokens[token] = {
        "email": email,
        "expires_at": (now + timedelta(minutes=RESET_TOKEN_EXPIRY_MINUTES)).isoformat(),
    }
    _save_reset_tokens(tokens)
    return token


def consume_reset_token(token: str, new_password: str) -> bool:
    """Validate reset token and update password. Returns True on success."""
    tokens = _load_reset_tokens()
    entry = tokens.get(token)
    if not entry:
        return False
    if datetime.fromisoformat(entry["expires_at"]) <= datetime.utcnow():
        return False
    email = entry["email"]
    user = _load_user(email)
    if not user:
        return False
    if len(new_password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    user["password_hash"] = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    _save_user(user)
    # Invalidate the used token
    del tokens[token]
    _save_reset_tokens(tokens)
    return True


def send_reset_email(email: str, token: str, base_url: str = "https://lumafox.ai"):
    """Send password reset email. Falls back to logging if SMTP not configured."""
    reset_url = f"{base_url}/app?reset={token}"
    subject = "Reset your Lumafox password"
    body = f"""Hi,

Someone requested a password reset for your Lumafox account ({email}).

Click the link below to set a new password — it expires in {RESET_TOKEN_EXPIRY_MINUTES} minutes:

{reset_url}

If you didn't request this, ignore this email.

— Lumafox
"""
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    from_email = os.environ.get("FROM_EMAIL", smtp_user or "noreply@lumafox.ai")

    if not smtp_host or not smtp_user:
        # Local / no SMTP configured — print reset link so devs can test
        print(f"[RESET LINK for {email}] {reset_url}")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = f"Lumafox <{from_email}>"
    msg["To"] = email

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(from_email, [email], msg.as_string())


# ── Convenience ──

def login(email: str, password: str) -> Optional[dict]:
    """Authenticate and return token + user info, or None."""
    user = authenticate(email, password)
    if not user:
        return None

    token = create_token(user["user_id"], user["email"])
    return {
        "token": token,
        "user": user,
    }
