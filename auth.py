"""
Authentication — email/password with JWT tokens.

Simple but real: bcrypt for passwords, JWT for sessions.
Replace with OAuth/SSO when ready for production.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import bcrypt
import jwt

# ── Config ──

SECRET_KEY = os.environ.get("JWT_SECRET", "voice-dev-secret-change-in-prod!")  # 32 bytes
TOKEN_EXPIRY_HOURS = int(os.environ.get("TOKEN_EXPIRY_HOURS", "72"))
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
