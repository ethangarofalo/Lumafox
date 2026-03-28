"""
Lumafox Voice — MCP Server

Exposes your trained Lumafox voice profiles as tools available directly inside
Claude (Claude Desktop, Claude.ai, or any MCP-compatible client).

SETUP
-----
1. Install dependencies:
       pip install "mcp[cli]" httpx

2. Get your Lumafox token. Two options:

   Option A — email/password (auto-login):
       Set LUMAFOX_EMAIL and LUMAFOX_PASSWORD env vars.

   Option B — direct token (paste from browser dev tools / /auth/login response):
       Set LUMAFOX_TOKEN env var.

3. Set LUMAFOX_URL to your deployed instance:
       LUMAFOX_URL=https://your-app.onrender.com   (deployed)
       LUMAFOX_URL=http://localhost:8000             (local dev)

4. Add to Claude Desktop config (~/.config/claude/claude_desktop_config.json):

   {
     "mcpServers": {
       "lumafox-voice": {
         "command": "python",
         "args": ["/path/to/Lumafox/mcp_server.py"],
         "env": {
           "LUMAFOX_URL": "https://your-app.onrender.com",
           "LUMAFOX_EMAIL": "you@example.com",
           "LUMAFOX_PASSWORD": "your-password"
         }
       }
     }
   }

TOOLS EXPOSED
-------------
  list_voices              — list your voice profiles
  create_voice             — create a new profile
  write_in_voice           — generate content in your voice
  rephrase_in_voice        — rewrite a passage in your voice
  teach_voice              — train the voice through dialogue
  analyze_text             — score a text against the voice
  export_voice             — get the full synthesized voice document
  clear_voice_history      — reset the conversation cache for a profile
"""

import asyncio
import os
import httpx
from typing import Optional
from mcp.server.fastmcp import FastMCP

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_URL = os.environ.get("LUMAFOX_URL", "http://localhost:8000").rstrip("/")
_DIRECT_TOKEN = os.environ.get("LUMAFOX_TOKEN", "")
_EMAIL = os.environ.get("LUMAFOX_EMAIL", "")
_PASSWORD = os.environ.get("LUMAFOX_PASSWORD", "")

# ── Server ─────────────────────────────────────────────────────────────────────

mcp = FastMCP("Lumafox Voice")

# In-memory conversation history per profile (cleared on server restart).
# Keeps the last MAX_HISTORY exchanges so the voice engine has context
# for corrections, synthesis, and offer-acceptance detection.
_conversation_cache: dict[str, list[dict]] = {}
MAX_HISTORY_PAIRS = 6  # 6 user + 6 agent messages


# ── Auth ───────────────────────────────────────────────────────────────────────

_cached_token: str = ""


async def _get_token() -> str:
    """Return a valid JWT. Uses direct token if set, otherwise logs in."""
    global _cached_token

    if _DIRECT_TOKEN:
        return _DIRECT_TOKEN

    if _cached_token:
        return _cached_token

    if not _EMAIL or not _PASSWORD:
        raise RuntimeError(
            "Set LUMAFOX_TOKEN, or set both LUMAFOX_EMAIL and LUMAFOX_PASSWORD."
        )

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/auth/login",
            json={"email": _EMAIL, "password": _PASSWORD},
            timeout=15,
        )
        resp.raise_for_status()
        _cached_token = resp.json()["token"]
    return _cached_token


async def _headers() -> dict:
    token = await _get_token()
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# ── Conversation cache helpers ─────────────────────────────────────────────────

def _get_history(profile_id: str) -> list[dict]:
    return _conversation_cache.get(profile_id, [])


def _append_exchange(profile_id: str, user_msg: str, voice_response: str):
    history = _conversation_cache.setdefault(profile_id, [])
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "agent", "content": voice_response})
    # Keep only the most recent exchanges
    _conversation_cache[profile_id] = history[-(MAX_HISTORY_PAIRS * 2):]


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
async def list_voices() -> str:
    """List all available Lumafox voice profiles for this account."""
    headers = await _headers()
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/profiles", headers=headers, timeout=15)
        resp.raise_for_status()

    profiles = resp.json()
    if not profiles:
        return "No voice profiles found. Use create_voice to make one."

    lines = []
    for p in profiles:
        lines.append(
            f"- {p['name']}  |  id: {p['profile_id']}  |  {p['refinement_count']} refinements"
        )
    return "\n".join(lines)


@mcp.tool()
async def create_voice(name: str, description: str = "") -> str:
    """
    Create a new Lumafox voice profile.

    Args:
        name: Name for this voice (e.g. "Ethan — tweets", "Formal essays")
        description: Optional initial voice description. Leave blank to build
                     entirely through teaching.
    """
    headers = await _headers()
    payload = {"name": name, "base_description": description}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BASE_URL}/profiles", json=payload, headers=headers, timeout=15)
        resp.raise_for_status()

    data = resp.json()
    return (
        f"Voice '{data['name']}' created.\n"
        f"Profile ID: {data['profile_id']}\n"
        f"Start teaching it with teach_voice."
    )


@mcp.tool()
async def write_in_voice(profile_id: str, instruction: str, context: str = "") -> str:
    """
    Generate new content in a trained voice.

    Args:
        profile_id: Voice profile ID (from list_voices)
        instruction: What to write — be specific.
                     Examples: "a tweet about ambition", "an opening paragraph for an essay
                     on solitude", "a LinkedIn post announcing my new company"
        context: Optional notes, outline, or rough draft to work from
    """
    headers = await _headers()
    payload = {"instruction": instruction, "context": context}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/profiles/{profile_id}/write",
            json=payload,
            headers=headers,
            timeout=90,
        )
        resp.raise_for_status()

    data = resp.json()
    return data.get("text", "")


@mcp.tool()
async def rephrase_in_voice(profile_id: str, text: str) -> str:
    """
    Rewrite a passage in the user's trained voice. Preserves the meaning and
    approximate length; transforms the prose into the voice's style.

    Args:
        profile_id: Voice profile ID
        text: The passage to rewrite
    """
    message = f"Rewrite: {text}"
    history = _get_history(profile_id)
    headers = await _headers()
    payload = {"message": message, "command": "auto", "conversation_history": history}

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/profiles/{profile_id}/teach",
            json=payload,
            headers=headers,
            timeout=90,
        )
        resp.raise_for_status()

    data = resp.json()
    response_text = data.get("response", "")
    _append_exchange(profile_id, message, response_text)
    return response_text


@mcp.tool()
async def teach_voice(
    profile_id: str,
    message: str,
    command: str = "auto",
) -> str:
    """
    Send a message to train or interact with a voice profile.

    The voice engine auto-detects your intent. You can:

    TRAIN the voice by:
      - Sharing your writing: "Here's something I wrote: [passage]"
      - Correcting output: "Too long — cut it in half", "More direct, less formal"
      - Stating rules: "I never use passive voice"
      - Marking things to avoid: "Never end with a question"
      - Having philosophical dialogue — the voice learns your thinking patterns

    GET OUTPUT by:
      - Asking for writing: "Write something about sovereignty"
      - Following up: "Try it shorter", "Make it rawer"
      - Synthesizing: "Put those together into one paragraph"

    The voice remembers the conversation within this session.

    Args:
        profile_id: Voice profile ID
        message: What you want to say to the voice
        command: Routing hint — leave as "auto" unless you want to force a specific
                 mode. Options: auto, correct, example, principle, voice, never
    """
    history = _get_history(profile_id)
    headers = await _headers()
    payload = {
        "message": message,
        "command": command,
        "conversation_history": history,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/profiles/{profile_id}/teach",
            json=payload,
            headers=headers,
            timeout=90,
        )
        resp.raise_for_status()

    data = resp.json()
    response_text = data.get("response", "")
    _append_exchange(profile_id, message, response_text)

    output = response_text
    if data.get("refinement_saved"):
        rtype = data.get("refinement_type", "refinement")
        count = data.get("refinement_count", "")
        output += f"\n\n· {rtype} saved ({count} total)"
    return output


@mcp.tool()
async def analyze_text(profile_id: str, text: str) -> str:
    """
    Analyze how closely a piece of writing matches the voice profile.
    Returns a craft-level analysis: what aligns, what diverges, and why.

    Args:
        profile_id: Voice profile ID
        text: The text to analyze
    """
    headers = await _headers()
    payload = {"text": text}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/profiles/{profile_id}/analyze",
            json=payload,
            headers=headers,
            timeout=90,
        )
        resp.raise_for_status()

    return resp.json().get("analysis", "")


@mcp.tool()
async def export_voice(profile_id: str) -> str:
    """
    Export the full synthesized voice document — the living description of the
    voice's style, principles, and patterns built up through training.

    Useful for inspecting what the voice has learned, or for copying the voice
    description into other tools.

    Args:
        profile_id: Voice profile ID
    """
    headers = await _headers()
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{BASE_URL}/profiles/{profile_id}/export",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()

    data = resp.json()
    name = data.get("profile_name", "Voice")
    count = data.get("refinement_count", 0)
    markdown = data.get("markdown", "")
    return f"# {name}  ({count} refinements)\n\n{markdown}"


@mcp.tool()
async def clear_voice_history(profile_id: str) -> str:
    """
    Clear the in-session conversation history for a voice profile.
    Use this to start a fresh teaching session without the previous exchanges
    influencing how the voice interprets your next message.

    Args:
        profile_id: Voice profile ID
    """
    _conversation_cache.pop(profile_id, None)
    return f"Conversation history cleared for {profile_id}. Starting fresh."


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
