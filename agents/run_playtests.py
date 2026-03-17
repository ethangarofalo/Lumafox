#!/usr/bin/env python3
"""
Voice Playtest Harness — End-User Simulator Agents

Runs scripted persona sessions against the Voice API, generates
JSON logs and a markdown report. Each persona exercises a different
failure mode: onboarding confusion, power-user precision, brand
consistency, adversarial injection, and quality evaluation.

Usage:
  # Dev auth (fast, local)
  python3 agents/run_playtests.py --auth dev

  # JWT auth (staging/prod)
  python3 agents/run_playtests.py --auth jwt --email you@test.com --password yourpass

  # Against a remote server
  python3 agents/run_playtests.py --base-url https://voice.yourdomain.com --auth jwt ...
"""

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, os.path.dirname(__file__))
from personas import PERSONAS


# ── Data Structures ──

@dataclass
class StepResult:
    name: str
    ok: bool
    status: Optional[int]
    ms: int
    request: dict
    response: Optional[dict]
    error: Optional[str] = None
    notes: Optional[list] = None


@dataclass
class PersonaReport:
    persona_key: str
    persona_name: str
    description: str
    started_at: str
    finished_at: str
    profile_id: Optional[str]
    steps: list
    issues_found: list
    expected_issues: list
    score: Optional[dict] = None


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ── API Client ──

class VoiceClient:
    def __init__(self, base_url: str, auth_mode: str, user_id: str = None,
                 email: str = None, password: str = None):
        self.base_url = base_url.rstrip("/")
        self.auth_mode = auth_mode
        self.user_id = user_id or f"playtest-{uuid.uuid4().hex[:8]}"
        self.email = email
        self.password = password
        self.token = None
        self.session = requests.Session()

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.auth_mode == "jwt" and self.token:
            h["Authorization"] = f"Bearer {self.token}"
        elif self.auth_mode == "dev":
            h["X-User-Id"] = self.user_id
        return h

    def setup_auth(self):
        """Register or login for JWT mode."""
        if self.auth_mode != "jwt":
            return
        # Try login first, then register
        try:
            r = self.session.post(f"{self.base_url}/auth/login",
                                  json={"email": self.email, "password": self.password}, timeout=30)
            if r.status_code == 200:
                self.token = r.json()["token"]
                self.user_id = r.json()["user"]["user_id"]
                return
        except Exception:
            pass
        # Register
        r = self.session.post(f"{self.base_url}/auth/register",
                              json={"email": self.email, "password": self.password, "name": "Playtest Agent"},
                              timeout=30)
        r.raise_for_status()
        self.token = r.json()["token"]
        self.user_id = r.json()["user"]["user_id"]

    def request(self, method: str, path: str, json_body=None) -> StepResult:
        url = f"{self.base_url}{path}"
        t0 = time.time()
        try:
            r = self.session.request(method, url, headers=self._headers(),
                                     json=json_body, timeout=60)
            ms = int((time.time() - t0) * 1000)
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text[:500]}
            ok = 200 <= r.status_code < 300
            return StepResult(name="", ok=ok, status=r.status_code, ms=ms,
                              request={"method": method, "path": path, "json": json_body},
                              response=data, error=None if ok else data.get("detail", str(data)))
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            return StepResult(name="", ok=False, status=None, ms=ms,
                              request={"method": method, "path": path, "json": json_body},
                              response=None, error=str(e))


# ── Persona Runner ──

def run_persona(client: VoiceClient, key: str, persona: dict,
                run_mode: str = "UNKNOWN") -> PersonaReport:
    """Run a single persona's full session and collect results."""
    steps = []
    issues = []
    profile_id = None
    started = now_iso()

    def step(name, method, path, body=None):
        r = client.request(method, path, body)
        r.name = name
        steps.append(r)
        return r

    # 1. Health check
    h = step("health", "GET", "/health")
    if run_mode == "MOCK" and h.ok:
        # Surface mock mode as a harness-level issue
        issues.append({
            "severity": "warning",
            "summary": "LLM not configured — running in mock mode",
            "detail": "Set ANTHROPIC_API_KEY in the server environment for real LLM responses. "
                      "All content checks will pass vacuously in mock mode.",
        })

    # 2. Create profile
    r = step("create_profile", "POST", "/profiles", {
        "name": persona["profile_name"],
        "base_description": persona.get("profile_description", ""),
    })
    if r.ok and isinstance(r.response, dict):
        profile_id = r.response.get("profile_id")

    if not profile_id:
        issues.append({"severity": "blocker", "summary": "Failed to create profile", "detail": r.error})
        return PersonaReport(key, persona["name"], persona["description"],
                             started, now_iso(), None, [asdict(s) for s in steps],
                             issues, persona.get("expected_issues", []))

    # 3. Teaching sequence
    history = []
    for i, (cmd, msg) in enumerate(persona.get("teach_sequence", []), 1):
        r = step(f"teach_{i}_{cmd}", "POST", f"/profiles/{profile_id}/teach", {
            "command": cmd, "message": msg, "conversation_history": history,
        })
        if r.ok and isinstance(r.response, dict):
            history.append({"role": "user", "content": msg})
            if "response" in r.response:
                history.append({"role": "agent", "content": r.response["response"]})
        elif not r.ok:
            issues.append({
                "severity": "major" if r.status and r.status >= 500 else "minor",
                "summary": f"Teach step {i} ({cmd}) failed",
                "detail": r.error,
            })

    # 4. Write
    write_body = {"instruction": persona.get("write_instruction", "Write something.")}
    if persona.get("write_context"):
        write_body["context"] = persona["write_context"]

    r = step("write", "POST", f"/profiles/{profile_id}/write", write_body)
    generated_text = r.response.get("text", "") if r.ok and isinstance(r.response, dict) else ""

    # Content-level checks — always run against generated text
    # These catch voice fidelity failures regardless of what expected_issues says.
    FORBIDDEN_WORDS = persona.get("forbidden_words", [])
    if generated_text and FORBIDDEN_WORDS:
        for word in FORBIDDEN_WORDS:
            if word.lower() in generated_text.lower():
                issues.append({
                    "severity": "major",
                    "summary": f"Generated text contains forbidden word/phrase: '{word}'",
                    "detail": f"Found in: {generated_text[:300]}",
                })

    FORBIDDEN_PATTERNS = persona.get("forbidden_patterns", [])
    if generated_text and FORBIDDEN_PATTERNS:
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in generated_text:
                issues.append({
                    "severity": "major",
                    "summary": f"Generated text contains forbidden pattern: '{pattern}'",
                    "detail": f"Found in: {generated_text[:300]}",
                })

    # Universal style guard: transitional machinery is the most common voice failure
    ALWAYS_FORBIDDEN = ["moreover", "furthermore", "in conclusion", "it should be noted",
                        "in summary", "to summarize", "needless to say"]
    if generated_text and run_mode == "LIVE":
        for phrase in ALWAYS_FORBIDDEN:
            if phrase in generated_text.lower():
                issues.append({
                    "severity": "minor",
                    "summary": f"Generated text contains generic transitional phrase: '{phrase}'",
                    "detail": f"Found in: {generated_text[:300]}",
                })

    # 5. Analyze
    if persona.get("analyze_text"):
        r = step("analyze", "POST", f"/profiles/{profile_id}/analyze", {
            "text": persona["analyze_text"],
        })

    # 6. Export
    r = step("export", "GET", f"/profiles/{profile_id}/export")

    # 7. Get refinements
    r = step("refinements", "GET", f"/profiles/{profile_id}/refinements")
    if r.ok and isinstance(r.response, dict):
        count = r.response.get("count", 0)
        expected_count = len([t for t in persona.get("teach_sequence", [])
                              if t[0] in ("principle", "never", "voice", "example", "correct")])
        if count < expected_count:
            issues.append({
                "severity": "minor",
                "summary": f"Expected {expected_count} refinements, found {count}",
                "detail": f"Some teach commands may not have saved",
            })

    finished = now_iso()

    # Score
    total_steps = len(steps)
    passed = sum(1 for s in steps if s.ok)
    score = {
        "steps_total": total_steps,
        "steps_passed": passed,
        "steps_failed": total_steps - passed,
        "issues_found": len(issues),
        "pass_rate": round(passed / total_steps * 100, 1) if total_steps else 0,
    }

    return PersonaReport(
        key, persona["name"], persona["description"],
        started, finished, profile_id,
        [asdict(s) for s in steps], issues,
        persona.get("expected_issues", []), score,
    )


# ── Report Generation ──

def generate_report(results: list, out_dir: Path, run_mode: str = "UNKNOWN"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON log
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))

    # Markdown report
    lines = [f"# Voice Playtest Report", f"*{now_iso()}*\n"]

    if run_mode == "MOCK":
        lines.append("> **MOCK RUN** — LLM not configured. Results are not meaningful for voice fidelity. "
                     "Set `ANTHROPIC_API_KEY` and re-run for real results.\n")
    else:
        lines.append(f"> **LIVE RUN** — Real LLM responses\n")

    total_steps = sum(r.score["steps_total"] for r in results if r.score)
    total_passed = sum(r.score["steps_passed"] for r in results if r.score)
    total_issues = sum(len(r.issues_found) for r in results)

    lines.append(f"**Overall: {total_passed}/{total_steps} steps passed, {total_issues} issues found**\n")
    lines.append("---\n")

    for r in results:
        emoji = "PASS" if r.score and r.score["pass_rate"] == 100 and len(r.issues_found) == 0 else "ISSUES"
        lines.append(f"## [{emoji}] {r.persona_name}")
        lines.append(f"*{r.description}*\n")
        if r.score:
            lines.append(f"- Steps: {r.score['steps_passed']}/{r.score['steps_total']} passed ({r.score['pass_rate']}%)")
        lines.append(f"- Profile ID: `{r.profile_id}`")
        lines.append(f"- Issues: {len(r.issues_found)}\n")

        if r.issues_found:
            lines.append("### Issues Found\n")
            for issue in r.issues_found:
                sev = issue.get("severity", "minor").upper()
                lines.append(f"- **[{sev}]** {issue['summary']}")
                if issue.get("detail"):
                    lines.append(f"  - {issue['detail'][:200]}")
            lines.append("")

        # Show failures
        failures = [s for s in r.steps if not s.get("ok", True)]
        if failures:
            lines.append("### Failed Steps\n")
            for s in failures:
                lines.append(f"- **{s['name']}** — status={s.get('status')} error={s.get('error', '')[:100]}")
            lines.append("")

        lines.append("---\n")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines))

    return json_path, md_path


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Voice Playtest Harness")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--auth", choices=["dev", "jwt"], default="dev")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--email", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--personas", nargs="*", default=None,
                        help="Run specific personas (keys). Default: all.")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    if args.auth == "jwt" and (not args.email or not args.password):
        parser.error("JWT auth requires --email and --password")

    client = VoiceClient(args.base_url, args.auth,
                         user_id=args.user_id, email=args.email, password=args.password)

    if args.auth == "jwt":
        print("Authenticating...")
        client.setup_auth()
        print(f"  User: {client.user_id}")

    # For playtests, upgrade the user to pro plan so profile limits don't interfere
    from pathlib import Path as P
    data_dir = P(os.environ.get("VOICE_DATA_DIR", P(__file__).parent.parent / "data"))
    subs_dir = data_dir / "subscriptions"
    subs_dir.mkdir(parents=True, exist_ok=True)
    (subs_dir / f"{client.user_id}.json").write_text(json.dumps({
        "user_id": client.user_id, "plan": "pro",
        "stripe_customer_id": None, "stripe_subscription_id": None,
    }))

    # Detect LLM mode by hitting health before the test loop
    try:
        health_r = client.session.get(f"{client.base_url}/health",
                                      headers=client._headers(), timeout=10)
        health_data = health_r.json()
        llm_configured = health_data.get("llm_configured", False)
    except Exception:
        llm_configured = False
    run_mode = "LIVE" if llm_configured else "MOCK"

    if run_mode == "MOCK":
        print(f"\n⚠  WARNING: Server reports llm_configured=false.")
        print(f"   All LLM responses will be mock placeholders.")
        print(f"   Set ANTHROPIC_API_KEY in the server environment for real results.\n")
    else:
        print(f"\n✓  LLM configured — running in LIVE mode\n")

    # Select personas
    keys = args.personas or list(PERSONAS.keys())
    print(f"Running {len(keys)} persona(s): {', '.join(keys)}\n")

    # Run
    results = []
    for key in keys:
        if key not in PERSONAS:
            print(f"  Unknown persona: {key} — skipping")
            continue
        persona = PERSONAS[key]
        print(f"  [{key}] {persona['name']}...", end=" ", flush=True)
        report = run_persona(client, key, persona, run_mode=run_mode)
        results.append(report)
        if report.score:
            status = f"{report.score['steps_passed']}/{report.score['steps_total']} steps"
        else:
            status = "no score"
        issues = f"{len(report.issues_found)} issues"
        print(f"{status}, {issues}")

    # Generate report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else Path(__file__).parent.parent / "data" / "playtests" / timestamp

    json_path, md_path = generate_report(results, out_dir, run_mode=run_mode)

    print(f"\nReport:  {md_path}")
    print(f"Details: {json_path}")

    # Summary
    total_issues = sum(len(r.issues_found) for r in results)
    if total_issues > 0:
        print(f"\n⚠ {total_issues} issue(s) found — see report for details")
        return 1
    else:
        print(f"\n✓ All personas passed cleanly")
        return 0


if __name__ == "__main__":
    sys.exit(main())
