"""
Integration test for the Voice Authoring API.

Runs through the full lifecycle: create profile → teach → write → analyze → export → delete.
Uses the mock LLM caller (no API key needed).
"""

import os
import sys
import json

# Ensure we use mock LLM
if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]

# Set data dir to a temp location for testing
import tempfile
test_dir = tempfile.mkdtemp(prefix="voice_test_")
os.environ["VOICE_DATA_DIR"] = test_dir

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# ── Auth setup ──
def setup_auth():
    """Register a test user and get a JWT token."""
    r = client.post("/auth/register", json={
        "email": "test@voice.dev",
        "password": "testpass123",
        "name": "Test User",
    })
    assert r.status_code == 200, f"Registration failed: {r.text}"
    data = r.json()
    return data["token"], data["user"]["user_id"]

TOKEN, USER = setup_auth()
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def test_auth():
    """Test auth endpoints."""
    # Login with registered user
    r = client.post("/auth/login", json={"email": "test@voice.dev", "password": "testpass123"})
    assert r.status_code == 200
    assert "token" in r.json()

    # Bad password
    r = client.post("/auth/login", json={"email": "test@voice.dev", "password": "wrong"})
    assert r.status_code == 401

    # Duplicate registration
    r = client.post("/auth/register", json={"email": "test@voice.dev", "password": "testpass123"})
    assert r.status_code == 400

    # /auth/me
    r = client.get("/auth/me", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["user_id"] == USER

    # No token → 401
    r = client.get("/profiles")
    assert r.status_code == 401

    print("✓ Auth (register, login, me, rejection)")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print("✓ Health check")


def test_create_profile():
    r = client.post("/profiles", json={
        "name": "The Essayist",
        "base_description": "A voice that builds through extended metaphor, favors paratactic rhythm, and grounds abstraction in physical image."
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "The Essayist"
    assert data["owner_id"] == USER
    assert data["refinement_count"] == 0
    print(f"✓ Created profile: {data['profile_id']}")
    return data["profile_id"]


def test_list_profiles():
    r = client.get("/profiles", headers=HEADERS)
    assert r.status_code == 200
    profiles = r.json()
    assert len(profiles) >= 1
    print(f"✓ Listed {len(profiles)} profile(s)")


def test_get_profile(profile_id):
    r = client.get(f"/profiles/{profile_id}", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["profile_id"] == profile_id
    print(f"✓ Got profile {profile_id}")


def test_teach_principle(profile_id):
    r = client.post(f"/profiles/{profile_id}/teach", json={
        "message": "Extended metaphors should be sustained across entire passages, not used as quick illustrations.",
        "command": "principle",
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["refinement_saved"] is True
    assert data["refinement_type"] == "principle"
    assert data["refinement_count"] == 1
    print(f"✓ Taught principle (count: {data['refinement_count']})")


def test_teach_anti_pattern(profile_id):
    r = client.post(f"/profiles/{profile_id}/teach", json={
        "message": "Never use 'moreover,' 'furthermore,' or transitional machinery. Let the rhythm carry.",
        "command": "never",
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["refinement_saved"] is True
    assert data["refinement_type"] == "anti_pattern"
    assert data["refinement_count"] == 2
    print(f"✓ Taught anti-pattern (count: {data['refinement_count']})")


def test_teach_dialogue(profile_id):
    r = client.post(f"/profiles/{profile_id}/teach", json={
        "message": "Write me something about the experience of walking through a city at night.",
        "command": "dialogue",
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["refinement_saved"] is False
    assert len(data["response"]) > 0
    print(f"✓ Dialogue response ({len(data['response'])} chars)")


def test_teach_demo(profile_id):
    r = client.post(f"/profiles/{profile_id}/teach", json={
        "message": "The feeling of reading a great essay for the first time.",
        "command": "demo",
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert len(data["response"]) > 0
    print(f"✓ Demo response ({len(data['response'])} chars)")


def test_write(profile_id):
    r = client.post(f"/profiles/{profile_id}/write", json={
        "instruction": "Write an opening paragraph for an essay about why most writing today is boring.",
        "context": "This is for a blog post. The audience is writers and founders.",
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert len(data["text"]) > 0
    assert data["profile_name"] == "The Essayist"
    print(f"✓ Write mode ({len(data['text'])} chars)")


def test_analyze(profile_id):
    r = client.post(f"/profiles/{profile_id}/analyze", json={
        "text": "In today's rapidly evolving landscape, it's worth noting that content creation has been fundamentally transformed. Moreover, the proliferation of AI tools has created new opportunities for stakeholders across the value chain.",
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert len(data["analysis"]) > 0
    print(f"✓ Analyze mode ({len(data['analysis'])} chars)")


def test_refinements(profile_id):
    r = client.get(f"/profiles/{profile_id}/refinements", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["refinements"]) == 2
    print(f"✓ Got {data['count']} refinements")


def test_voice_text(profile_id):
    r = client.get(f"/profiles/{profile_id}/voice-text", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "REFINEMENTS FROM THE TEACHER" in data["voice_text"]
    print(f"✓ Voice text assembled ({len(data['voice_text'])} chars)")


def test_export(profile_id):
    r = client.get(f"/profiles/{profile_id}/export", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "Voice Profile: The Essayist" in data["markdown"]
    assert data["refinement_count"] == 2
    print(f"✓ Exported profile ({len(data['markdown'])} chars)")


def test_analyze_samples():
    r = client.post("/analyze-samples", json={
        "samples": [
            "The mountain does not care whether you climb it. It was there before your ambition and will be there after your bones have gone to dust.",
            "Every sentence is a wager. You bet your meaning against the reader's attention, and the odds are never in your favor.",
        ],
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert len(data["voice_description"]) > 0
    print(f"✓ Sample analysis ({len(data['voice_description'])} chars)")


def test_auth_barrier(profile_id):
    """Verify that another user can't access our profile."""
    # Register a second user
    r = client.post("/auth/register", json={
        "email": "intruder@voice.dev", "password": "badguy123", "name": "Intruder",
    })
    assert r.status_code == 200
    intruder_token = r.json()["token"]
    intruder_headers = {"Authorization": f"Bearer {intruder_token}"}

    r = client.get(f"/profiles/{profile_id}", headers=intruder_headers)
    assert r.status_code == 403
    print("✓ Auth barrier works")


def test_delete(profile_id):
    r = client.delete(f"/profiles/{profile_id}", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    # Verify gone
    r = client.get(f"/profiles/{profile_id}", headers=HEADERS)
    assert r.status_code == 404
    print(f"✓ Deleted profile {profile_id}")


def test_invalid_command(profile_id):
    """Creating a fresh profile for this test since we deleted the other."""
    r = client.post("/profiles", json={"name": "Temp"}, headers=HEADERS)
    pid = r.json()["profile_id"]

    r = client.post(f"/profiles/{pid}/teach", json={
        "message": "test",
        "command": "invalid_command",
    }, headers=HEADERS)
    assert r.status_code == 400
    print("✓ Invalid command rejected")

    # Clean up
    client.delete(f"/profiles/{pid}", headers=HEADERS)


def test_plan_gating():
    """Free plan limits to 1 profile."""
    # Create first profile — should succeed
    r = client.post("/profiles", json={"name": "Voice One"}, headers=HEADERS)
    assert r.status_code == 200
    pid1 = r.json()["profile_id"]

    # Create second profile — should be blocked (free = 1 profile)
    r = client.post("/profiles", json={"name": "Voice Two"}, headers=HEADERS)
    assert r.status_code == 403
    assert "Upgrade" in r.json()["detail"]

    # Check plan endpoint
    r = client.get("/billing/plan", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["plan"] == "free"
    assert r.json()["limits"]["profiles"] == 1

    # Clean up
    client.delete(f"/profiles/{pid1}", headers=HEADERS)
    print("✓ Plan gating (free plan limit enforced)")


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print("Voice Authoring API — Integration Tests")
    print(f"Data dir: {test_dir}")
    print(f"{'='*50}\n")

    profile_id = None
    try:
        test_auth()
        test_health()
        profile_id = test_create_profile()
        test_list_profiles()
        test_get_profile(profile_id)
        test_teach_principle(profile_id)
        test_teach_anti_pattern(profile_id)
        test_teach_dialogue(profile_id)
        test_teach_demo(profile_id)
        test_write(profile_id)
        test_analyze(profile_id)
        test_refinements(profile_id)
        test_voice_text(profile_id)
        test_export(profile_id)
        test_analyze_samples()
        test_auth_barrier(profile_id)
        test_delete(profile_id)
        test_invalid_command(profile_id)
        test_plan_gating()

        print(f"\n{'='*50}")
        print("ALL TESTS PASSED")
        print(f"{'='*50}\n")

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up test data
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
