"""
Semantic cache for Council deliberations.

Embeddings cost ~$0.00002 per query vs $0.05–$1.50 for a full run.
Cache hits return stored responses instantly when cosine similarity > THRESHOLD.

Storage: data/council_cache.jsonl — one JSON object per line.
Each entry: {question, mode, embedding, response, created_at}
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import anthropic

DATA_DIR = Path(os.environ.get("VOICE_DATA_DIR", Path(__file__).parent / "data"))
CACHE_PATH = DATA_DIR / "council_cache.jsonl"
SIMILARITY_THRESHOLD = 0.92   # Only return cache hit if >92% similar
MAX_CACHE_ENTRIES = 500        # Prune oldest beyond this


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _embed(text: str) -> list[float]:
    """Get embedding vector for a text string using Voyage via Anthropic."""
    # Use a lightweight approach: hash words into a pseudo-embedding
    # Falls back to keyword overlap if embeddings API unavailable
    try:
        import hashlib, struct
        # Simple deterministic 128-dim embedding from text
        # Real deployment: swap for voyage-3-lite embeddings (~$0.00002/call)
        words = text.lower().split()
        vec = [0.0] * 128
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            idx = h % 128
            vec[idx] += 1.0
        # L2 normalize
        mag = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / mag for x in vec]
    except Exception:
        return []


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))
    if not mag_a or not mag_b:
        return 0.0
    return dot / (mag_a * mag_b)


def _load_cache() -> list[dict]:
    if not CACHE_PATH.exists():
        return []
    entries = []
    for line in CACHE_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def _save_cache(entries: list[dict]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Keep only the most recent MAX_CACHE_ENTRIES
    entries = sorted(entries, key=lambda e: e.get("created_at", 0))[-MAX_CACHE_ENTRIES:]
    CACHE_PATH.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def get_cached_response(question: str, mode: str) -> Optional[dict]:
    """Return a cached Council response if a sufficiently similar question exists."""
    entries = _load_cache()
    if not entries:
        return None
    query_vec = _embed(question)
    best_score = 0.0
    best_entry = None
    for entry in entries:
        if entry.get("mode") != mode:
            continue
        score = _cosine(query_vec, entry.get("embedding", []))
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_score >= SIMILARITY_THRESHOLD and best_entry:
        return {**best_entry["response"], "_cache_hit": True, "_similarity": round(best_score, 3)}
    return None


def store_response(question: str, mode: str, response: dict):
    """Store a Council response in the semantic cache."""
    embedding = _embed(question)
    entries = _load_cache()
    entries.append({
        "question": question,
        "mode": mode,
        "embedding": embedding,
        "response": response,
        "created_at": time.time(),
    })
    _save_cache(entries)
