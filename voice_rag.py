"""
voice_rag.py — Voyage AI embedding layer for Lumafox Voice.

Provides semantic retrieval of refinements and writing samples,
replacing the flat-dump approach in build_refinement_context()
with context-aware selection. When the current prompt is a tweet,
the model retrieves tweet corrections. When it's a philosophical essay,
it retrieves those samples — rather than treating all stored voice
data as equally relevant to every generation request.

Architecture:
    Embeddings stored as .npz alongside existing JSONL profile files.
    Cosine similarity computed with numpy — no external vector DB required.
    Graceful fallback to full-context retrieval when Voyage AI is unavailable.

Integration points in voice_engine.py:
    1. build_refinement_context()  — replace full dump with retrieve_relevant_refinements()
    2. ingest_writing_samples()    — call index_writing_samples() after LLM analysis
    3. write_with_voice()          — retrieve relevant samples alongside refinements

Setup:
    pip install voyageai numpy
    VOYAGE_API_KEY=your_key  (add to .env)
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Voyage AI client — lazy init so the module loads without the package
# ---------------------------------------------------------------------------

_voyage_client = None

def _get_voyage_client():
    """Return a cached Voyage AI client, or None if unavailable."""
    global _voyage_client
    if _voyage_client is not None:
        return _voyage_client
    try:
        import voyageai
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            return None
        _voyage_client = voyageai.Client(api_key=api_key)
        return _voyage_client
    except ImportError:
        return None


def rag_available() -> bool:
    """Return True if Voyage AI is configured and reachable."""
    return _get_voyage_client() is not None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "voyage-3"          # Best general-purpose model
EMBEDDING_DIM   = 1024                 # voyage-3 output dimension


def embed(texts: list[str]) -> Optional[np.ndarray]:
    """
    Embed a list of texts with Voyage AI.

    Returns an (n, EMBEDDING_DIM) float32 array, or None on failure.
    Voyage AI recommends batching; this handles lists of any size.
    """
    client = _get_voyage_client()
    if client is None:
        return None
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    try:
        result = client.embed(texts, model=EMBEDDING_MODEL, input_type="document")
        return np.array(result.embeddings, dtype=np.float32)
    except Exception as e:
        print(f"[voice_rag] Voyage AI embed failed: {e}")
        return None


def embed_query(text: str) -> Optional[np.ndarray]:
    """
    Embed a single query string.

    Uses input_type='query' as Voyage AI recommends asymmetric
    query/document embeddings for retrieval tasks.
    """
    client = _get_voyage_client()
    if client is None:
        return None
    try:
        result = client.embed([text], model=EMBEDDING_MODEL, input_type="query")
        return np.array(result.embeddings[0], dtype=np.float32)
    except Exception as e:
        print(f"[voice_rag] Voyage AI query embed failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and a matrix of document vectors.

    Returns a 1-D array of similarity scores in [-1, 1].
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms  = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm


# ---------------------------------------------------------------------------
# Refinement index  (one per voice profile)
# ---------------------------------------------------------------------------
#
# Storage layout alongside existing JSONL files:
#
#   {profile_dir}/
#       refinements.jsonl          ← existing (unchanged)
#       rag_refinements.npz        ← embeddings + parallel text array
#
# The .npz contains two arrays:
#   "embeddings"  shape (n, EMBEDDING_DIM)  float32
#   "texts"       shape (n,)                object (str)
#
# Array index matches refinement order in refinements.jsonl, so existing
# JSONL logic is untouched — RAG is purely additive.

def _refinement_index_path(profile_dir: str) -> Path:
    return Path(profile_dir) / "rag_refinements.npz"


def _load_refinement_index(profile_dir: str) -> tuple[np.ndarray, list[str]]:
    """Load stored embeddings and texts. Returns (embeddings, texts)."""
    path = _refinement_index_path(profile_dir)
    if not path.exists():
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []
    data  = np.load(path, allow_pickle=True)
    embs  = data["embeddings"].astype(np.float32)
    texts = data["texts"].tolist()
    return embs, texts


def _save_refinement_index(
    profile_dir: str,
    embeddings: np.ndarray,
    texts: list[str],
) -> None:
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        _refinement_index_path(profile_dir),
        embeddings=embeddings,
        texts=np.array(texts, dtype=object),
    )


def index_refinement(profile_dir: str, refinement_text: str) -> bool:
    """
    Embed and append a single refinement to the profile's RAG index.

    Call this inside save_refinement() immediately after writing to JSONL:

        # voice_engine.py — inside save_refinement()
        from voice_rag import index_refinement
        index_refinement(profile_dir, text_to_embed)

    Returns True on success, False if Voyage AI is unavailable.
    """
    new_vec = embed([refinement_text])
    if new_vec is None:
        return False

    existing_embs, existing_texts = _load_refinement_index(profile_dir)

    updated_embs  = np.vstack([existing_embs, new_vec]) if existing_embs.size else new_vec
    updated_texts = existing_texts + [refinement_text]

    _save_refinement_index(profile_dir, updated_embs, updated_texts)
    return True


def retrieve_relevant_refinements(
    profile_dir: str,
    query: str,
    all_refinements: list[dict],
    top_k: int = 6,
) -> list[dict]:
    """
    Return the top_k most semantically relevant refinements for the current query.

    Drop-in replacement for the full refinement list in build_refinement_context().
    Falls back to returning all_refinements unchanged if RAG is unavailable,
    so voice generation is never blocked by a missing API key.

    Args:
        profile_dir:      Path to the profile's data directory.
        query:            The current generation prompt or task description.
        all_refinements:  The full list of refinement dicts from load_refinements().
        top_k:            Number of refinements to return.

    Returns:
        A filtered list of refinement dicts, ordered by relevance.

    Example (voice_engine.py — inside build_refinement_context()):

        from voice_rag import retrieve_relevant_refinements

        refinements = load_refinements(profile_id)
        if len(refinements) > top_k:
            refinements = retrieve_relevant_refinements(
                profile_dir, current_prompt, refinements, top_k=8
            )
        # then proceed with existing formatting logic unchanged
    """
    if not all_refinements:
        return all_refinements

    if len(all_refinements) <= top_k:
        return all_refinements

    query_vec = embed_query(query)
    if query_vec is None:
        # Voyage AI unavailable — return most recent refinements as fallback
        return all_refinements[-top_k:]

    embs, texts = _load_refinement_index(profile_dir)

    # Index may be empty or mismatched if RAG was added after some refinements
    # were saved. Rebuild transparently if needed.
    if len(texts) < len(all_refinements):
        _rebuild_refinement_index(profile_dir, all_refinements)
        embs, texts = _load_refinement_index(profile_dir)

    if embs.shape[0] == 0:
        return all_refinements[-top_k:]

    n = min(len(all_refinements), embs.shape[0])
    scores  = _cosine_similarity(query_vec, embs[:n])
    indices = np.argsort(scores)[::-1][:top_k]

    return [all_refinements[i] for i in sorted(indices)]


def _rebuild_refinement_index(profile_dir: str, refinements: list[dict]) -> None:
    """
    Rebuild the full refinement index from a list of refinement dicts.

    Called automatically when the index is stale (e.g. RAG added mid-profile).
    Expects each refinement dict to have a 'content' or 'text' key.
    """
    texts = []
    for r in refinements:
        text = r.get("content") or r.get("text") or json.dumps(r)
        texts.append(str(text))

    vecs = embed(texts)
    if vecs is None:
        return
    _save_refinement_index(profile_dir, vecs, texts)


# ---------------------------------------------------------------------------
# Writing sample index  (uploaded files / ingest_writing_samples)
# ---------------------------------------------------------------------------
#
# Storage layout:
#
#   {profile_dir}/
#       rag_samples.npz            ← embeddings + parallel text array
#
# Samples are chunked at roughly paragraph length before embedding so
# that long documents don't collapse into a single averaged vector.

CHUNK_SIZE = 400  # characters per chunk (roughly 80-100 words)


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into overlapping chunks at paragraph or sentence boundaries."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) <= size:
            current = (current + " " + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks or [text[:size]]


def _sample_index_path(profile_dir: str) -> Path:
    return Path(profile_dir) / "rag_samples.npz"


def _load_sample_index(profile_dir: str) -> tuple[np.ndarray, list[str]]:
    path = _sample_index_path(profile_dir)
    if not path.exists():
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []
    data  = np.load(path, allow_pickle=True)
    embs  = data["embeddings"].astype(np.float32)
    texts = data["texts"].tolist()
    return embs, texts


def index_writing_samples(profile_dir: str, sample_texts: list[str]) -> bool:
    """
    Embed and index a list of writing samples for semantic retrieval.

    Call this inside ingest_writing_samples() after the LLM analysis step:

        # voice_engine.py — inside ingest_writing_samples()
        from voice_rag import index_writing_samples
        index_writing_samples(profile_dir, raw_sample_texts)

    Each sample is chunked before embedding so long documents remain
    retrievable at the passage level, not just as whole-document averages.

    Returns True on success, False if Voyage AI is unavailable.
    """
    all_chunks = []
    for text in sample_texts:
        all_chunks.extend(_chunk_text(text))

    if not all_chunks:
        return False

    vecs = embed(all_chunks)
    if vecs is None:
        return False

    existing_embs, existing_texts = _load_sample_index(profile_dir)

    updated_embs  = np.vstack([existing_embs, vecs]) if existing_embs.size else vecs
    updated_texts = existing_texts + all_chunks

    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        _sample_index_path(profile_dir),
        embeddings=updated_embs,
        texts=np.array(updated_texts, dtype=object),
    )
    return True


def retrieve_relevant_samples(
    profile_dir: str,
    query: str,
    top_k: int = 3,
) -> list[str]:
    """
    Return the top_k most relevant writing sample passages for the current query.

    Use this in write_with_voice() to inject concrete style references:

        # voice_engine.py — inside write_with_voice()
        from voice_rag import retrieve_relevant_samples

        examples = retrieve_relevant_samples(profile_dir, prompt, top_k=3)
        if examples:
            example_block = "\\n\\n".join(f'EXAMPLE:\\n{e}' for e in examples)
            # inject example_block into the generation prompt

    Returns an empty list if RAG is unavailable or no samples are indexed.
    """
    query_vec = embed_query(query)
    if query_vec is None:
        return []

    embs, texts = _load_sample_index(profile_dir)
    if embs.shape[0] == 0:
        return []

    scores  = _cosine_similarity(query_vec, embs)
    indices = np.argsort(scores)[::-1][:top_k]
    return [texts[i] for i in indices]


# ---------------------------------------------------------------------------
# Index health / diagnostics
# ---------------------------------------------------------------------------

def index_stats(profile_dir: str) -> dict:
    """
    Return a summary of what's currently indexed for a profile.

    Useful for debugging and for the /voice/profile/{id} admin endpoint.
    """
    ref_embs, ref_texts   = _load_refinement_index(profile_dir)
    samp_embs, samp_texts = _load_sample_index(profile_dir)
    return {
        "rag_available":        rag_available(),
        "refinements_indexed":  len(ref_texts),
        "sample_chunks_indexed": len(samp_texts),
        "embedding_model":      EMBEDDING_MODEL,
        "embedding_dim":        EMBEDDING_DIM,
    }
