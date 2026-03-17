"""
Knowledge graph integration via cognee.

Provides three capabilities:
  1. SOURCE GROUNDING — Ingest actual philosophical texts so traditions
     reason from the source material, not summaries
  2. CROSS-TRADITION MAPPING — Track relationships between concepts
     across traditions (where they agree, conflict, or subtly differ)
  3. RETRIEVAL — Before an agent reasons, retrieve the most relevant
     passages and relationships from its tradition's corpus

Cognee is optional. Polis works without it using the markdown tradition
files and JSONL refinements. With cognee installed and sources loaded,
agents get grounded in the actual texts.

Usage:
    from knowledge import KnowledgeGraph

    kg = KnowledgeGraph()
    await kg.initialize()

    # Ingest a tradition's source texts
    await kg.ingest_tradition("aristotelian", [
        "path/to/nicomachean_ethics.txt",
        "path/to/politics.txt",
    ])

    # Retrieve relevant context for a question
    context = await kg.retrieve("aristotelian", "Is contemplation threatened by AI?")

    # Map a concept across traditions
    mappings = await kg.cross_reference("justice")
"""

import asyncio
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

TRADITIONS_DIR = Path(__file__).parent / "traditions"
SOURCES_DIR = TRADITIONS_DIR / "sources"


# ──────────────────────────────────────────────────────────────────
# Cognee availability check
# ──────────────────────────────────────────────────────────────────

try:
    import cognee
    COGNEE_AVAILABLE = True
except ImportError:
    COGNEE_AVAILABLE = False


@dataclass
class RetrievedContext:
    """Context retrieved from the knowledge graph for agent grounding."""
    tradition: str
    passages: list[str]
    related_concepts: list[dict]       # [{concept, relationship, connected_to}]
    cross_tradition_links: list[dict]  # [{concept, tradition_a, tradition_b, relationship}]

    def to_prompt(self) -> str:
        """Format as text for injection into agent prompts."""
        lines = []
        if self.passages:
            lines.append("RELEVANT PASSAGES FROM SOURCE TEXTS:")
            for i, p in enumerate(self.passages[:5], 1):
                # Truncate long passages
                truncated = p[:500] + "..." if len(p) > 500 else p
                lines.append(f"  [{i}] {truncated}")
            lines.append("")

        if self.related_concepts:
            lines.append("RELATED CONCEPTS IN YOUR TRADITION:")
            for c in self.related_concepts[:8]:
                lines.append(f"  - {c['concept']} ({c['relationship']}) → {c['connected_to']}")
            lines.append("")

        if self.cross_tradition_links:
            lines.append("CROSS-TRADITION CONNECTIONS:")
            for link in self.cross_tradition_links[:5]:
                lines.append(
                    f"  - '{link['concept']}' in {link['tradition_a']} "
                    f"↔ {link['tradition_b']}: {link['relationship']}"
                )
            lines.append("")

        return "\n".join(lines) if lines else ""


class KnowledgeGraph:
    """
    Knowledge graph interface for Polis.

    When cognee is available, uses it for full graph-RAG capabilities.
    When cognee is not available, falls back to simple keyword-based
    retrieval from the source text files.
    """

    def __init__(self):
        self.initialized = False
        self.use_cognee = COGNEE_AVAILABLE
        self._source_cache: dict[str, list[str]] = {}  # tradition -> [text chunks]

    async def initialize(self):
        """Initialize the knowledge graph backend."""
        if self.use_cognee:
            try:
                # cognee auto-configures on first use
                self.initialized = True
                print("  Knowledge graph: cognee initialized")
            except Exception as e:
                print(f"  Knowledge graph: cognee failed ({e}), falling back to local")
                self.use_cognee = False
                self.initialized = True
        else:
            self.initialized = True
            print("  Knowledge graph: using local fallback (install cognee for full graph-RAG)")

    async def ingest_tradition(self, tradition_name: str,
                                file_paths: Optional[list[str]] = None):
        """
        Ingest source texts for a tradition into the knowledge graph.

        If no file_paths given, looks in traditions/sources/{tradition_name}/
        """
        if file_paths is None:
            source_dir = SOURCES_DIR / tradition_name
            if not source_dir.exists():
                print(f"  No sources found for {tradition_name} at {source_dir}")
                return
            file_paths = [str(p) for p in source_dir.glob("*.txt")]
            file_paths.extend(str(p) for p in source_dir.glob("*.md"))

        if not file_paths:
            print(f"  No source files found for {tradition_name}")
            return

        print(f"  Ingesting {len(file_paths)} source(s) for {tradition_name}...")

        for fpath in file_paths:
            path = Path(fpath)
            if not path.exists():
                print(f"    Skipping {fpath} (not found)")
                continue

            text = path.read_text(encoding="utf-8", errors="replace")
            chunks = self._chunk_text(text, chunk_size=1000, overlap=200)

            if self.use_cognee:
                # Tag each chunk with the tradition for scoped retrieval
                tagged_text = f"[TRADITION: {tradition_name}] [SOURCE: {path.stem}]\n\n{text}"
                await cognee.add(tagged_text)

            # Always cache locally as fallback
            if tradition_name not in self._source_cache:
                self._source_cache[tradition_name] = []
            self._source_cache[tradition_name].extend(chunks)

            print(f"    Ingested: {path.name} ({len(chunks)} chunks)")

        if self.use_cognee:
            print(f"  Cognifying {tradition_name} sources...")
            await cognee.cognify()
            print(f"  Done — knowledge graph updated for {tradition_name}")

    async def ingest_all_traditions(self):
        """Ingest sources for all traditions that have a sources/ directory."""
        if not SOURCES_DIR.exists():
            print(f"  No sources directory found at {SOURCES_DIR}")
            return

        for tradition_dir in SOURCES_DIR.iterdir():
            if tradition_dir.is_dir():
                await self.ingest_tradition(tradition_dir.name)

    async def retrieve(self, tradition_name: str, query: str,
                       max_passages: int = 5) -> RetrievedContext:
        """
        Retrieve relevant context for a tradition given a query.

        Uses cognee's graph-RAG when available, falls back to simple
        keyword matching against cached source chunks.
        """
        passages = []
        related_concepts = []
        cross_links = []

        if self.use_cognee:
            try:
                # Search cognee's knowledge graph
                results = await cognee.search(
                    query_text=f"[{tradition_name}] {query}"
                )
                for result in results[:max_passages]:
                    if hasattr(result, "text"):
                        passages.append(result.text)
                    elif isinstance(result, dict) and "text" in result:
                        passages.append(result["text"])
                    elif isinstance(result, str):
                        passages.append(result)
                    else:
                        passages.append(str(result))
            except Exception as e:
                print(f"  Cognee search failed: {e}, using local fallback")
                passages = self._local_search(tradition_name, query, max_passages)
        else:
            passages = self._local_search(tradition_name, query, max_passages)

        return RetrievedContext(
            tradition=tradition_name,
            passages=passages,
            related_concepts=related_concepts,
            cross_tradition_links=cross_links,
        )

    async def cross_reference(self, concept: str) -> list[dict]:
        """
        Find how a concept appears across different traditions.

        Returns mappings showing how different traditions understand
        the same concept differently.
        """
        mappings = []

        if self.use_cognee:
            try:
                results = await cognee.search(query_text=concept)
                # Parse results to find cross-tradition connections
                for result in results:
                    text = str(result)
                    for tradition in self._get_tradition_names():
                        if tradition.lower() in text.lower():
                            mappings.append({
                                "concept": concept,
                                "tradition": tradition,
                                "context": text[:300],
                            })
            except Exception:
                pass

        # Local fallback: search each tradition's cache
        if not mappings:
            for tradition, chunks in self._source_cache.items():
                query_lower = concept.lower()
                relevant = [c for c in chunks if query_lower in c.lower()]
                if relevant:
                    mappings.append({
                        "concept": concept,
                        "tradition": tradition,
                        "context": relevant[0][:300],
                    })

        return mappings

    def _local_search(self, tradition_name: str, query: str,
                      max_results: int = 5) -> list[str]:
        """Simple keyword-based search against cached source chunks."""
        chunks = self._source_cache.get(tradition_name, [])
        if not chunks:
            return []

        # Score chunks by keyword overlap
        query_words = set(query.lower().split())
        scored = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scored.append((overlap, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:max_results]]

    def _chunk_text(self, text: str, chunk_size: int = 1000,
                    overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        # Split on paragraph boundaries first
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from end of current chunk
                words = current_chunk.split()
                overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else words
                current_chunk = " ".join(overlap_words) + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:chunk_size]]

    def _get_tradition_names(self) -> list[str]:
        """Get all tradition names from the traditions directory."""
        names = []
        for f in TRADITIONS_DIR.glob("*.md"):
            if not f.stem.endswith("_refined"):
                names.append(f.stem)
        return names


# ──────────────────────────────────────────────────────────────────
# CLI for managing the knowledge graph
# ──────────────────────────────────────────────────────────────────

async def _cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="Manage POLIS knowledge graph")
    sub = parser.add_subparsers(dest="command")

    ingest = sub.add_parser("ingest", help="Ingest source texts for a tradition")
    ingest.add_argument("tradition", help="Tradition name")
    ingest.add_argument("files", nargs="*", help="Paths to source files (optional)")

    ingest_all = sub.add_parser("ingest-all", help="Ingest all traditions with sources")

    search = sub.add_parser("search", help="Search the knowledge graph")
    search.add_argument("tradition", help="Tradition to search")
    search.add_argument("query", help="Search query")

    xref = sub.add_parser("xref", help="Cross-reference a concept across traditions")
    xref.add_argument("concept", help="Concept to look up")

    args = parser.parse_args()

    kg = KnowledgeGraph()
    await kg.initialize()

    if args.command == "ingest":
        files = args.files if args.files else None
        await kg.ingest_tradition(args.tradition, files)

    elif args.command == "ingest-all":
        await kg.ingest_all_traditions()

    elif args.command == "search":
        ctx = await kg.retrieve(args.tradition, args.query)
        print(f"\nResults for '{args.query}' in {args.tradition}:\n")
        print(ctx.to_prompt() or "  No results found.")

    elif args.command == "xref":
        mappings = await kg.cross_reference(args.concept)
        print(f"\nCross-references for '{args.concept}':\n")
        for m in mappings:
            print(f"  [{m['tradition']}] {m['context'][:200]}")
            print()


if __name__ == "__main__":
    asyncio.run(_cli_main())
