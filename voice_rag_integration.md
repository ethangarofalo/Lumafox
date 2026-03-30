# voice_rag.py — Integration Guide

Three changes to voice_engine.py. Nothing structural moves.

---

## 1. Add to requirements.txt

```
voyageai>=0.3.0
numpy>=1.24.0
```

Add `VOYAGE_API_KEY=your_key` to `.env` and `.env.example`.

---

## 2. save_refinement() — index on write

Find the line where the refinement is appended to the JSONL file and add one call after it:

```python
from voice_rag import index_refinement

# existing JSONL write logic (unchanged)
with open(refinements_path, "a") as f:
    f.write(json.dumps(refinement_dict) + "\n")

# new: embed and index for retrieval
index_refinement(profile_dir, refinement_dict.get("content", ""))
```

The call is fire-and-forget. If Voyage AI is unavailable it returns False silently — saving still succeeds.

---

## 3. build_refinement_context() — retrieve instead of dump

Before the existing loop that formats all refinements, add a retrieval step:

```python
from voice_rag import retrieve_relevant_refinements

refinements = load_refinements(profile_id)

# Only retrieve when the profile is large enough that selectivity matters.
# Below this threshold, pass everything as before.
RAG_THRESHOLD = 10
if len(refinements) > RAG_THRESHOLD and current_prompt:
    refinements = retrieve_relevant_refinements(
        profile_dir,
        query=current_prompt,
        all_refinements=refinements,
        top_k=8,
    )

# existing formatting loop unchanged below
```

`current_prompt` is whatever the user is asking the voice to generate. If that variable isn't already in scope at this call site, pass it through from `write_with_voice()`.

---

## 4. ingest_writing_samples() — index uploaded files

After the LLM analysis extracts principles from uploaded samples, index the raw text:

```python
from voice_rag import index_writing_samples

# existing LLM analysis (unchanged)
analysis = llm_call(analyze_prompt)
# ... parse and save principles ...

# new: embed raw samples for concrete example retrieval
index_writing_samples(profile_dir, raw_texts)  # raw_texts = list of uploaded file contents
```

---

## 5. write_with_voice() — inject concrete examples (optional, high value)

This is the highest-value integration. After building the voice prompt, prepend retrieved examples:

```python
from voice_rag import retrieve_relevant_samples

examples = retrieve_relevant_samples(profile_dir, prompt, top_k=3)
if examples:
    example_block = "\n\n".join(f"EXAMPLE FROM YOUR OWN WRITING:\n{e}" for e in examples)
    full_prompt = f"{voice_prompt}\n\n{example_block}\n\nNOW WRITE:\n{prompt}"
else:
    full_prompt = f"{voice_prompt}\n\nNOW WRITE:\n{prompt}"
```

This is what closes the loop — instead of the model working purely from extracted principles, it has concrete passages from the user's actual writing to match against. The difference in output quality is significant.

---

## Rebuild existing profiles

For profiles created before RAG was added, the index is empty. Rebuild on first request:

```python
# voice_rag.py handles this automatically inside retrieve_relevant_refinements()
# via _rebuild_refinement_index() — no manual migration needed.
```
