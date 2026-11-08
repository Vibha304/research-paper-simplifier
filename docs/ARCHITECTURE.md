# Architecture: Step-by-Step

## 1. Layered design

```
┌──────────────────────────── frontend/ (Streamlit) ────────────────────────────┐
│  app_shell → views (analyze, library, chat, similarity, system) → components  │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │ calls only AppContainer
┌──────────────────────────── backend/ (application) ───────────────────────────┐
│  container.py (DI)   analysis_pipeline.py   library_service.py                │
│  models.py (Pydantic)   prompts.py (declarative AnalysisTask list)            │
└───────┬───────────────────────────┬──────────────────────────────┬────────────┘
        │                           │                              │
┌───────▼──── services/ ────────────▼──────────────────────────────▼────────────┐
│ pdf_service  chunking  llm_service  embedding  vector_store  rag  chat         │
│ similarity   scoring   export      project_kit                                │
└───────┬───────────────────────────┬──────────────────────────────┬────────────┘
        │                           │                              │
   PyMuPDF        BuiltinEngine (default) | Ollama (optional)   ChromaDB (data/chroma)
                                                         SQLite via database/
```

Dependencies only point downward. The frontend never runs SQL, calls Ollama or touches ChromaDB directly. Services never import Streamlit. `utils/` (config, errors, logging, text, JSON, progress) can be used by every layer.

## 2. Analysis engines

The pipeline talks to an **engine** through the `LLMClient` interface (`chat`, `stream_chat`, `embed`). `backend/container.py` selects it using `RPS_ENGINE`:

| | `builtin` (default) | `ollama` (optional) |
|---|---|---|
| Setup | none, pure Python | install Ollama + pull two models |
| How sections are produced | extractive NLP + domain templates (`generate_task`) | prompts + JSON output from Llama 3.2 |
| Embeddings | 512-dim hashed bag-of-words (stable blake2b hashing) | `nomic-embed-text` |
| Speed | a few seconds per paper | 2–5 minutes per paper on CPU |
| Hallucination risk | none for paper facts (quotes the paper, with pages) | low (grounded by RAG), but possible |

The built-in engine also implements the optional `StructuredGenerator` protocol (`generate_task`, `answer_question`). When the pipeline and chat service detect it, they pass it structured inputs (all chunks, retrieved chunks, earlier sections) instead of building prompts. Everything else (PDF handling, chunking, ChromaDB, SQLite, scoring, exports, the starter kit and the UI) is shared by both engines.

### Built-in engine internals (`services/builtin_engine.py`, `nlp_toolkit.py`, `domain_profiles.py`)
1. **Digest (once per paper):** chunks are split into clean prose sentences, tagged with their section and page. Front matter, "the paper is organised as follows" sentences, equations and table fragments are filtered out. The digest also records keywords (the paper's "Index Terms" line, or RAKE key phrases), glossary terms, datasets (a curated list plus a "*X dataset/benchmark*" pattern), tools, metrics and the domain.
2. **Sentence ranking:** each sentence gets a score from its cosine similarity to the section centroid plus weighted cue phrases. Near-duplicates are removed, and the chosen sentences are returned in reading order.
3. **Per task:** summary = top abstract sentences + domain analogy + key terms. Problem = problem cues, excluding contribution sentences. Contributions = "we propose / introduce / demonstrate" sentences. Methodology = ranked method-section sentences + jargon decoder. Limitations and future work = cue-ranked sentences outside the introduction, topped up with clearly labelled *Implied* or *Suggested* items. Project, roadmap and questions = domain templates filled with the extracted facts.
4. **Scores:** difficulty combines math density, advanced terminology, length, domain base difficulty and deep-learning usage. Resume impact combines trending topics and practicality. Time is a level baseline plus page count.
5. **Chat:** sentences from the retrieved excerpts are ranked against the question by embedding similarity and keyword overlap, then streamed as cited bullets. Low confidence produces an honest "couldn't find this" answer.

## 3. Request flow: analyzing a paper

1. **Upload** (`frontend/views/analyze.py`). Streamlit receives the bytes. `pipeline.find_existing()` hashes them (SHA-256). If the same paper was analyzed before, the user can open the saved result instantly or force a re-analysis.
2. **Validation** (`PDFService.validate`). Checks for an empty file, the `.pdf` extension, the size limit (50 MB) and the `%PDF` magic bytes. This happens before any parsing.
3. **Extraction** (`PDFService.extract`, 0–12% of the progress bar).
   * Opens the PDF from memory, rejects encrypted files and files over 100 pages.
   * Processes **one page at a time** (constant memory), wrapping each page in try/except so one broken page doesn't fail the paper.
   * **Column-aware ordering:** text blocks are split into left, right and spanning groups, then emitted header → left column → right column → footer. This fixes the interleaving problem in IEEE/ACM two-column layouts.
   * Cleans ligatures and hyphenation, drops page numbers, and **removes running headers/footers** (blocks repeated at page edges on ≥40% of pages).
   * Detects the venue (IEEE/ACM/Springer/arXiv/Elsevier) from the first pages before header stripping. Detects the title from metadata, falling back to the largest font on page 1.
   * Rejects scanned PDFs when the text density is below 80 chars/page.
4. **Chunking** (`SectionAwareChunker`, 12–15%).
   * Detects headings with regexes that tolerate numbering ("III.", "4.2", "A.") and the inline IEEE style "Abstract—…".
   * Chunks hold about 1,400 characters with 200 characters of overlap, **never cross a section boundary**, and carry `section` and `page_start/page_end` metadata.
   * The **References / Acknowledgments sections are excluded**, because citation lists pollute semantic search.
5. **Indexing** (`RAGService.index`, 15–35%). Chunks are embedded in batches of 16 through Ollama (`nomic-embed-text`, with its `search_document:` prefix) and stored in ChromaDB. Embeddings are always passed in explicitly, so **Chroma never downloads a model** and the app stays offline. The paper row is saved to SQLite at this point, so chat works even if generation is interrupted later.
6. **Generation** (`AnalysisPipeline._generate`, 35–95%). With the built-in engine, each task goes to `generate_task` (see §2). With Ollama, eight declarative `AnalysisTask`s run in order (overview → methodology → resources → critique → project → roadmap/assessment → viva → interview). For each task:
   1. **Multi-query retrieval:** 2–3 task-specific queries, each retrieving top-2k candidates.
   2. **Section-aware re-ranking:** +0.08 cosine bonus when a chunk comes from a section that usually answers the task (for example, `limitations`/`conclusion` for the critique task). Results are de-duplicated and the top k are kept.
   3. **Context packing:** chunks are ordered by score and tagged `[Excerpt n | section | p. x]` within a 9,000-character budget.
   4. **Prompt** = system rules (stay grounded, never invent, JSON only) + the **paper's opening** (title/abstract/intro, always included as anchor context) + retrieved excerpts + **known facts** from earlier tasks (keeps later sections consistent) + instructions + JSON schema.
   5. **Ollama `format: json`**, then `extract_json` (handles code fences, surrounding prose and trailing commas). Invalid JSON gets **one retry** at temperature 0.
   6. **Validation and merge:** Pydantic `mode="before"` validators coerce typical small-model quirks (bullet strings into lists, `q`/`a` keys, "8/10" or 0–100 scales, flat assessment keys into a nested object). If a key is still invalid, only that key is dropped and a warning is recorded.
   7. **Fault tolerance:** a failing task adds a warning and the pipeline continues. Only infrastructure errors (Ollama down, model missing) abort the run.
7. **Post-processing** (95–100%).
   * `merge_detected_resources` cross-checks datasets and tools against a curated vocabulary using exact word-boundary matches in the PDF text. Items the model missed are added as "(detected in text)", and "not mentioned" placeholders are removed.
   * `ScoringService.finalize` blends the LLM's judgement (60%) with explainable heuristics (40%):
     * **Difficulty** = math-symbol and equation density + advanced terminology + length, mapped to ≤3 Beginner, ≤6 Intermediate, otherwise Advanced.
     * **Resume impact** = trending-topic coverage + "sweet-spot" challenge + practicality (named datasets and tools).
     * **Time** = level baseline + implementation steps + modules, averaged with the LLM's estimate and shown as hours and weeks at 10 h/week.
   * `SimilarityService.index_paper` embeds a **profile** (title + summary + problem + contributions + keywords + method) into the `profiles_*` collection.
   * The analysis JSON and score columns are written to the `analyses` table.
8. **Rendering** (`analysis_view.py`). Metric cards, then eight tabs. PDF and starter-kit bytes are cached per analysis ID with `st.cache_data`.

Progress reporting uses `ProgressTracker`: each stage receives a callback that maps its local 0→1 progress onto its slice of the overall bar. The bar never moves backwards. The UI shows the percentage, elapsed time and a ✅/⏳ step log inside `st.status`.

## 4. Advanced features

| Feature | How it works |
|---|---|
| **Chat assistant** | `ChatService.ask`: retrieves for the question (for short follow-ups, the previous question is added to the query), builds system + last 4 turns + excerpts, and **streams** tokens from Ollama (`st.write_stream`). Both turns are saved with their source citations. |
| **Viva / interview questions** | Dedicated RAG tasks with different retrieval queries (concepts/results vs. implementation/trade-offs); the interview task also receives the known facts about the project. |
| **Roadmap** | Generated together with the assessment from the project context and rendered as a timeline; exported as a checklist in `ROADMAP.md`. |
| **Similarity search** | Cosine search over per-paper profile vectors, either from a stored paper or from free text. |
| **Build this Paper** | `ProjectKitService.build_zip` creates README, ROADMAP, PAPER_ANALYSIS, VIVA_PREP, requirements.txt (tech stack mapped to pip packages), `src/` modules (sanitized identifiers), `main.py` that runs all stages, and a smoke test. |
| **Scores** | See step 7 above. Each heuristic factor is shown in the rationale for transparency. |

## 5. Key design decisions

* **Interface over implementation:** `LLMClient` is a `Protocol`. Tests inject `FakeLLM`; supporting a different local backend (llama.cpp, LM Studio) means writing one class.
* **Composition root:** `backend/container.py` is the only place objects are wired together. Streamlit caches one container per process (`st.cache_resource`).
* **Declarative tasks:** adding a new analysis section means adding one `AnalysisTask` plus one model field. The pipeline doesn't change.
* **Short-lived SQLite connections** per unit of work, plus WAL mode. This is safe with Streamlit's multi-threaded reruns, and `ON DELETE CASCADE` keeps related data consistent.
* **Per-embedding-model collections** prevent dimension clashes when the embedding model changes.
* **Security:** all model output placed in custom HTML is escaped with `html.escape`, and files are never written to disk unless the user exports them.
* **Errors:** a typed hierarchy (`AppError` → PDF, LLM, Storage, Export). Each error has a technical message for logs and a `user_message` for the UI. Unexpected exceptions are logged with a traceback and shown as a generic message.

## 6. Testing strategy

* `test_pdf_and_chunking.py`: synthetic two-column IEEE-style PDFs (reading order, header stripping, venue and title detection), plus invalid, corrupt, too-long and scanned files, heading detection and chunk invariants.
* `test_utils_and_models.py`: JSON recovery, coercion, scoring and resource detection.
* `test_llm_client.py`: Ollama error mapping, retries and the legacy embeddings fallback (HTTP mocked).
* `test_builtin_engine.py`: the default engine on a realistic multi-section paper (grounded summary, contributions, datasets, limitations, future work, scores, questions, chat citations, similarity).
* `test_pipeline_end_to_end.py`: the full flow on real ChromaDB and SQLite (analysis, dedupe, forced re-analysis, chat, exports, starter kit, cascade delete, similarity, and survival when one task fails).
