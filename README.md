# NeuroRouter

**A Jev-powered probabilistic control plane for multi-agent AI systems.**

NeuroRouter separates probabilistic routing judgments from deterministic policy. Jev will
classify atomic properties of a request, Python policy will turn those probabilities into an
execution plan, bounded specialist agents will gather evidence, and an LLM will synthesize only
when needed. Every stage is designed to be inspectable through persisted traces.

## Implementation status: Phases 1–5

This repository currently provides the foundation:

- validated Pydantic contracts for state, routing, execution, and traces;
- typed YAML and environment configuration without secrets in runtime state;
- a thread-safe SQLite trace repository and request tracer;
- a multi-page Streamlit control-center shell;
- a secret-free `StateBuilder` for validated routing context;
- one batched Jev call containing 10 independent Choice, Noul, and Score judgments;
- a concrete async TypeSafe SDK adapter with strict response normalization;
- complete probability preservation and a configurable, capability-aware failure fallback;
- a deterministic policy engine for capability gating, agent dependencies, model tiers, citations,
  quality control, and review escalation;
- bounded General and Web Research agents with normalized results and isolated failures;
- a free, no-key Wikipedia search provider behind a replaceable `WebSearchProvider` interface;
- dependency-aware orchestration with explicit parallel groups and per-agent telemetry;
- PDF, TXT, and Markdown extraction with page-aware document metadata;
- deterministic overlapping chunks and stable content identifiers;
- a configurable embedding boundary with key-free local hashing embeddings by default;
- persistent cosine retrieval through ChromaDB and structured local evidence from `RAGAgent`;
- unit tests for deterministic foundation code.

Code and Finance agents, synthesis, and quality-gate evaluation are intentionally not implemented
yet. The Jev adapter is ready for a `TYPESAFE_API_KEY`, while unit tests use injected responses and
require no external service.

The Phase 4 web provider searches English Wikipedia rather than the entire public web. This keeps
local demos keyless and honest about source coverage. A broader provider can be substituted through
the same interface without changing `WebResearchAgent` or orchestration code.

## Local setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,rag]"
copy .env.example .env  # Windows; use `cp` on macOS/Linux
streamlit run app.py
```

No API key is required for the current local pipeline. The trace database is created at
`data/neurorouter.db`; the vector collection is persisted under `data/chroma/`.

Run checks with:

```bash
pytest
ruff check .
```

## Configuration

- `config/settings.yaml`: runtime capabilities, telemetry, UI, and future model tiers.
- `config/thresholds.yaml`: deterministic routing and quality policy thresholds.
- `config/prompts.yaml`: versioned prompt placeholders for later phases.
- `.env`: credentials and local overrides; this file is ignored by Git.

## Local RAG usage

The Phase 5 API intentionally stays independent of Streamlit so ingestion and retrieval can be
tested or reused by later interfaces:

```python
from pathlib import Path

from neurorouter.rag import ChromaVectorStore, Chunker, HashingEmbeddingProvider, Retriever
from neurorouter.tools.document_loader import DocumentLoader

retriever = Retriever(
    chunker=Chunker(chunk_size=1000, overlap=150),
    embeddings=HashingEmbeddingProvider(dimensions=384),
    vector_store=ChromaVectorStore(Path("data/chroma"), "neurorouter_documents"),
)
retriever.index(DocumentLoader().load("data/example.pdf"))
matches = retriever.retrieve("What does the document say about routing?")
```

The default hashing provider is deterministic, local, and free; it is best suited to portfolio
demos and lexical-semantic retrieval. Its interface can later accept a stronger free embedding
provider without coupling ChromaDB, the retriever, or `RAGAgent` to that provider.

Environment overrides use the `NEUROROUTER_` prefix and `__` for nested keys, for example
`NEUROROUTER_TELEMETRY__DATABASE_PATH=data/demo.db`.

### Free-tier LLM providers

The application defaults to `mock`, which requires no key and cannot incur API charges. The
configuration also defines interchangeable `groq`, `gemini`, and `openrouter` providers for the
later synthesis phase. Set the matching key in `.env` and set
`NEUROROUTER_LLM__PROVIDER` to its name.

`allow_paid_models` defaults to `false`. OpenRouter uses its `openrouter/free` router, Gemini uses
Flash/Flash-Lite models available on its free tier, and Groq is constrained to the quota attached
to a free-tier account. Exhausted quota must fail visibly; NeuroRouter will not silently switch to
a billable model. Provider model IDs remain configuration rather than application logic because
free-tier catalogs and limits can change.

## Architecture direction

```text
User -> State Builder -> Jev Router -> Policy Engine -> Execution Plan
     -> Specialist Agents -> Context Aggregator -> LLM Synthesizer
     -> Jev Quality Gate -> Accept / Retry / Review
```

The SQLite schema is forward-compatible with this lifecycle: the request trace stores snapshots,
while ordered stage events store timings, dependencies, structured payloads, and errors.
