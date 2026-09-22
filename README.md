# NeuroRouter

**A Jev-powered probabilistic control plane for multi-agent AI systems.**

NeuroRouter separates probabilistic routing judgments from deterministic policy. Jev will
classify atomic properties of a request, Python policy will turn those probabilities into an
execution plan, bounded specialist agents will gather evidence, and an LLM will synthesize only
when needed. Every stage is designed to be inspectable through persisted traces.

## Implementation status: Phases 1–7

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
- bounded Code and Finance specialists with structured synthesis handoffs;
- disabled-by-default Python execution with AST restrictions, isolated mode, timeouts, and capped
  output;
- deterministic financial ratios and a strict fresh-web-evidence guard for current-market claims;
- a context aggregator with evidence normalization, deduplication, ranking, citations, and a
  deterministic character budget;
- provider-neutral LLM contracts and working Mock, Groq, Gemini, and OpenRouter adapters;
- complexity-tier model routing, versioned synthesis prompts, and an evidence-aware synthesis
  pipeline;
- unit tests for deterministic foundation code.

The Jev quality gate and retry controller are intentionally not implemented yet. The Jev adapter
is ready for a `TYPESAFE_API_KEY`, while all provider tests use injected transports and require no
external service or quota.

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

- `config/settings.yaml`: runtime capabilities, telemetry, aggregation limits, and model tiers.
- `config/thresholds.yaml`: deterministic routing and quality policy thresholds.
- `config/prompts.yaml`: versioned synthesis instructions and evidence-rendering template.
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

## Code and finance safety

`CodeAgent` classifies generation, debugging, explanation, and data-analysis work and inspects
supplied Python syntax. Execution requires both an explicit request context flag and an enabled
`CodeExecutionTool`. The default tool never executes code. The optional local runner supports only
a calculation-oriented Python subset: imports, attributes, definitions, dynamic calls, and file
access are rejected; execution uses isolated interpreter mode with a strict timeout and output cap.
It is intentionally not described as a general-purpose security sandbox.

`FinanceAgent` calculates common ratios only from structured values supplied to it and preserves
upstream web sources as evidence. If the execution plan requests fresh web information but the web
agent supplies no evidence, Finance returns a review state and explicitly blocks current-market
claims.

Environment overrides use the `NEUROROUTER_` prefix and `__` for nested keys, for example
`NEUROROUTER_TELEMETRY__DATABASE_PATH=data/demo.db`.

### Free-tier LLM providers

The application defaults to `mock`, which requires no key and cannot incur API charges. The
configuration also defines working interchangeable `groq`, `gemini`, and `openrouter` synthesis
providers. Set the matching key in `.env` and set `NEUROROUTER_LLM__PROVIDER` to its name.

`allow_paid_models` defaults to `false`. OpenRouter is runtime-validated to use only
`openrouter/free` or explicit `:free` models. Gemini uses stable Flash models listed for its free
tier, and Groq uses GPT-OSS models within the quota attached to a free account. Use Gemini and Groq
keys from projects/accounts without billing enabled if zero billing exposure is required. Exhausted
quota or missing credentials fail visibly; NeuroRouter never retries through another provider or
silently selects a billable model. Provider model IDs remain configuration rather than application
logic because free-tier catalogs and limits change.

## Architecture direction

```text
User -> State Builder -> Jev Router -> Policy Engine -> Execution Plan
     -> Specialist Agents -> Context Aggregator -> LLM Synthesizer
     -> Jev Quality Gate -> Accept / Retry / Review
```

The SQLite schema is forward-compatible with this lifecycle: the request trace stores snapshots,
while ordered stage events store timings, dependencies, structured payloads, and errors.
