# NeuroRouter

**A Jev-powered probabilistic control plane for multi-agent AI systems.**

NeuroRouter separates probabilistic routing judgments from deterministic policy. Jev will
classify atomic properties of a request, Python policy will turn those probabilities into an
execution plan, bounded specialist agents will gather evidence, and an LLM will synthesize only
when needed. Every stage is designed to be inspectable through persisted traces.

## Phase 1 status

This repository currently provides the foundation:

- validated Pydantic contracts for state, routing, execution, and traces;
- typed YAML and environment configuration without secrets in runtime state;
- a thread-safe SQLite trace repository and request tracer;
- a multi-page Streamlit control-center shell;
- a narrow Jev client protocol ready for the Phase 2 adapter;
- unit tests for deterministic foundation code.

Agent execution, live Jev calls, RAG, synthesis, and quality-gate behavior are intentionally not
implemented yet.

## Local setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
copy .env.example .env  # Windows; use `cp` on macOS/Linux
streamlit run app.py
```

No API key is required to run Phase 1. The local trace database is created at
`data/neurorouter.db` on first use.

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

Environment overrides use the `NEUROROUTER_` prefix and `__` for nested keys, for example
`NEUROROUTER_TELEMETRY__DATABASE_PATH=data/demo.db`.

## Architecture direction

```text
User -> State Builder -> Jev Router -> Policy Engine -> Execution Plan
     -> Specialist Agents -> Context Aggregator -> LLM Synthesizer
     -> Jev Quality Gate -> Accept / Retry / Review
```

The SQLite schema is forward-compatible with this lifecycle: the request trace stores snapshots,
while ordered stage events store timings, dependencies, structured payloads, and errors.
