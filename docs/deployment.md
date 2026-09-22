# Deployment guide

NeuroRouter is designed to run locally first. The default configuration uses SQLite, local Chroma,
Wikipedia search, hashing embeddings, and a mock synthesizer, so a clone can start without an
account or API charge.

## Local development

```bash
python -m venv .venv
# activate the environment
python -m pip install -e ".[dev,rag]"
cp .env.example .env
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` and copy with
`Copy-Item .env.example .env`.

Generated state lives under `data/` and is excluded from version control. Remove or archive that
directory between public demos when traces may contain private request text.

## Docker

```bash
docker build -t neurorouter .
docker run --rm -p 8501:8501 --env-file .env -v neurorouter-data:/app/data neurorouter
```

The image runs as a non-root user and exposes a Streamlit health check. The named volume preserves
SQLite traces and Chroma indexes across containers. Do not bake `.env` or Streamlit secrets into an
image.

## Streamlit Community Cloud

1. Fork the repository and create an app targeting `app.py`.
2. Use Python 3.11 or later.
3. Add credentials in the platform secret/environment settings; never commit them.
4. Keep `NEUROROUTER_LLM__PROVIDER=mock` unless a hosted free-tier key is configured.

Community-hosted filesystems may be ephemeral. SQLite traces and local Chroma data can disappear on
restart, so this deployment is appropriate for a portfolio demo, not durable production storage.
For persistence, mount durable storage or replace the repository/vector interfaces with managed
services.

## Optional external services

| Service | Required? | Purpose | Failure behavior |
| --- | --- | --- | --- |
| TypeSafe / Jev | No | probabilistic routing and quality judgments | configured conservative fallback |
| Groq, Gemini, or OpenRouter | No | natural-language synthesis | request fails visibly; no provider hopping |
| ChromaDB extra | Only for RAG | local vector storage | RAG capability is unavailable |
| Wikipedia | No | keyless development web evidence | remaining agents can continue |

Provider keys belong in `.env` locally or the hosting platform's secret manager. `settings.yaml`
contains key names, never secret values.

## Zero-billing posture

- The mock provider is the only unconditional zero-cost option and remains the default.
- `llm.allow_paid_models` is false.
- OpenRouter accepts only its free router or model IDs ending in `:free`.
- Groq and Gemini calls use the quota/policy of the supplied account. Use accounts without billing
  enabled if charges must be impossible.
- There is no automatic provider failover, because that could cross a billing boundary.

Free-tier catalogs and quotas change. Review `config/settings.yaml` against the provider's official
model and pricing pages before a public deployment.

## Production considerations

The included runtime is intentionally a single-process portfolio architecture. Before production,
add authentication, per-user authorization, durable managed persistence, encryption and retention
policies, request limits, secret rotation, provider-specific retry/backoff, background ingestion,
and an actual isolation boundary for any untrusted code execution. Do not enable the local code
runner in a shared or internet-facing deployment.

## Health and verification

Streamlit exposes `/_stcore/health` on port 8501. Before release, run:

```bash
ruff format --check .
ruff check .
pytest --cov=neurorouter
streamlit run app.py --server.headless=true
```
