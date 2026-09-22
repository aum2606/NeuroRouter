# Demo runbook

This runbook is designed for a 30–60 second screen recording. It uses observable system behavior
rather than prepared benchmark claims.

## Before recording

1. Install the project with `python -m pip install -e ".[dev,rag]"`.
2. Copy `.env.example` to `.env`.
3. Keep `NEUROROUTER_LLM__PROVIDER=mock` for a guaranteed no-key run, or select one configured
   hosted free-tier provider and add only its key.
4. Start with `streamlit run app.py` and open the Command Center.
5. Optionally upload a short PDF or Markdown file on the Knowledge Base page.

If `TYPESAFE_API_KEY` is absent, the UI correctly labels routing as fallback behavior. For a demo of
real probability distributions, configure Jev before recording.

## 45-second narrative

1. **Ask:** “Compare the latest public announcements about Gemini and Groq for a low-cost AI
   prototype, cite the sources, and flag anything uncertain.”
2. **Point to routing:** intent, web probability, citation probability, complexity, and risk are
   separate judgments—not one opaque workflow decision.
3. **Point to the graph and timeline:** enabled specialists, concurrency, dependencies, and measured
   stage latency are visible while the request runs.
4. **Open the quality result:** show the independent support, omission, contradiction, and retrieval
   probabilities plus the final accept/retry/review action.
5. **Open Decision Lab:** lower or raise the web threshold and show the counterfactual plan change
   without another model call.
6. **Finish in Trace Explorer:** show that the route, plan, sources, agent timings, model tier, retries,
   and final status were persisted under one trace ID.

Expected behavior depends on live Jev probabilities and current source availability; do not promise
a specific score or route in the recording.

## Additional scenarios

### Local-document route

Upload a document, then ask:

> According to the uploaded document, what are the three main design constraints? Cite the relevant
> chunks and say when the document does not support a conclusion.

Use this to show page/chunk metadata, local hashing embeddings, RAG activation, and evidence-bound
synthesis.

### Code route

> Explain why this Python function returns duplicate items and propose a tested correction:
> `def unique(xs): return list(set(xs))`

Use this to show CodeAgent activation and that execution remains disabled by default. The useful
discussion is the lost ordering, not arbitrary shell execution.

### Finance guardrail

> Compare the current valuation of two public companies and cite the market data used.

Use this to show the fresh-evidence rule. If web evidence is unavailable, FinanceAgent should not
invent current values; the trace should expose a degraded or review outcome.

### Counterfactual policy

Choose a stored trace with `needs_web` near the configured threshold. In Decision Lab, move only the
web threshold across that probability. Show that the stored model judgment stays fixed while the
deterministic plan changes.

## Recording tips

- Use the dark theme and a 16:9 browser window.
- Keep the right-side probability and latency panels visible while submitting.
- Expand raw payloads only briefly; lead with normalized decisions and policy explanations.
- Avoid presenting fallback routing as a real Jev result.
- Avoid quoting evaluation metrics until a measured run exists in the Evaluations page.
