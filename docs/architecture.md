# NeuroRouter architecture

NeuroRouter is a synchronous control plane with bounded asynchronous execution. It deliberately
keeps probabilistic classification, deterministic authorization, specialist work, language
generation, and quality control in separate modules.

## Component boundaries

```mermaid
flowchart TB
    subgraph Interface
        UI[Streamlit pages]
        CLI[Evaluation CLIs]
    end

    subgraph Control[Control plane]
        SB[StateBuilder]
        JR[JevRouter]
        PE[PolicyEngine]
        OR[Orchestrator]
        CA[ContextAggregator]
        SY[Synthesizer]
        QG[QualityGate]
        RC[RetryController]
    end

    subgraph Specialists[Bounded specialists]
        GA[GeneralAgent]
        WA[WebResearchAgent]
        RA[RAGAgent]
        CO[CodeAgent]
        FA[FinanceAgent]
    end

    subgraph Providers[Replaceable providers]
        JEV[TypeSafe / Jev]
        WEB[WebSearchProvider]
        VS[VectorStore]
        EMB[EmbeddingProvider]
        LLM[LLMProvider]
        RUN[CodeExecutionTool]
    end

    subgraph Persistence
        SQL[(SQLite)]
        CH[(ChromaDB)]
        CFG[YAML + environment]
    end

    UI --> SB --> JR --> PE --> OR
    JR --> JEV
    OR --> GA & WA & RA & CO & FA
    WA --> WEB
    RA --> VS --> CH
    VS --> EMB
    CO -. explicit opt-in .-> RUN
    GA & WA & RA & CO & FA --> CA --> SY --> QG --> RC
    SY --> LLM
    QG --> JEV
    RC -. bounded retry .-> OR
    SB & JR & PE & OR & CA & SY & QG & RC --> SQL
    CFG --> SB & PE & OR & SY & QG
    CLI --> JR
```

Provider interfaces isolate external APIs. A model, web, embedding, vector, or execution provider
can change without rewriting policy or agent contracts.

## Request lifecycle

```mermaid
sequenceDiagram
    actor User
    participant CP as Control Plane
    participant Jev as Jev Router
    participant Policy as Policy Engine
    participant Agents as Specialist Agents
    participant LLM as LLM Synthesizer
    participant Gate as Jev Quality Gate
    participant DB as SQLite Trace Store

    User->>CP: request + conversation + attachments
    CP->>DB: create trace and persist validated state
    CP->>Jev: one batch of atomic routing questions
    Jev-->>CP: distributions, confidences, latency, raw response
    CP->>Policy: routing decision + capability state
    Policy-->>CP: deterministic execution plan + reasons
    CP->>DB: persist route and plan
    par independent work
        CP->>Agents: execute dependency-ready specialists
    end
    Agents-->>CP: structured results, evidence, sources, timings
    CP->>LLM: request + plan + budgeted evidence packet
    LLM-->>CP: candidate response + usage
    CP->>Gate: request + candidate + evidence metadata
    Gate-->>CP: six atomic Noul probabilities
    alt accepted
        CP-->>User: final response
    else bounded retry
        CP->>Agents: retrieve / reconcile if policy requires
        CP->>LLM: regenerate with updated constraints
    else review
        CP-->>User: best response + uncertainty notice
    end
    CP->>DB: finalize trace
```

## Routing contract

The initial Jev call does not ask for a workflow. It evaluates ten independent judgments that
share the same state:

- one intent `Choice` across seven semantically described options;
- seven `Noul` propositions for web, RAG, code, data analysis, current information, citations, and
  multi-source research;
- ordered complexity and risk `Score` questions.

The `RoutingDecision` stores normalized values and complete probability distributions. A Noul is
already the probability that its proposition is true; it is not converted into a Choice
confidence.

## Deterministic planning

`PolicyEngine` compares those probabilities with `config/thresholds.yaml`, intersects requested
tools with runtime capabilities, selects an LLM tier primarily from complexity, and emits plain
policy explanations. The Decision Lab can replay the same stored decision against other thresholds
without calling Jev or an LLM.

This separation makes model judgment and business policy independently testable:

```text
probability (Jev) + threshold/capability (Python) = execution decision
```

## Quality and retry state machine

```mermaid
stateDiagram-v2
    [*] --> Candidate
    Candidate --> Accepted: all deterministic quality rules pass
    Candidate --> Retrieve: weak support / more retrieval needed
    Candidate --> Regenerate: unsupported or missing claims
    Candidate --> Reconcile: contradiction detected
    Retrieve --> Candidate: retry budget remains
    Regenerate --> Candidate: retry budget remains
    Reconcile --> Candidate: retry budget remains
    Candidate --> Review: risk policy, gate failure, or retry limit
    Accepted --> [*]
    Review --> [*]
```

The quality batch asks whether the response answers the request, is supported, contains unsupported
claims, omits important information, contradicts evidence, or needs more retrieval. Python policy
composes the result. `quality.max_retries` defaults to two.

## Persistence model

SQLite stores one request-level trace plus ordered stage events. Snapshots include validated state,
routing distributions, the execution plan, agent and tool results, token usage, synthesis, quality
attempts, retries, final status, and bounded errors. ChromaDB stores only indexed chunks and their
retrieval metadata. Credentials are loaded through a separate settings object and are never part of
the router state.

## Design invariants

1. Jev makes atomic judgments; it does not generate user-facing prose or choose a full workflow.
2. Only deterministic policy authorizes capabilities and retry actions.
3. Agents are bounded executors, not autonomous recursive planners.
4. External providers are selected explicitly; no silent paid-provider fallback exists.
5. Every loop is bounded and every stage can be traced.
6. Failure in one optional specialist does not erase successful independent results.
7. Dashboard metrics represent persisted executions, never seeded marketing data.
