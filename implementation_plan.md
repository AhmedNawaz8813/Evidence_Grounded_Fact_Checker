# TRUST Agents v2 — Final Implementation Plan

## Objective

Replace the label-dependent, dataset-derived evidence corpus with externally retrieved, source-grounded evidence and measure whether improved evidence coverage materially improves fact-checking performance.

The current [`prepare_evidence_corpus.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/scripts/prepare_evidence_corpus.py) constructs its evidence corpus by selecting training records where `label == 'true'` (line 33). This means the retrieval corpus is built using ground-truth labels and therefore does not represent independent external evidence — a serious methodological flaw that compromises the validity of the retrieval stage, even though it technically uses training data rather than test labels.

### Three Independent Research Questions

1. **Does external evidence improve retrieval?**
2. **Does better evidence improve the existing verifier?**
3. **Does the multi-agent reasoning architecture add value once evidence quality is no longer the main bottleneck?**

---

## Architecture

### Standard Pipeline

```
                     CLAIM
                       │
                       ▼
                Claim Extraction
                (bypass for short statements)
                       │
                       ▼
                 Query Generation
                       │
                       ▼
                Search Backend(s)
                       │
                       ▼
              Document Retrieval
                       │
                       ▼
              Passage Extraction
                       │
                       ▼
              Evidence Ranking
                       │
                       ▼
              Evidence Selection
                       │
                       ▼
                 Existing Verifier  ← UNCHANGED
                       │
                       ▼
                Existing Explainer  ← UNCHANGED
```

### Research Pipeline (Phase 8+)

```
                   High-quality evidence
                           │
                     Decomposition
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
              Atomic 1  Atomic 2  Atomic 3
                 │         │         │
               Jury      Jury      Jury
                 └─────────┼─────────┘
                           ▼
                    Logic Aggregator
```

---

## Core Design Principle: Search ≠ Evidence

```
Search Result → Document → Relevant Passage → Evidence
```

Four-stage process:
- **Stage A — Search:** Find potentially relevant documents
- **Stage B — Document retrieval:** Download/read promising documents
- **Stage C — Passage retrieval:** Locate exact relevant paragraphs/sentences
- **Stage D — Evidence selection:** Rank by relevance, source quality, directness

---

## File Changes

### New Files

#### [NEW] `TRUST_agents/agents/web_search_retriever.py`
Core retrieval engine:
- `SearchBackend` abstract base class
- `WikipediaBackend` (Phase 2A)
- `WebSearchBackend` stub (Phase 5)
- `WebSearchRetriever.retrieve(claim, top_k)` — main entry point
- Internal: query_gen → search → fetch → passage_split → rank → deduplicate → bundle
- **LIAR fast-path:** Single-sentence inputs bypass claim extraction, go directly to query generation. Multi-proposition detection triggers standard extraction path.

#### [NEW] `TRUST_agents/agents/evidence_ranker.py` (Phase 2C)
- `EvidenceRanker` with configurable scoring weights
- Heuristic scoring: semantic_sim + lexical_overlap + entity_match + temporal_match + source_quality
- MMR diversity selection (adapted from existing [`mmr_rerank()`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/agents/retrieval_agent_core.py#L149-L176))
- No cross-encoder until Phase 6+

#### [NEW] `TRUST_agents/agents/source_quality.py` (Phase 2C)
- Source type classification: wikipedia, government, news, academic, blog, unknown
- Fixed quality scores per type
- URL/domain parsing

#### [NEW] `TRUST_agents/agents/retrieval_cache.py` (Phase 2B)
- SQLite-backed cache keyed by query hash + backend type
- Stores: query, timestamp, source, URL, content, scores
- TTL support for expiration
- Cache hit/miss logging

#### [NEW] `scripts/evaluate_retrieval.py`
- Retrieval metrics: evidence_found_rate, avg_evidence_count, source_diversity, avg_relevance_score
- Coverage = non-uncertain / total predictions
- Selective accuracy = correct non-uncertain / total non-uncertain
- Evidence sufficiency rate

#### [NEW] `scripts/analyze_failures.py`
- Failure taxonomy (F1–F8):
  - F1 — Wrong claim extraction
  - F2 — Search failure (no results)
  - F3 — Missing evidence (results found but irrelevant)
  - F4 — Irrelevant evidence selected
  - F5 — Conflicting evidence
  - F6 — Incorrect verifier reasoning
  - F7 — Incorrect uncertainty handling
  - F8 — Logic aggregation error
- Per-type counts and example export

#### [NEW] `data/gold_evidence/` (Phase 3B)
- 30–50 manually annotated LIAR claims:
  ```json
  {
    "claim": "...",
    "gold_label": "true|false",
    "evidence_urls": ["..."],
    "relevant_passages": ["..."],
    "evidence_relationship": "supports|contradicts",
    "source_quality": "high|medium|low"
  }
  ```
- Used to evaluate retriever independently of the LLM verifier

---

### Modified Files

#### [MODIFY] [`orchestrator.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/orchestrator.py)
- Add `evidence_source` parameter: `"local" | "web"` (add `"both"` in Phase 5+)
- Route to `WebSearchRetriever` when `evidence_source="web"`
- All other logic unchanged (verifier, explainer, normalization)

#### [MODIFY] [`orchestrator_research.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/orchestrator_research.py)
- Same `evidence_source` parameter
- Per-atomic-claim retrieval already exists (line 192) — just swap retrieval function

#### [MODIFY] [`evaluate_predictions.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/scripts/evaluate_predictions.py)
- Add per-class precision and recall
- Add coverage and selective accuracy metrics
- Add experiment source label for comparison

#### [MODIFY] [`run_trust_agents.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/scripts/run_trust_agents.py) / [`run_trust_research.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/scripts/run_trust_research.py)
- Add `--evidence-source` flag: `local | web`
- Skip local index check when `evidence_source="web"`
- **Add per-run cost tracking:** LLM calls, search calls, tokens in/out, estimated API cost, runtime

#### [MODIFY] [`requirements.txt`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/requirements.txt)
- Add `wikipedia-api>=0.6.0`
- (Phase 5) Add search API package

---

### Unchanged Files (Experimental Control)

| File | Reason |
|------|--------|
| [`verifier.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/agents/verifier.py) | Must stay constant for causal clarity |
| [`verifier_tools.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/agents/verifier_tools.py) | Same |
| [`claim_extractor.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/agents/claim_extractor.py) | Same |
| [`explainer.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/agents/explainer.py) | Same |
| [`decomposer_agent.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/agents2.0/decomposer_agent.py) | Same |
| [`delphi_jury.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/agents2.0/delphi_jury.py) | Same |
| [`logic_aggregator.py`](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/agents2.0/logic_aggregator.py) | Same |

---

## `hybrid_score` Compatibility

The verifier ([line 62](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/agents/verifier.py#L62)) and Delphi jury ([line 147](file:///c:/Users/ahmed/Desktop/Ahmed/Research%20Intern%20work/TRUST%202/news_agent/TRUST_agents/agents2.0/delphi_jury.py#L147)) both read `hybrid_score` from evidence items.

**Strategy:** Map `evidence_score` → `hybrid_score` in the evidence bundle for Phases 1–6. This preserves compatibility without touching the verifier. Document the changed semantics in a comment. Move to a proper schema change (separate `retrieval_score`, `evidence_score`, `source_quality` fields) only in Phase 8+.

---

## Evidence Bundle Format

```json
{
  "text": "...",
  "title": "...",
  "source": "en.wikipedia.org",
  "url": "https://en.wikipedia.org/wiki/...",
  "source_type": "wikipedia",
  "passage_id": "wiki_Obama_healthcare_3",
  "hybrid_score": 0.88,
  "metadata": {
    "retrieval_score": 0.91,
    "relevance_score": 0.88,
    "source_quality": 0.90,
    "evidence_score": 0.88,
    "query_used": "Affordable Care Act insurance premiums 2010"
  }
}
```

The `hybrid_score` top-level field maintains verifier compatibility. The full breakdown lives in `metadata` for analysis.

---

## Per-Experiment Cost Tracking

Every run records:

| Metric | Description |
|--------|-------------|
| `num_claims` | Total claims processed |
| `llm_calls` | Total LLM API calls |
| `search_calls` | Total search API calls |
| `tokens_in` | Total input tokens |
| `tokens_out` | Total output tokens |
| `estimated_cost_usd` | Estimated API cost |
| `runtime_seconds` | Wall-clock time |
| `cost_per_sample` | Cost / num_claims |

Target output table:

| System | Accuracy | Macro-F1 | Coverage | Cost/sample | Latency |
|--------|----------|----------|----------|-------------|---------|
| Local baseline | — | — | — | — | — |
| Wikipedia | — | — | — | — | — |
| Wikipedia + ranking | — | — | — | — | — |
| Web + reranking | — | — | — | — | — |
| Research pipeline | — | — | — | — | — |

---

## Implementation Phases

### Phase 1 — Establish Baseline
Run the current system on a **fixed stratified sample of 200 LIAR val examples** (100 true, 100 false). Record: accuracy, macro-F1, coverage, uncertain rate, runtime, cost. Save exact predictions.

### Phase 2A — Wikipedia Retrieval (Minimal)
Build `WebSearchRetriever` with `WikipediaBackend` only:
- Claim → Wikipedia search API → fetch top-N articles → sentence/paragraph extraction → basic top-k selection → existing verifier
- 2 queries per claim: original + entity-focused
- No caching, no ranking, no source scoring
- Rate limiting with `tenacity` (Wikipedia: 200 req/min)

### Phase 2B — Add Caching
Add `retrieval_cache.py`:
- SQLite cache for search results and fetched documents
- Cache hit/miss logging
- Enables reproducible re-runs

### Phase 2C — Add Heuristic Evidence Ranking
Add `evidence_ranker.py` + `source_quality.py`:
- Heuristic scoring with configurable weights
- MMR diversity selection
- Source quality classification

**Ablation table from Phase 2:**

| Version | Retrieval | What it tells you |
|---------|-----------|-------------------|
| Baseline | Local BM25 + FAISS | Current performance |
| V1 | Wikipedia search (basic top-k) | Does external evidence alone help? |
| V2 | Wikipedia + heuristic ranking | Does evidence quality selection matter? |
| V3 | Wikipedia + ranking + source quality | Does source awareness matter? |

### Phase 3A — Stratified Manual Retrieval QA
Sample **35–45 claims** stratified:
- 10–15 true claims
- 10–15 false claims
- 10–15 difficult/ambiguous claims

Score each retrieved evidence bundle on:

| Dimension | Question |
|-----------|----------|
| **Relevance** | Is it actually about the claim? |
| **Sufficiency** | Could a reasonable verifier decide from it? |
| **Source quality** | Is the source credible? |
| **Directness** | Does it directly address the proposition? |

Score each dimension 0–3 (none / weak / moderate / strong). This produces a small **retrieval quality dataset** rather than informal inspection.

### Phase 3B — Gold Evidence Analysis Set
Create `data/gold_evidence/` with **30–50 manually annotated LIAR claims**:
- Claim text + correct label
- Relevant evidence URL(s) and passages
- Evidence relationship (supports/contradicts)
- Source quality rating

This enables **retriever-only evaluation** independent of the LLM verifier:
```
Retriever finds correct evidence? → YES/NO
Verifier gets correct answer?     → YES/NO
```

Four outcome categories:
- ✅ Good retrieval + correct verdict → system works
- ⚠️ Good retrieval + wrong verdict → verifier problem
- ⚠️ Bad retrieval + correct verdict → lucky guess
- ❌ Bad retrieval + wrong verdict → retrieval problem

### Phase 4 — Wikipedia Experiment
Run on same 200 stratified samples. Compare V1/V2/V3 against Phase 1 baseline.

### Phase 5 — Web Search Backend
Add `WebSearchBackend` (requires API key decision). Expand to 4 query formulations.

### Phase 6 — Advanced Reranking
Optional cross-encoder reranking. Measure whether it improves over heuristic ranking.

### Phase 7 — Adaptive Retrieval
Evidence sufficiency gate. Max 2–3 retrieval rounds. Only search again when evidence is insufficient.

### Phase 8 — Research Pipeline
Web evidence + Delphi jury + Logic aggregation. Compare against single-verifier baseline with identical evidence.

### Phase 9 — Final Evaluation
Full LIAR val/test set. Complete metrics. Ablation table. Cost analysis. Failure taxonomy.

---

## LIAR Fast-Path

For single short factual statements (typical LIAR format):

```
Single short factual statement
        ↓
Multi-proposition check
        ├── Single proposition → bypass claim extraction → query generation
        └── Multiple propositions → standard claim extraction pipeline
```

Detection heuristic: if input is ≤ 2 sentences and contains no conjunction-linked independent clauses, treat as single claim.

---

## Verification Plan

### Automated
```bash
# Phase 2A: Unit test Wikipedia backend
python -m pytest tests/test_wikipedia_backend.py -v

# Phase 2A: Integration test — 5 known claims
python scripts/test_web_retrieval.py --claims 5

# Phase 4: Full comparison
python scripts/run_trust_agents.py --dataset liar --split val --limit 200 --evidence-source web
python scripts/evaluate_predictions.py --preds outputs/trust_agents/liar_web_predictions.jsonl

# Phase 3B: Retriever-only evaluation against gold evidence
python scripts/evaluate_retrieval.py --gold data/gold_evidence/ --retriever web
```

### Manual
- Phase 3A: Stratified retrieval QA with scoring rubric
- Phase 3B: Gold evidence annotation (30–50 claims)
- Compare evidence bundles side-by-side: local vs. web
- Check for prompt injection vectors in retrieved web content

---

## Open Questions (Requiring User Input)

> [!IMPORTANT]
> ### Q1. Wikipedia API package choice
> - **`wikipedia-api`** (simpler, auto-handles redirects) — **recommended for Phase 2A**
> - **Direct `requests` to MediaWiki API** (more control)
> - **`mwclient`** (full client)

> [!IMPORTANT]
> ### Q2. Do you have a search API key for Phase 5?
> Google Custom Search, SerpAPI, Brave Search, or DuckDuckGo (free but limited)?

> [!IMPORTANT]
> ### Q3. Internal label taxonomy timing
> Keep `true/false/uncertain` internally for Phases 1–6, add `SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE` in Phase 8+ (avoids touching verifier prompt during controlled experiment)?
