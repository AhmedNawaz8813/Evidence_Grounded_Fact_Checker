# TRUST Agents v2 — Complete Project Explanation (The Whole 9 Yards)

---

## 1. HIGH-LEVEL GOAL

The goal of this project is to build and evaluate a **multi-agent AI system for automated fact-checking** using Large Language Models (LLMs). The system takes a factual claim (e.g., a political statement from a dataset like LIAR) and determines whether it is **true**, **false**, or **uncertain**, along with a confidence score and reasoning.

The project has two major phases:

1. **TRUST Agents v1 (Original Pipeline)**: A 4-step agent pipeline (Claim Extraction → Local Evidence Retrieval → Verification → Explanation) using a local evidence corpus built from the training dataset itself.
2. **TRUST Agents v2 (This Work)**: Replaces the flawed local evidence corpus with **externally retrieved, source-grounded evidence from Wikipedia** (and eventually the broader web), and adds a research-grade pipeline with **claim decomposition, multi-agent jury verification, and logical aggregation**.

The core **research questions** are:
1. Does external evidence (from Wikipedia/web) improve retrieval quality over the label-dependent local corpus?
2. Does better evidence improve the existing LLM verifier's accuracy?
3. Does the multi-agent reasoning architecture (decomposition + jury + logic aggregation) add value once evidence quality is no longer the bottleneck?

---

## 2. THE PROBLEM: WHY THIS PROJECT EXISTS

### 2.1 The Methodological Flaw in v1

The original system had a critical flaw in how it built its evidence corpus. The script `prepare_evidence_corpus.py` (line 33) constructs its evidence corpus by selecting training records where `label == 'true'`. This means:

- The retrieval corpus is built using **ground-truth labels**.
- The evidence is not independent external evidence — it's recycled training data filtered by the answer key.
- Even though it uses training data (not test labels), this is still a **serious methodological compromise** because the evidence corpus inherently encodes label information, making retrieval trivially biased.

**In a real-world fact-checking scenario, you don't have access to labels.** You need to find evidence from external sources (Wikipedia, news articles, government databases, etc.) and reason over it.

### 2.2 The Uncertain Prediction Problem

The v1 system (and even v2) suffers from an extremely high "uncertain" rate. In v1 experiments:
- **70.5% of predictions** were "uncertain" (141 out of 200 claims).
- In the research pipeline, this rose to **82.5%** (165 out of 200).

The LIAR dataset has binary gold labels (true/false), so every "uncertain" prediction is automatically wrong when evaluated, regardless of the uncertain-handling strategy used (mapping uncertain→false, uncertain→true, or dropping them).

### 2.3 The Cost-Accuracy Tradeoff

The project uses GPT-4.1-mini (a smaller, cheaper model) for all LLM calls. More capable models (GPT-4o, GPT-4-turbo) would likely produce better results but cost 10-20x more per API call. Since the pipeline makes multiple LLM calls per claim (claim extraction, verification, explanation, and in the research pipeline: decomposition + 4 jury verifiers + explanation), costs scale quickly.

---

## 3. DATASET: LIAR

The system is evaluated on the **LIAR dataset**, a widely-used benchmark for fake news detection:

- ~12,800 short political statements from PolitiFact.com
- Originally has 6 labels (pants-fire, false, barely-true, half-true, mostly-true, true)
- **Binarized** for this project: mapped to just `true` and `false`
- Typical examples are short, 1-2 sentence factual claims by politicians
- The dataset is split into train/val/test; experiments use the **validation set** (or a stratified subset of 200 examples)

The processed data lives in `data/processed/` as JSONL files (one JSON object per line, with fields `id`, `text`, and `label`).

---

## 4. ARCHITECTURE OVERVIEW

### 4.1 Standard Pipeline (agents/)

```
CLAIM (raw text input)
    │
    ▼
[1] Claim Extractor Agent
    - Uses NER (Named Entity Recognition), dependency parsing, and LLM
    - Extracts individual factual claims from the input text
    - Implemented as a ReAct agent (LangGraph + LangChain)
    │
    ▼
[2] Evidence Retriever
    - LOCAL mode: BM25 + FAISS dense retrieval over local corpus
    - WEB mode: Wikipedia API retrieval → passage extraction → ranking
    - Returns top-k evidence passages with scores
    │
    ▼
[3] Verifier Agent
    - ReAct agent that compares claims against evidence
    - Tools: compare_claim_evidence, aggregate_evidence, generate_verdict, confidence_calibration
    - Outputs: verdict (true/false/uncertain), confidence (0.0-1.0), reasoning
    │
    ▼
[4] Explainer Agent
    - ReAct agent that generates human-readable explanations
    - Tools: summarize_verification, generate_explanation, cite_evidence, create_report
    - Outputs: complete fact-check report with citations
    │
    ▼
FINAL OUTPUT: {verdict, confidence, label, reasoning, summary, evidence_count}
```

### 4.2 Research Pipeline (agents2.0/)

```
CLAIM (raw text input)
    │
    ▼
[1] Decomposer Agent (LoCal-inspired)
    - Breaks claim into ATOMIC sub-claims
    - Identifies LOGICAL structure (AND, OR, IMPLIES, NOT)
    - Detects CAUSAL relationships (cause → effect)
    - Assigns complexity score (0.0-1.0)
    │
    ▼
[2] For EACH atomic claim:
    ├── Evidence Retrieval (local or web)
    ├── Verification via Delphi Jury OR single verifier
    │
    │   DELPHI JURY (4 personas, each an independent LLM call):
    │   ├── StrictLegalist: Only trusts .gov, .edu, major news
    │   ├── OpenWebPragmatist: Considers broader sources
    │   ├── CausalSkeptic: Focuses on temporal/numeric consistency
    │   └── ConspiracyDetector: Flags conspiratorial patterns
    │
    │   Each persona returns: {verdict, confidence, reasoning, key_concerns}
    │   Trust scores computed per persona based on evidence quality + confidence
    │   Final verdict = trust-weighted vote aggregation
    │
    ▼
[3] Logic Aggregator
    - Takes atomic verdicts + logical structure formula
    - Evaluates: e.g., "C1 AND C2" where C1=true, C2=false → false
    - Computes confidence from agreeing atomic verdicts
    - Fallback: majority vote if logic evaluation fails
    │
    ▼
[4] Explainer Agent (same as standard pipeline)
    │
    ▼
FINAL OUTPUT: {verdict, confidence, atomic_verdicts, logic_structure, metadata}
```

### 4.3 The Orchestrators

There are two orchestrator classes that coordinate the pipelines:

- **`TRUSTOrchestrator`** (in `orchestrator.py`): Controls the standard pipeline. Key feature: `_normalize_verdict()` method that handles messy LLM outputs (e.g., mapping "supported" → "true", handling confidence values in 0-100 range vs 0-1 range, extracting verdicts from natural language text).
- **`ResearchTRUSTOrchestrator`** (in `orchestrator_research.py`): Controls the research pipeline. Adds claim decomposition, per-atomic-claim evidence retrieval, Delphi jury, and logic aggregation.

Both accept an `evidence_source` parameter: `"local"` for BM25+FAISS or `"web"` for Wikipedia retrieval.

---

## 5. EVERY COMPONENT IN DETAIL

### 5.1 Claim Extractor (`claim_extractor.py`)

- **What it does**: Takes raw input text and extracts individual factual claims.
- **Implementation**: LangGraph ReAct agent with 3 tools:
  - `ner_claim_extraction_tool`: Uses spaCy NER to find entities and build claims
  - `dependency_claim_extraction_tool`: Uses dependency parsing to extract grammatical claims
  - `llm_claim_extraction_tool`: Uses the LLM directly for zero-shot claim extraction
- **LLM**: GPT-4.1-mini (temperature=0.1 for consistency)
- **Output**: A JSON list of claim strings, e.g. `{"claims": ["Biden won the 2020 election", "Biden became president in 2021"]}`
- **Problem observed**: Sometimes extracts near-duplicate claims (e.g., the original statement and a paraphrased version), inflating the claim count without adding value.

### 5.2 Local Evidence Retrieval (`evidence_retrieval.py` + `retrieval_agent_core.py`)

- **What it does**: Finds relevant evidence passages from a pre-indexed local corpus.
- **Implementation**: LangGraph ReAct agent with tools:
  - `search_evidence_tool`: Hybrid BM25 + dense (sentence-transformers + FAISS) retrieval
  - `get_passage_tool`: Retrieve a specific passage by ID
  - `list_indexed_documents_tool`: List all indexed documents
- **Retrieval core** (`retrieval_agent_core.py`, 441 lines):
  - Uses `rank_bm25.BM25Okapi` for sparse retrieval
  - Uses `sentence-transformers` for dense embeddings
  - Uses `FAISS` for approximate nearest neighbor search
  - Supports MMR (Maximal Marginal Relevance) re-ranking for diversity
  - Can ingest PDF, HTML, and text files
  - Sentence-aware chunking with configurable chunk size (default 160 words)
  - Save/load index capability
- **The flaw**: The corpus is built from training data where `label == 'true'` (see `prepare_evidence_corpus.py` line 33), meaning the evidence corpus encodes ground-truth information.

### 5.3 Web Search Retriever (`web_search_retriever.py`) — THE KEY v2 ADDITION

This is the **central new component** of TRUST Agents v2. It replaces the label-dependent local corpus with external evidence from Wikipedia (Phase 2A) and eventually the broader web (Phase 5+).

**Architecture:**
```
Claim → [LIAR fast-path check] → Query Generation → Search Backend(s)
       → Document Retrieval → Passage Extraction → Scoring/Ranking
       → Deduplication → Evidence Bundle
```

**Sub-components:**

#### 5.3.1 LIAR Fast-Path Detection (`_is_single_proposition()`)
- Heuristic to detect single short factual statements (typical LIAR format)
- If input is ≤ 2 sentences and has no conjunction-linked independent clauses → bypass claim extraction
- Detects multi-proposition claims by counting verb phrases (≥ 3 verbs with coordinating conjunctions = multi-proposition)

#### 5.3.2 Query Generation (`_generate_queries()`)
- Generates 2 search queries per claim:
  1. **Cleaned claim**: Original text with quotes/attribution removed
  2. **Entity-focused query**: Extracted proper nouns + numbers with context (e.g., "Barack Obama 44th president")
- Deduplicates queries if both are identical

#### 5.3.3 Search Backend ABC (`SearchBackend`)
- Abstract base class defining the interface: `search(query, max_results) → List[Dict]`
- Each result dict must contain: `title`, `url`, `content`, `source`, `source_type`

#### 5.3.4 Wikipedia Backend (`WikipediaBackend`)
- Uses the `wikipedia-api` Python package
- **Two search strategies**:
  1. Direct title lookup (exact match)
  2. Candidate title generation from query (extracts capitalized noun phrases, proper nouns, and short queries as potential article titles)
- Retry logic via `tenacity`: exponential backoff on network errors, max 3 attempts
- Rate limited to ≤ 200 requests/minute
- Returns full article text, title, URL, source metadata

#### 5.3.5 Passage Extraction (`_split_into_passages()`)
- Splits Wikipedia article text into overlapping sentence-window passages
- Default: 5 sentences per window, 1 sentence overlap
- Uses NLTK `sent_tokenize` for sentence boundary detection
- Short texts (< 30 chars) or texts with ≤ 5 sentences returned as-is
- Max 20 passages per document

#### 5.3.6 Basic Relevance Scoring (Phase 2A fallback) (`_basic_relevance_score()`)
- Used when the full ranker is not available
- **Scoring formula**: `score = 0.6 × word_overlap + 0.4 × entity_overlap`
- Word overlap: Jaccard-like intersection of claim tokens with passage tokens divided by claim token count
- Entity overlap: Proportion of proper nouns from the claim found in the passage
- Length penalty: passages < 10 words get 0.5× penalty, passages > 300 words get 0.8× penalty
- Source quality: Wikipedia gets 0.90, everything else gets 0.50

#### 5.3.7 Top-K Selection with Deduplication (`_select_top_k()`)
- Sorts passages by score descending
- Near-duplicate detection using word-set Jaccard similarity (threshold: 0.85)
- Returns top-k non-duplicate passages

#### 5.3.8 The `WebSearchRetriever` Class (Main Entry Point)
- `retrieve(claim, top_k=5)` is the main method
- Pipeline steps:
  1. LIAR fast-path check
  2. Generate 2 queries
  3. For each backend × each query: check cache → search → store in cache
  4. Deduplicate documents by URL
  5. Extract passages from each document
  6. Score and select using ranker (Phase 2C) or basic scoring (Phase 2A fallback)
  7. Return list of evidence bundle dicts

### 5.4 Evidence Ranker (`evidence_ranker.py`) — Phase 2C

- **What it does**: Heuristic multi-dimensional scoring of evidence passages with MMR diversity selection.
- **Scoring dimensions** (with default weights):
  - `lexical_overlap` (0.30): Word overlap between claim and passage
  - `entity_match` (0.30): Proper noun overlap
  - `temporal_match` (0.10): Number/date overlap
  - `source_quality` (0.20): Quality score based on URL classification
  - `length_penalty` (0.10): Penalizes very short (< 10 words → 0.3) or very long (> 400 words → 0.6) passages; sweet spot 30-150 words → 1.0
- **Weighted sum**: `score = Σ(weight_i × feature_i)` for all dimensions
- **MMR Diversity Selection** (`select()` method):
  - Balances relevance vs. redundancy: `mmr_score = diversity × relevance - (1 - diversity) × max_similarity`
  - Default `diversity=0.7` (heavily favors relevance, with some diversity)
  - Uses word-set Jaccard similarity for redundancy measurement
  - Guarantees at least `min(3, top_k)` selections even if MMR scores are negative

### 5.5 Source Quality Classifier (`source_quality.py`)

- **What it does**: Classifies URLs into source types and assigns fixed quality scores.
- **Source types and scores**:
  - `academic`: 0.93 (.edu, .ac.uk, arxiv.org, etc.)
  - `government`: 0.92 (.gov, .gov.uk, etc.)
  - `wikipedia`: 0.90 (wikipedia.org)
  - `factcheck`: 0.88 (politifact.com, snopes.com, etc.)
  - `news_major`: 0.80 (nytimes.com, bbc.com, reuters.com, etc.)
  - `news_minor`: 0.60 (any domain with "news" in it)
  - `unknown`: 0.40 (default)
  - `blog`: 0.35 (medium.com, substack.com, etc.)
  - `social_media`: 0.20 (twitter.com, reddit.com, etc.)
- Uses URL parsing to extract domain, then checks against curated lists.

### 5.6 Retrieval Cache (`retrieval_cache.py`)

- **What it does**: SQLite-backed cache to avoid redundant Wikipedia API calls and enable reproducible re-runs.
- **Key**: SHA-256 hash of `"{backend_type}::{query.strip().lower()}"` — ensures same query to same backend always hits the same cache entry.
- **Schema**: `(key TEXT PRIMARY KEY, query TEXT, backend TEXT, timestamp REAL, data TEXT)` where data is JSON-serialized search results.
- **TTL**: 7-day expiration by default. Expired entries are deleted on access.
- **WAL mode**: Uses SQLite Write-Ahead Logging for better concurrent performance.
- **Stats tracking**: Hit/miss counters for monitoring cache effectiveness.
- **Location**: `retrieval_index/retrieval_cache.db`

### 5.7 Verifier Agent (`verifier.py`)

- **What it does**: Takes a claim + evidence passages and determines if the claim is true, false, or uncertain.
- **Implementation**: LangGraph ReAct agent with 4 tools:
  - `compare_claim_evidence_tool`: Compare claim against individual evidence passages
  - `aggregate_evidence_tool`: Aggregate multiple evidence assessments into a summary
  - `generate_verdict_tool`: Generate final verdict from aggregated assessment
  - `confidence_calibration_tool`: Calibrate confidence based on evidence quality
- **Key prompt instruction**: "Do not be overly cautious. If the evidence heavily implies a verdict (true or false) but doesn't explicitly state it word-for-word, make an educated deduction rather than defaulting to 'uncertain'."
- **Evidence format**: Takes top 3 evidence passages, shows score and first 300 characters of each.
- **Output parsing**: Extracts JSON `{"verdict": "...", "confidence": 0.X, "reasoning": "..."}` from the LLM's final message using regex.
- **Reads `hybrid_score`** from evidence items (line 62) — this is why the web retriever must set `hybrid_score` as the top-level compatibility field.
- **This agent is UNCHANGED between v1 and v2** — kept constant as an experimental control to isolate the effect of evidence quality.

### 5.8 Explainer Agent (`explainer.py`)

- **What it does**: Generates human-readable explanations of the fact-check result.
- **Implementation**: LangGraph ReAct agent with 4 tools:
  - `summarize_verification_tool`: Create a concise summary
  - `generate_explanation_tool`: Generate detailed explanation with citations
  - `cite_evidence_tool`: Format evidence citations
  - `create_report_tool`: Compile comprehensive report
- **This agent is UNCHANGED between v1 and v2.**

### 5.9 Decomposer Agent (`decomposer_agent.py`) — Research Pipeline

- **What it does**: Decomposes complex claims into atomic sub-claims with logical structure.
- **Based on**: LoCal paper (ACM 2024)
- **Implementation**: Direct OpenAI API call (not ReAct agent) with structured JSON output.
- **Outputs a `DecomposedClaim` dataclass**:
  - `atomic_claims`: List of individually verifiable facts
  - `logic_structure`: Logical formula (e.g., "C1 AND C2", "C1 IMPLIES (C2 OR C3)")
  - `causal_edges`: List of cause-effect relationships with temporal ordering
  - `complexity_score`: 0.0-1.0 indicating claim complexity
- **Example**:
  ```
  Input: "After policy X passed in 2020, unemployment decreased by 5%"
  Output:
    atomic_claims: ["Policy X passed in 2020", "Unemployment decreased by 5%"]
    logic_structure: "C1 AND C2"
    causal_edges: [{"cause": "Policy X passed", "effect": "unemployment decreased", "temporal": true}]
    complexity_score: 0.5
  ```
- **Fallback**: If LLM call fails, treats the entire claim as a single atomic claim (C1).

### 5.10 Delphi Jury (`delphi_jury.py`) — Research Pipeline

- **What it does**: Multi-agent verification with 4 distinct LLM personas and trust-weighted voting.
- **Based on**: Delphi method for expert consensus + Cleanlab BOLAA trust scoring.
- **4 Personas** (each is a separate LLM call with a different system prompt):
  1. **StrictLegalist**: Only trusts .gov, .edu, major news. Very conservative with "true" verdicts. Defaults to "uncertain" for questionable evidence.
  2. **OpenWebPragmatist**: Considers broader sources. Focuses on content quality over source prestige. Uses common sense and contextual reasoning.
  3. **CausalSkeptic**: Verifies temporal ordering (cause before effect). Checks numeric claims. Looks for logical fallacies (correlation ≠ causation).
  4. **ConspiracyDetector**: Identifies conspiracy theory markers, cherry-picked evidence, logical leaps, unfalsifiable claims.
- **Trust Score Computation** (`_compute_trust_scores()`):
  - Base trust from average evidence `hybrid_score` (0.4 weight)
  - Confidence calibration (0.4 weight)
  - Error penalty: 0.2 trust for errored verdicts
  - Bonus: 1.1× multiplier for decisive verdicts (true/false) when evidence quality > 0.6
  - Clamped to [0.0, 1.0]
- **Aggregation** (`_aggregate_with_trust()`):
  - Each persona's weight = trust × confidence
  - Weighted votes accumulated per category (true/false/uncertain)
  - Normalized to sum to 1.0
  - Winner = category with highest weighted vote
  - Final confidence = winner's normalized weighted vote share

### 5.11 Logic Aggregator (`logic_aggregator.py`) — Research Pipeline

- **What it does**: Combines atomic claim verdicts using logical formulas to produce a final compound verdict.
- **Based on**: LoCal paper (ACM 2024)
- **Supported operators**: AND, OR, NOT, IMPLIES
- **IMPLIES handling**: `A IMPLIES B` is converted to `(not A) or B` (material conditional)
- **Three-valued logic**: If any atomic verdict is "uncertain" (maps to `None`), the entire expression returns `None` → "uncertain"
- **Confidence computation**: Average confidence of atomic verdicts that agree with the final verdict direction
- **Fallback**: Majority vote if logic formula evaluation fails (e.g., malformed formula)

### 5.12 Verdict Normalization (`orchestrator.py: _normalize_verdict()`)

A critical fix that handles the messiness of LLM outputs:
- Maps alternative labels: "supported" → "true", "contradicted" → "false", "insufficient" → "uncertain"
- Extracts verdicts from natural language: "The claim is false" → "false"
- Normalizes confidence: if > 1.0, assumes percentage format and divides by 100
- Clamps confidence to [0.0, 1.0]
- Handles non-string verdicts gracefully (defaults to "uncertain")

---

## 6. EVIDENCE BUNDLE FORMAT (Verifier Compatibility)

Every evidence passage, whether from local retrieval or web retrieval, must conform to this format for the verifier to consume it:

```json
{
    "text": "The actual evidence passage text...",
    "title": "Source Title",
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

The **`hybrid_score`** top-level field is critical — both the verifier (line 62) and Delphi jury (line 147) read this field to weight evidence. The full score breakdown in `metadata` is for analysis purposes.

---

## 7. DATA FLOW: END-TO-END EXAMPLE

Here's exactly what happens when you run:
```bash
python scripts/run_trust_agents.py --dataset liar --split val --limit 5 --evidence-source web
```

1. **Load dataset**: Reads `data/processed/liar_val.jsonl`, takes first 5 examples.
2. **Initialize orchestrator**: Creates `TRUSTOrchestrator(evidence_source="web")`, which instantiates `WebSearchRetriever(backends=[WikipediaBackend()])`.
3. **For each example**:
   a. **Claim Extraction**: The claim extractor agent (3 LLM tool calls + 1 final LLM response) extracts factual claims.
   b. **For each extracted claim**:
      - **Web Evidence Retrieval**:
        - Generate 2 queries (cleaned claim + entity-focused)
        - For each query, search Wikipedia (direct title lookup + candidate title generation)
        - Check SQLite cache first; on miss, call Wikipedia API
        - Deduplicate documents by URL
        - Extract sentence-window passages (5 sentences, 1 overlap)
        - Score using EvidenceRanker (lexical overlap + entity match + temporal match + source quality + length penalty)
        - Select top-k via MMR diversity
      - **Verification**: Verifier agent (2-3 LLM tool calls + 1 final response) produces verdict
      - **Normalization**: Orchestrator normalizes verdict format and confidence range
      - **Explanation**: Explainer agent (4 LLM tool calls + 1 final response) produces human-readable report
   c. **Aggregate**: If multiple claims, aggregate verdicts (any false → false, else any true → true, else uncertain)
   d. **Save**: Write prediction as JSONL line to `outputs/trust_agents/liar_web_predictions.jsonl`

---

## 8. EVALUATION METHODOLOGY

### 8.1 Evaluation Script (`evaluate_predictions.py`)

**Metrics computed:**
- **Accuracy**: Standard accuracy (correct predictions / total predictions)
- **F1 Binary**: F1 score with "true" as positive class
- **F1 Macro**: Average of per-class F1 scores
- **Per-class precision, recall, F1**: For both "true" and "false" classes
- **Coverage**: Proportion of predictions that are NOT uncertain (decisive predictions / total)
- **Selective accuracy**: Accuracy computed only on non-uncertain predictions (how good the system is when it actually makes a decision)
- **Uncertain rate**: Proportion of predictions that are "uncertain"
- **Confusion matrix**: 2×2 matrix for true/false predictions vs true/false gold labels

**Uncertain handling strategies** (since gold labels are binary but predictions are 3-class):
- `uncertain → false`: Default. Treats uncertain as false.
- `uncertain → true`: Treats uncertain as true.
- `uncertain → drop`: Excludes uncertain predictions from evaluation entirely.

### 8.2 Experimental Setup

- **Fixed test set**: Stratified sample of 200 LIAR validation examples (100 true, 100 false)
- **Model**: GPT-4.1-mini for all LLM calls
- **Evidence sources compared**: Local BM25+FAISS vs. Wikipedia web retrieval
- **Baselines**: Fine-tuned BERT, fine-tuned RoBERTa, zero-shot GPT-4.1-nano

---

## 9. RESULTS

### 9.1 Baseline Models (Fine-Tuned on LIAR)

| Model | Accuracy | F1 Macro | Type |
|-------|----------|----------|------|
| BERT (fine-tuned) | **65.2%** | **0.726** | Fine-tuned classifier |
| RoBERTa (fine-tuned) | 64.1% | 0.726 | Fine-tuned classifier |
| GPT-4.1-nano (zero-shot) | 58.0% | 0.528 | LLM baseline |

### 9.2 Agent Pipeline Results (GPT-4.1-mini, 200 examples)

| Pipeline | Accuracy (unc→false) | Accuracy (unc→true) | F1 Macro (unc→false) | Uncertain Rate |
|----------|----------------------|---------------------|-----------------------|----------------|
| Original Pipeline (local evidence) | 48.5% | 52.0% | 0.428 | **70.5%** |
| Research Pipeline (LoCal + Delphi) | 49.5% | 52.0% | 0.363 | **82.5%** |

### 9.3 Prediction Distribution (Original Pipeline, 200 examples)

| Predicted Label | Count |
|-----------------|-------|
| true | 37 |
| uncertain | 141 |
| false | 22 |

### 9.4 Prediction Distribution (Research Pipeline, 200 examples)

| Predicted Label | Count |
|-----------------|-------|
| true | 9 |
| uncertain | 165 |
| false | 26 |

### 9.5 Web Evidence Retrieval (Early Results, 4 examples)

From the `liar_web_predictions.jsonl` file (only 4 examples processed with web evidence):

| Claim | Gold | Pred | Confidence | Notes |
|-------|------|------|------------|-------|
| "Less Americans working now than in the 70s" | false | uncertain | 0.63 | 2 near-duplicate claims extracted, both uncertain |
| Obama didn't use Holy Bible at swearing-in | false | **false** ✓ | 0.41 | 4 claims extracted, 1 false verdict drove overall prediction |
| Organizations parading as social welfare... | false | uncertain | 0.20 | Very low confidence, complex claim |
| Nearly half of Oregon's children are poor | true | uncertain | 0.28 | Wikipedia likely didn't have specific Oregon poverty data |

### 9.6 Web Evidence (Subset-200, 27 examples processed)

From the `liar_val_subset200_web_predictions.jsonl` file: The first 5 examples all returned "uncertain" — indicating that even with Wikipedia evidence, the verifier is still very conservative and defaults to uncertainty when evidence doesn't directly confirm/deny the claim.

---

## 10. ANALYSIS: WHY THE SYSTEM UNDERPERFORMS

### 10.1 The Uncertainty Epidemic

The #1 problem is the extremely high uncertain rate. The causes are:

1. **Conservative LLM behavior**: GPT-4.1-mini (smaller model) tends to say "uncertain" when evidence is indirect or partial. The prompt instruction to "not be overly cautious" helps but doesn't fully solve it.
2. **Evidence relevance gap**: Wikipedia articles often discuss the topic but don't directly confirm/deny the specific statistical claim. E.g., for "nearly half of Oregon's children are poor", Wikipedia might have an article about Oregon but not the specific poverty statistics.
3. **Passage granularity**: 5-sentence passage windows may not align with the exact relevant paragraph. The relevant sentence might be buried in an irrelevant passage, diluting its signal.
4. **Three-class vs binary mismatch**: The system outputs 3 classes (true/false/uncertain) but the gold labels are binary. Every "uncertain" is wrong.

### 10.2 Pipeline Error Compounding

Each step in the pipeline can introduce errors:
- **Claim extraction errors**: Extracting near-duplicate claims or missing the key claim.
- **Query generation failures**: Generated queries may not match Wikipedia article titles.
- **Search failures**: Wikipedia may not have the relevant article.
- **Passage extraction misses**: The relevant sentence may be skipped or buried.
- **Ranking errors**: Irrelevant passages may score higher than relevant ones.
- **Verification errors**: The LLM may misinterpret the evidence.

In a 6-step pipeline, if each step has 85% accuracy, the end-to-end accuracy is 0.85^6 ≈ 37.7%.

### 10.3 Fine-Tuned Models vs. Agents

Fine-tuned BERT/RoBERTa models outperform the agent pipeline because:
- They're trained specifically on the LIAR binary classification task
- They directly map input text to output label (no intermediate steps)
- They don't suffer from the uncertain problem (forced binary output)
- They leverage task-specific patterns learned from training data

### 10.4 Cost Implications

The agent pipeline makes **many LLM API calls per claim**:
- Standard pipeline: ~8-12 calls (3 for claim extraction, 2-3 for verification, 4 for explanation)
- Research pipeline: ~20-30 calls (3 for decomposition, 4 jury verifiers × N atomic claims, plus extraction and explanation)
- Each call to GPT-4.1-mini costs money; GPT-4o would cost 10-20x more

---

## 11. IMPLEMENTATION PHASES (PLANNED & COMPLETED)

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Establish baseline on 200 LIAR val examples with local evidence | ✅ Done |
| **Phase 2A** | Wikipedia retrieval (WikipediaBackend, basic scoring) | ✅ Done |
| **Phase 2B** | SQLite retrieval cache | ✅ Done |
| **Phase 2C** | Heuristic evidence ranker + MMR diversity + source quality | ✅ Done |
| **Phase 3A** | Stratified manual retrieval QA (35-45 claims, 4 scoring dimensions) | 🔲 Planned |
| **Phase 3B** | Gold evidence annotation set (30-50 claims) | 🔲 Planned |
| **Phase 4** | Full Wikipedia experiment on 200 stratified samples | ⏳ In Progress |
| **Phase 5** | Web search backend (Google/Brave/DuckDuckGo API) | 🔲 Planned |
| **Phase 6** | Cross-encoder advanced reranking | 🔲 Planned |
| **Phase 7** | Adaptive retrieval (evidence sufficiency gate) | 🔲 Planned |
| **Phase 8** | Research pipeline with web evidence | 🔲 Planned |
| **Phase 9** | Final evaluation on full LIAR val/test | 🔲 Planned |

---

## 12. KEY DESIGN DECISIONS

### 12.1 Why `hybrid_score` Instead of a New Schema

The verifier and Delphi jury both read `hybrid_score` from evidence items. Rather than modifying these unchanged agents (which would break the controlled experiment), the web retriever maps its `evidence_score` → `hybrid_score` at the top level, with the full breakdown in `metadata`. A proper schema change is deferred to Phase 8+.

### 12.2 Why Wikipedia First (Not Web Search)

- Wikipedia is free, reliable, well-structured, and doesn't require API keys
- Articles have consistent format (easy to parse and extract passages)
- High source quality (0.90 score)
- Rate limits are generous (200 req/min)
- Serves as a controlled test of external evidence before adding the noise of general web search

### 12.3 Why 4 Personas in the Delphi Jury

Each persona covers a different failure mode:
- StrictLegalist catches low-quality evidence acceptance
- OpenWebPragmatist prevents excessive conservatism
- CausalSkeptic catches temporal and logical errors
- ConspiracyDetector catches misinformation patterns

The disagreement between personas is itself informative — high disagreement suggests the claim is genuinely difficult.

### 12.4 Why LoCal-Style Decomposition

Complex claims like "After policy X passed in 2020, unemployment decreased by 5%" contain multiple verifiable facts and causal relationships. Verifying the whole claim at once is harder than:
1. Verify "Policy X passed in 2020" independently
2. Verify "Unemployment decreased by 5%" independently
3. Combine using "C1 AND C2" logic

This allows targeted evidence retrieval per atomic claim and clearer error attribution.

---

## 13. FILE STRUCTURE AND CODEBASE MAP

```
TRUST 2/
├── implementation_plan.md          # Detailed 9-phase research plan
└── news_agent/
    ├── .env                         # API keys (OPENAI_API_KEY, MODEL)
    ├── README.md                    # Project overview
    ├── HOW_TO_RUN.md               # Running instructions
    ├── QUICK_START.md              # Quick start guide
    ├── Makefile                     # Build/data preparation targets
    ├── pyproject.toml               # Python package config
    ├── requirements.txt             # Dependencies
    │
    ├── TRUST_agents/                # Core agent code
    │   ├── __init__.py
    │   ├── orchestrator.py          # Standard pipeline orchestrator
    │   ├── orchestrator_research.py # Research pipeline orchestrator
    │   ├── utils.py                 # Shared utilities
    │   │
    │   ├── agents/                  # Standard pipeline agents
    │   │   ├── claim_extractor.py       # ReAct claim extraction agent
    │   │   ├── claim_extractor_tools.py # NER, dep parse, LLM tools
    │   │   ├── evidence_retrieval.py    # ReAct evidence retrieval agent
    │   │   ├── retrieval_agent_core.py  # BM25 + FAISS retrieval engine
    │   │   ├── retrieval_agent_tools.py # Search, index, get passage tools
    │   │   ├── verifier.py              # ReAct verification agent
    │   │   ├── verifier_tools.py        # Compare, aggregate, verdict tools
    │   │   ├── explainer.py             # ReAct explanation agent
    │   │   ├── explainer_tools.py       # Summary, explain, cite, report tools
    │   │   │
    │   │   ├── web_search_retriever.py  # [NEW v2] Wikipedia/web retriever
    │   │   ├── evidence_ranker.py       # [NEW v2] Heuristic scorer + MMR
    │   │   ├── source_quality.py        # [NEW v2] URL source classifier
    │   │   └── retrieval_cache.py       # [NEW v2] SQLite search cache
    │   │
    │   └── agents2.0/              # Research pipeline agents
    │       ├── README.md
    │       ├── __init__.py
    │       ├── decomposer_agent.py      # LoCal-style claim decomposition
    │       ├── delphi_jury.py           # Multi-agent jury verification
    │       └── logic_aggregator.py      # Logical formula aggregation
    │
    ├── scripts/                     # Runner and utility scripts
    │   ├── run_trust_agents.py          # Run standard pipeline on dataset
    │   ├── run_trust_research.py        # Run research pipeline on dataset
    │   ├── evaluate_predictions.py      # Compute evaluation metrics
    │   ├── test_web_retrieval.py        # Smoke test for web retriever
    │   ├── prepare_evidence_corpus.py   # Build local evidence corpus (FLAWED)
    │   ├── index_evidence.py            # Index corpus for BM25+FAISS
    │   ├── compare_all_results.py       # Compare all experiment results
    │   ├── compare_all_metrics.py       # Compare all metrics
    │   ├── download_liar.py             # Download LIAR dataset
    │   ├── prepare_liar.py              # Process LIAR into JSONL
    │   ├── baseline_llm.py              # Zero-shot LLM baseline
    │   ├── train_hf.py                  # Fine-tune BERT/RoBERTa
    │   ├── eval_hf.py                   # Evaluate fine-tuned models
    │   └── ...
    │
    ├── data/
    │   ├── raw/                     # Raw dataset downloads
    │   ├── processed/               # Processed JSONL datasets
    │   └── evidence_corpus/         # Local evidence corpus (label-dependent)
    │
    ├── outputs/
    │   ├── trust_agents/            # Standard pipeline predictions
    │   │   ├── liar_predictions.jsonl             # Local evidence, 200 examples
    │   │   ├── liar_web_predictions.jsonl          # Web evidence, 4 examples
    │   │   ├── liar_val_subset200_web_predictions.jsonl  # Web evidence, 27 examples
    │   │   └── metrics_liar.json                  # Evaluation metrics
    │   └── trust_research/          # Research pipeline predictions
    │       └── liar_predictions.jsonl             # LoCal+Delphi, 200 examples
    │
    ├── results/
    │   ├── baselines_liar.json      # BERT/RoBERTa/LLM baseline results
    │   └── comparison_all.json      # All methods compared
    │
    └── retrieval_index/             # FAISS index + cache storage
        └── retrieval_cache.db       # SQLite cache for web search results
```

---

## 14. TECHNOLOGIES AND DEPENDENCIES

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | OpenAI GPT-4.1-mini | All agent reasoning |
| Agent framework | LangGraph + LangChain | ReAct agent implementation |
| Sparse retrieval | rank_bm25 (BM25Okapi) | Keyword-based search |
| Dense retrieval | sentence-transformers + FAISS | Semantic similarity search |
| Wikipedia API | wikipedia-api (Python) | External evidence retrieval |
| NLP | spaCy (en_core_web_sm) | NER and dependency parsing |
| Tokenization | NLTK (punkt_tab) | Sentence boundary detection |
| Retry logic | tenacity | Exponential backoff for API calls |
| PDF parsing | PyPDF2 | Document ingestion |
| HTML parsing | BeautifulSoup4 | Document ingestion |
| ML baselines | Hugging Face Transformers | BERT/RoBERTa fine-tuning |
| Evaluation | scikit-learn | Accuracy, F1, confusion matrix |
| Cache | SQLite3 (stdlib) | Search result caching |
| Data science | numpy | Trust score computation |
| Environment | python-dotenv | API key management |
| Progress | tqdm | Progress bars |

---

## 15. REFERENCES AND INSPIRATION

1. **LoCal** (ACM 2024): Logical and Causal Fact-Checking — inspired the claim decomposition + logic aggregation approach
2. **Delphi Method**: Classical expert consensus technique — inspired the multi-agent jury system
3. **Cleanlab BOLAA**: Trust scoring for language model agents — inspired the trust-weighted voting
4. **TRUST-VL / TRUST-Instruct** (ACL 2024): News assistance with LLMs — the original TRUST framework
5. **LIAR Dataset**: Wang 2017, benchmark for fake news detection
6. **MEGA-RAG**: Evidence refinement techniques (planned for future phases)
7. **MMR (Maximal Marginal Relevance)**: Carbonell & Goldstein 1998 — used for diversity in evidence selection

---

## 16. FAILURE TAXONOMY (PLANNED)

The implementation plan defines 8 failure types for systematic error analysis:

| Code | Failure Type | Description |
|------|-------------|-------------|
| F1 | Wrong claim extraction | Claim extractor misidentifies or misses factual claims |
| F2 | Search failure | No search results returned at all |
| F3 | Missing evidence | Results found but none are relevant to the claim |
| F4 | Irrelevant evidence selected | Relevant evidence exists but wrong passages selected |
| F5 | Conflicting evidence | Evidence supports both true and false interpretations |
| F6 | Incorrect verifier reasoning | Evidence is good but verifier reaches wrong conclusion |
| F7 | Incorrect uncertainty handling | System defaults to uncertain when evidence is sufficient |
| F8 | Logic aggregation error | Atomic verdicts correct but combination logic fails |

---

## 17. WHAT THE PLANNED ABLATION TABLE LOOKS LIKE

| Version | Retrieval Method | Accuracy | Macro-F1 | Coverage | Cost/Sample | What It Tests |
|---------|-----------------|----------|----------|----------|-------------|---------------|
| Baseline | Local BM25 + FAISS | 48.5% | 0.428 | 29.5% | — | Current performance |
| V1 | Wikipedia search (basic top-k) | — | — | — | — | Does external evidence alone help? |
| V2 | Wikipedia + heuristic ranking | — | — | — | — | Does evidence quality selection matter? |
| V3 | Wikipedia + ranking + source quality | — | — | — | — | Does source awareness matter? |
| V4 | Web search + reranking | — | — | — | — | Does broader web help? |
| V5 | Research pipeline (LoCal + Delphi + web) | — | — | — | — | Does multi-agent architecture add value? |

---

## 18. SUMMARY

This project is a **research internship investigation** into whether multi-agent LLM architectures can perform automated fact-checking. The key contributions are:

1. **Identified a methodological flaw** in the original evidence corpus (label-dependent evidence).
2. **Built a Wikipedia-based external evidence retrieval system** with caching, heuristic ranking, MMR diversity, and source quality classification.
3. **Implemented a research-grade pipeline** with LoCal-style claim decomposition, Delphi multi-agent jury verification, and logical aggregation.
4. **Established baselines** (fine-tuned BERT at 65.2%, RoBERTa at 64.1%, zero-shot GPT at 58%).
5. **Discovered the uncertainty problem**: Agent pipelines produce 70-82% uncertain predictions, devastating accuracy when evaluated against binary gold labels.
6. **Drew practical conclusions**: Sophisticated agent architectures require capable models to work; fine-tuned classifiers remain superior for structured classification tasks; the cost-accuracy tradeoff is significant.

The project is ongoing, with Phases 3-9 (manual QA, gold evidence annotation, web search backend, cross-encoder reranking, adaptive retrieval, and full evaluation) still planned.
