# Evidence Grounded Fact Checker (TRUST Agents v2)

A research-driven, multi-agent automated fact-checking system designed to replace label-dependent evidence retrieval with externally grounded, source-verified evidence.

## Overview

Traditional automated fact-checking pipelines often rely on internal training labels or synthetic corpora to retrieve evidence, creating methodological circularity. The **Evidence Grounded Fact Checker** implements an end-to-end framework that:
1. Decomposes complex real-world claims into atomic claims and logical dependencies.
2. Retrieves external, source-grounded evidence from live knowledge sources (e.g., Wikipedia) via a hybrid retrieval pipeline.
3. Ranks and evaluates sources based on domain credibility and semantic relevance.
4. Uses a multi-agent Delphi Jury to conduct nuanced, confidence-weighted verification.
5. Employs logic aggregation to resolve multi-hop, compound claims.

---

## Repository Structure

```
├── implementation_plan.md              # Detailed implementation plan & research design
├── trust_agents_v2_full_explanation.md # In-depth technical breakdown and methodology
├── news_agent/                         # Core codebase
│   ├── TRUST_agents/                   # Multi-agent framework
│   │   ├── agents/                     # Baseline agents (extraction, ranking, verification, explainer)
│   │   └── agents2.0/                  # Advanced agents (decomposer, delphi jury, logic aggregator)
│   ├── scripts/                        # Evaluation, testing, and indexing scripts
│   ├── data/                           # Evaluation & benchmark datasets (LIAR, FNN)
│   ├── pyproject.toml                  # Project packaging configuration
│   ├── requirements.txt                # Python dependencies
│   ├── Makefile                        # Common tasks and commands
│   ├── .env.example                    # Environment variable template
│   └── README.md                       # news_agent package documentation
└── .gitignore                          # Repository ignore patterns
```

---

## Architecture

```
                     CLAIM
                       │
                       ▼
                Claim Extraction
                 (LLM / ReAct)
                       │
                       ▼
               Claim Decomposition
                 (Atomic claims)
                       │
                       ▼
           External Evidence Retrieval
          (Live Wikipedia / Search API)
                       │
                       ▼
           Source Quality & Evidence Ranking
            (Credibility + Semantic scores)
                       │
                       ▼
                 Delphi Jury
         (Multi-persona verification)
                       │
                       ▼
               Logic Aggregation
            (Truth-table evaluation)
                       │
                       ▼
            VERDICT & EXPLANATION
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key

### 2. Installation

```bash
cd news_agent
pip install -r requirements.txt
pip install -e .
```

### 3. Environment Setup

Copy `.env.example` to `.env` and fill in your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Running the System

To run the fact-checking agent on claims:

```bash
python scripts/run_trust_agents.py
```

To run the research pipeline with advanced decomposition and jury verification:

```bash
python scripts/run_trust_research.py
```

To test external web search retrieval:

```bash
python scripts/test_web_retrieval.py
```

---

## Documentation

- [Implementation Plan](implementation_plan.md)
- [TRUST Agents v2 Full Technical Explanation](trust_agents_v2_full_explanation.md)
- [News Agent Documentation](news_agent/README.md)
- [How to Run Guide](news_agent/HOW_TO_RUN.md)
