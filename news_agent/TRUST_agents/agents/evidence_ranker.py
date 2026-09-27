# -*- coding: utf-8 -*-

"""
Evidence Ranker — Heuristic scoring + MMR diversity selection (Phase 2C).

Scoring dimensions:
  - semantic_sim   (word-overlap proxy until Phase 6 cross-encoder)
  - lexical_overlap
  - entity_match
  - temporal_match (date/number proximity)
  - source_quality (from source_quality.py)

MMR diversity selection adapted from retrieval_agent_core.mmr_rerank().
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

from TRUST_agents.agents.source_quality import classify_source

logger = logging.getLogger("TRUST_agents.evidence_ranker")


# Default scoring weights (should sum to ~1.0)
DEFAULT_WEIGHTS = {
    "lexical_overlap":  0.30,
    "entity_match":     0.30,
    "temporal_match":   0.10,
    "source_quality":   0.20,
    "length_penalty":   0.10,
}


class EvidenceRanker:
    """
    Heuristic evidence ranker with configurable weights and MMR diversity.

    Usage::

        ranker = EvidenceRanker()
        scored = ranker.score(claim, passages)
        selected = ranker.select(scored, top_k=5, diversity=0.7)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    def score(
        self,
        claim: str,
        passages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Score each passage dict in-place and return the list.

        Expects passage dicts with at minimum: ``text``, ``url``.
        Adds/updates: ``hybrid_score``, ``metadata.ranking_scores``.
        """
        claim_tokens = set(re.findall(r"\w+", claim.lower()))
        claim_proper = set(re.findall(r"[A-Z][a-z]+", claim))
        claim_numbers = set(re.findall(r"\d+", claim))

        for p in passages:
            text = p.get("text", "")
            url = p.get("url", "")
            p_tokens = set(re.findall(r"\w+", text.lower()))

            # --- Lexical overlap ---
            lexical = (
                len(claim_tokens & p_tokens) / len(claim_tokens)
                if claim_tokens
                else 0.0
            )

            # --- Entity match ---
            p_proper = set(re.findall(r"[A-Z][a-z]+", text))
            entity = (
                len(claim_proper & p_proper) / len(claim_proper)
                if claim_proper
                else 0.0
            )

            # --- Temporal / numeric match ---
            p_numbers = set(re.findall(r"\d+", text))
            temporal = (
                len(claim_numbers & p_numbers) / len(claim_numbers)
                if claim_numbers
                else 0.0
            )

            # --- Source quality ---
            source_type, sq = classify_source(url)
            p.setdefault("source_type", source_type)

            # --- Length penalty ---
            word_count = len(p_tokens)
            if word_count < 10:
                length_score = 0.3
            elif word_count > 400:
                length_score = 0.6
            else:
                # Sweet spot: 30-150 words
                length_score = 1.0 if 30 <= word_count <= 150 else 0.8

            # --- Weighted sum ---
            w = self.weights
            score = (
                w["lexical_overlap"] * lexical
                + w["entity_match"] * entity
                + w["temporal_match"] * temporal
                + w["source_quality"] * sq
                + w["length_penalty"] * length_score
            )

            p["hybrid_score"] = round(score, 4)

            # Store breakdown in metadata
            ranking_scores = {
                "lexical_overlap": round(lexical, 4),
                "entity_match": round(entity, 4),
                "temporal_match": round(temporal, 4),
                "source_quality": round(sq, 4),
                "source_type": source_type,
                "length_score": round(length_score, 4),
                "evidence_score": round(score, 4),
                "relevance_score": round(score, 4),
            }
            p.setdefault("metadata", {})
            p["metadata"]["ranking_scores"] = ranking_scores

        return passages

    def select(
        self,
        passages: List[Dict[str, Any]],
        top_k: int = 5,
        diversity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Select top-k passages using MMR-style diversity.

        With diversity=1.0, purely greedy by score.
        With diversity=0.0, maximises dissimilarity from already-selected passages.
        """
        if not passages:
            return []

        # Sort by score
        ranked = sorted(passages, key=lambda p: p.get("hybrid_score", 0), reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_token_sets: List[set] = []

        for candidate in ranked:
            if len(selected) >= top_k:
                break

            c_tokens = set(re.findall(r"\w+", candidate.get("text", "").lower()))

            if not selected_token_sets:
                # First selection — just pick highest score
                selected.append(candidate)
                selected_token_sets.append(c_tokens)
                continue

            # MMR: balance relevance vs. redundancy
            relevance = candidate.get("hybrid_score", 0)

            # Max similarity to any already-selected passage
            max_sim = 0.0
            for prev_tokens in selected_token_sets:
                if c_tokens and prev_tokens:
                    jaccard = len(c_tokens & prev_tokens) / len(c_tokens | prev_tokens)
                    max_sim = max(max_sim, jaccard)

            mmr_score = diversity * relevance - (1 - diversity) * max_sim

            # Accept if MMR score is positive or we have room
            if mmr_score > -0.1 or len(selected) < min(3, top_k):
                selected.append(candidate)
                selected_token_sets.append(c_tokens)

        return selected
