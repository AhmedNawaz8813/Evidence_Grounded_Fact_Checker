# -*- coding: utf-8 -*-

"""
Web Search Retriever — External evidence retrieval for TRUST Agents v2.

Replaces the label-dependent local evidence corpus with externally retrieved,
source-grounded evidence.  Phase 2A: Wikipedia only.

Pipeline:
    claim → [fast-path check] → query_gen → search → fetch → passage_split
          → rank (basic top-k) → deduplicate → evidence bundle

Evidence bundles carry a top-level `hybrid_score` for verifier compatibility
(see verifier.py L62, delphi_jury.py L147).  Full score breakdown is in
`metadata`.
"""

from __future__ import annotations

import hashlib
import logging
import re
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    import wikipediaapi
except ImportError:
    wikipediaapi = None

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("TRUST_agents.web_search_retriever")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvidencePassage:
    """A single evidence passage with provenance and scoring."""
    text: str
    title: str
    source: str                       # e.g. "en.wikipedia.org"
    url: str
    source_type: str                  # e.g. "wikipedia"
    passage_id: str
    hybrid_score: float               # verifier-compatible top-level score
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Search backend ABC
# ---------------------------------------------------------------------------

class SearchBackend(ABC):
    """Abstract base for search backends (Wikipedia, web search, etc.)."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Return a list of raw search results.

        Each result dict must contain at minimum:
            - title: str
            - url: str
            - content: str  (full document text)
            - source: str   (domain)
            - source_type: str
        """
        ...


# ---------------------------------------------------------------------------
# Wikipedia backend (Phase 2A)
# ---------------------------------------------------------------------------

class WikipediaBackend(SearchBackend):
    """
    Retrieves Wikipedia articles via the ``wikipedia-api`` package.

    Rate-limited to ≤ 200 req/min via tenacity exponential backoff.
    """

    def __init__(self, language: str = "en", user_agent: str = "TRUST-Agents/2.0"):
        if wikipediaapi is None:
            raise ImportError(
                "wikipedia-api is required for WikipediaBackend. "
                "Install with: pip install wikipedia-api"
            )
        self._wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

    # Retry on transient network errors
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.3, max=5),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    def _fetch_page(self, title: str) -> Optional[wikipediaapi.WikipediaPage]:
        page = self._wiki.page(title)
        if page.exists():
            return page
        return None

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for articles matching *query*.

        Uses the MediaWiki opensearch API (via wikipedia-api title lookup)
        and falls back to a naive title search with the query terms.
        """
        results: List[Dict[str, Any]] = []
        seen_titles: set = set()

        # --- Strategy 1: direct title lookup ---
        page = self._fetch_page(query)
        if page is not None and page.title.lower() not in seen_titles:
            seen_titles.add(page.title.lower())
            results.append(self._page_to_result(page))

        # --- Strategy 2: search via opensearch-style lookup ---
        # wikipedia-api doesn't expose search directly, so we try
        # key noun-phrases from the query as titles.
        candidate_titles = self._generate_candidate_titles(query)
        for title in candidate_titles:
            if len(results) >= max_results:
                break
            if title.lower() in seen_titles:
                continue
            page = self._fetch_page(title)
            if page is not None:
                seen_titles.add(page.title.lower())
                results.append(self._page_to_result(page))

        logger.info(
            "[WikipediaBackend] query=%r → %d articles found",
            query[:80], len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _page_to_result(page: "wikipediaapi.WikipediaPage") -> Dict[str, Any]:
        return {
            "title": page.title,
            "url": page.fullurl,
            "content": page.text,
            "source": "en.wikipedia.org",
            "source_type": "wikipedia",
        }

    @staticmethod
    def _generate_candidate_titles(query: str) -> List[str]:
        """
        Heuristic title candidates derived from the query text.

        Extracts capitalised noun-phrases and proper nouns that are likely
        Wikipedia article titles.
        """
        # Remove common stopwords / filler
        stopwords = {
            "the", "a", "an", "is", "was", "were", "are", "been", "be",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "need",
            "dare", "ought", "used", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "out", "off", "over",
            "under", "again", "further", "then", "once", "that", "this",
            "these", "those", "than", "said", "says", "say", "claim",
            "claims", "according", "about", "not", "no", "nor", "but",
            "or", "and", "so", "if", "when", "while", "because", "since",
            "it", "its", "he", "she", "they", "we", "you", "i", "my",
            "his", "her", "their", "our", "your", "more", "most", "less",
            "very", "just", "also", "now", "still", "already", "ever",
        }

        candidates: List[str] = []

        # 1) Extract capitalised multi-word phrases (likely proper nouns)
        proper_phrases = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", query)
        candidates.extend(proper_phrases)

        # 2) Extract individual capitalised words (skip sentence starts)
        words = query.split()
        for idx, w in enumerate(words):
            clean = re.sub(r"[^A-Za-z0-9]", "", w)
            if not clean:
                continue
            if clean[0].isupper() and idx > 0 and clean.lower() not in stopwords:
                candidates.append(clean)

        # 3) Full query as a title (sometimes works for short claims)
        # Clean punctuation
        clean_query = re.sub(r"[^\w\s]", "", query).strip()
        if len(clean_query.split()) <= 6:
            candidates.append(clean_query)

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for c in candidates:
            key = c.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(c.strip())

        return unique[:10]  # cap to avoid excessive lookups


# ---------------------------------------------------------------------------
# Passage extraction
# ---------------------------------------------------------------------------

def _split_into_passages(
    text: str,
    title: str,
    max_sentences: int = 5,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Split *text* into overlapping sentence-window passages.

    Returns a list of passage strings.  Empty / very short texts are returned
    as-is (single passage).
    """
    if not text or len(text.strip()) < 30:
        return [text.strip()] if text and text.strip() else []

    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return [text.strip()]

    passages: List[str] = []
    step = max(1, max_sentences - overlap_sentences)
    for start in range(0, len(sentences), step):
        window = sentences[start : start + max_sentences]
        passage = " ".join(window)
        if passage.strip():
            passages.append(passage.strip())

    return passages


# ---------------------------------------------------------------------------
# LIAR fast-path detection
# ---------------------------------------------------------------------------

def _is_single_proposition(text: str) -> bool:
    """
    Heuristic: return True if *text* is a single short factual statement
    (typical LIAR format) that does not need claim extraction.

    Detection: ≤ 2 sentences and no conjunction-linked independent clauses.
    """
    sentences = sent_tokenize(text.strip())
    if len(sentences) > 2:
        return False
    # Check for coordinating conjunctions linking clauses
    multi_clause_patterns = [
        r"\b(and|but|or|yet|so)\b.*\b(is|was|are|were|has|have|had|does|did|will|would)\b",
    ]
    for pat in multi_clause_patterns:
        if re.search(pat, text, re.IGNORECASE):
            # Only flag if there are multiple verb phrases
            verbs = re.findall(
                r"\b(is|was|are|were|has|have|had|does|did|will|would|can|could|should|may|might)\b",
                text, re.IGNORECASE,
            )
            if len(verbs) >= 3:
                return False
    return True


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def _generate_queries(claim: str) -> List[str]:
    """
    Generate 2 search queries for a claim:
      1. The original claim (lightly cleaned).
      2. An entity-focused reformulation.
    """
    queries: List[str] = []

    # Query 1: cleaned claim
    clean = re.sub(r"[\"']", "", claim).strip()
    # Remove trailing attribution like "-- John Doe" or "- CNN"
    clean = re.sub(r"\s*[-–—]\s*[A-Z][A-Za-z\s,]+$", "", clean).strip()
    queries.append(clean)

    # Query 2: entity-focused (extract proper nouns + key nouns)
    proper_nouns = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", claim)
    # Also capture numbers with context
    numbers_ctx = re.findall(r"\d[\d,\.]*\s*(?:percent|%|million|billion|trillion|thousand)?", claim, re.IGNORECASE)

    entity_query_parts = list(dict.fromkeys(proper_nouns))  # dedup, keep order
    if numbers_ctx:
        entity_query_parts.extend(numbers_ctx[:2])

    if entity_query_parts:
        entity_query = " ".join(entity_query_parts[:5])
        if entity_query.lower().strip() != clean.lower().strip():
            queries.append(entity_query)

    return queries


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class WebSearchRetriever:
    """
    External evidence retriever for fact-checking claims.

    Supports:
      - Phase 2A: Wikipedia-only backend with basic scoring
      - Phase 2B: SQLite cache for reproducible re-runs
      - Phase 2C: Heuristic evidence ranker + MMR diversity
    """

    def __init__(
        self,
        backends: Optional[List[SearchBackend]] = None,
        max_passages_per_doc: int = 20,
        passage_max_sentences: int = 5,
        passage_overlap: int = 1,
        use_cache: bool = True,
        cache_db_path: str = "retrieval_index/retrieval_cache.db",
        use_ranker: bool = True,
    ):
        if backends is None:
            backends = [WikipediaBackend()]
        self.backends = backends
        self.max_passages_per_doc = max_passages_per_doc
        self.passage_max_sentences = passage_max_sentences
        self.passage_overlap = passage_overlap

        # Phase 2B: Cache
        self._cache = None
        if use_cache:
            try:
                from TRUST_agents.agents.retrieval_cache import RetrievalCache
                self._cache = RetrievalCache(db_path=cache_db_path)
            except Exception as e:
                logger.warning("[WebSearchRetriever] Cache init failed: %s", e)

        # Phase 2C: Evidence ranker
        self._ranker = None
        if use_ranker:
            try:
                from TRUST_agents.agents.evidence_ranker import EvidenceRanker
                self._ranker = EvidenceRanker()
            except Exception as e:
                logger.warning("[WebSearchRetriever] Ranker init failed: %s", e)

    def retrieve(self, claim: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve evidence for *claim*.

        Returns a list of evidence bundle dicts compatible with the existing
        verifier (top-level ``hybrid_score``).

        Args:
            claim: The factual claim to find evidence for.
            top_k: Maximum number of evidence passages to return.

        Returns:
            List[Dict] — each dict follows the EvidencePassage schema.
        """
        logger.info("[WebSearchRetriever] Retrieving evidence for: %s", claim[:100])

        # Step 1: LIAR fast-path — skip claim extraction for single propositions
        is_single = _is_single_proposition(claim)
        if is_single:
            logger.debug("[WebSearchRetriever] Single-proposition fast-path activated")

        # Step 2: Generate queries
        queries = _generate_queries(claim)
        logger.info("[WebSearchRetriever] Generated %d queries: %s", len(queries), queries)

        # Step 3: Search all backends (with cache)
        raw_results: List[Dict[str, Any]] = []
        for backend in self.backends:
            backend_name = type(backend).__name__
            for query in queries:
                # Check cache first
                if self._cache is not None:
                    cache_key = self._cache.make_key(query, backend_name)
                    cached = self._cache.get(cache_key)
                    if cached is not None:
                        raw_results.extend(cached)
                        continue

                try:
                    results = backend.search(query, max_results=3)
                    raw_results.extend(results)
                    # Store in cache
                    if self._cache is not None:
                        self._cache.put(
                            cache_key, results,
                            query=query, backend=backend_name,
                        )
                except Exception as e:
                    logger.warning(
                        "[WebSearchRetriever] Backend %s failed for query %r: %s",
                        backend_name, query[:60], e,
                    )

        if self._cache is not None:
            stats = self._cache.stats()
            logger.info(
                "[WebSearchRetriever] Cache stats: hits=%d, misses=%d",
                stats["hits"], stats["misses"],
            )

        if not raw_results:
            logger.warning("[WebSearchRetriever] No search results found")
            return []

        # Step 4: Deduplicate documents by URL
        seen_urls: set = set()
        unique_docs: List[Dict[str, Any]] = []
        for doc in raw_results:
            url = doc.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_docs.append(doc)

        logger.info("[WebSearchRetriever] %d unique documents after dedup", len(unique_docs))

        # Step 5: Extract passages from each document
        all_passages: List[EvidencePassage] = []
        for doc in unique_docs:
            content = doc.get("content", "")
            title = doc.get("title", "Unknown")
            if not content:
                continue

            passages = _split_into_passages(
                content, title,
                max_sentences=self.passage_max_sentences,
                overlap_sentences=self.passage_overlap,
            )

            for idx, ptext in enumerate(passages[: self.max_passages_per_doc]):
                pid = _make_passage_id(doc.get("source_type", "unknown"), title, idx)
                ep = EvidencePassage(
                    text=ptext,
                    title=title,
                    source=doc.get("source", "unknown"),
                    url=doc.get("url", ""),
                    source_type=doc.get("source_type", "unknown"),
                    passage_id=pid,
                    hybrid_score=0.0,  # will be set after scoring
                    metadata={
                        "query_used": queries[0] if queries else claim,
                        "doc_index": unique_docs.index(doc),
                        "passage_index": idx,
                    },
                )
                all_passages.append(ep)

        if not all_passages:
            logger.warning("[WebSearchRetriever] No passages extracted from documents")
            return []

        logger.info("[WebSearchRetriever] Extracted %d passages total", len(all_passages))

        # Step 6: Score and select
        if self._ranker is not None:
            # Phase 2C: use heuristic ranker + MMR diversity
            passage_dicts = [ep.to_dict() for ep in all_passages]
            scored = self._ranker.score(claim, passage_dicts)
            selected = self._ranker.select(scored, top_k=top_k, diversity=0.7)
            logger.info(
                "[WebSearchRetriever] Ranker selected %d passages (top_k=%d)",
                len(selected), top_k,
            )
            return selected
        else:
            # Phase 2A fallback: basic lexical scoring
            scored = _basic_relevance_score(claim, all_passages)
            selected = _select_top_k(scored, top_k)
            logger.info(
                "[WebSearchRetriever] Basic scoring selected %d passages (top_k=%d)",
                len(selected), top_k,
            )
            return [ep.to_dict() for ep in selected]


# ---------------------------------------------------------------------------
# Basic relevance scoring (Phase 2A — no embeddings)
# ---------------------------------------------------------------------------

def _basic_relevance_score(
    claim: str,
    passages: List[EvidencePassage],
) -> List[EvidencePassage]:
    """
    Score passages by lexical overlap with the claim.

    This is a simple word-overlap heuristic for Phase 2A.
    Phase 2C will replace this with the full ``EvidenceRanker``.
    """
    claim_tokens = set(re.findall(r"\w+", claim.lower()))
    if not claim_tokens:
        return passages

    for ep in passages:
        passage_tokens = set(re.findall(r"\w+", ep.text.lower()))
        if not passage_tokens:
            ep.hybrid_score = 0.0
            continue

        # Jaccard-like overlap
        intersection = claim_tokens & passage_tokens
        overlap = len(intersection) / len(claim_tokens) if claim_tokens else 0.0

        # Bonus for proper noun overlap (entity matching)
        claim_proper = set(re.findall(r"[A-Z][a-z]+", claim))
        passage_proper = set(re.findall(r"[A-Z][a-z]+", ep.text))
        entity_overlap = (
            len(claim_proper & passage_proper) / len(claim_proper)
            if claim_proper
            else 0.0
        )

        # Combined score
        score = 0.6 * overlap + 0.4 * entity_overlap

        # Length penalty — very short or very long passages are less useful
        word_count = len(passage_tokens)
        if word_count < 10:
            score *= 0.5
        elif word_count > 300:
            score *= 0.8

        ep.hybrid_score = round(score, 4)
        ep.metadata["retrieval_score"] = round(overlap, 4)
        ep.metadata["entity_overlap"] = round(entity_overlap, 4)
        ep.metadata["relevance_score"] = round(score, 4)
        ep.metadata["evidence_score"] = round(score, 4)
        ep.metadata["source_quality"] = 0.90 if ep.source_type == "wikipedia" else 0.50

    return passages


def _select_top_k(
    passages: List[EvidencePassage],
    top_k: int,
    dedup_threshold: float = 0.85,
) -> List[EvidencePassage]:
    """
    Select top-k passages, removing near-duplicates.

    Near-duplicate detection uses word-set Jaccard similarity.
    """
    # Sort by score descending
    ranked = sorted(passages, key=lambda p: p.hybrid_score, reverse=True)

    selected: List[EvidencePassage] = []
    selected_token_sets: List[set] = []

    for ep in ranked:
        if len(selected) >= top_k:
            break

        tokens = set(re.findall(r"\w+", ep.text.lower()))

        # Check for near-duplicate
        is_dup = False
        for prev_tokens in selected_token_sets:
            if not tokens or not prev_tokens:
                continue
            jaccard = len(tokens & prev_tokens) / len(tokens | prev_tokens)
            if jaccard >= dedup_threshold:
                is_dup = True
                break

        if not is_dup:
            selected.append(ep)
            selected_token_sets.append(tokens)

    return selected


def _make_passage_id(source_type: str, title: str, index: int) -> str:
    """Generate a stable passage ID."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title)[:40]
    return f"{source_type}_{slug}_{index}"
