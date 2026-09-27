#!/usr/bin/env python3
"""
Smoke test for the web search retriever.

Tests the WebSearchRetriever with a few known claims to verify:
1. Wikipedia articles are found
2. Passages are extracted and scored
3. Evidence bundles have the correct format (hybrid_score, etc.)
4. Cache works on second call

Usage:
    python scripts/test_web_retrieval.py
    python scripts/test_web_retrieval.py --claims 5
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# Suppress noisy loggers
for noisy in ["httpcore", "httpx", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Known claims for testing (mix of true and false)
TEST_CLAIMS = [
    {
        "claim": "Barack Obama was the 44th president of the United States.",
        "expected_label": "true",
        "should_find": ["Obama", "president"],
    },
    {
        "claim": "The Great Wall of China is visible from space with the naked eye.",
        "expected_label": "false",
        "should_find": ["Great Wall", "China"],
    },
    {
        "claim": "Water boils at 100 degrees Celsius at sea level.",
        "expected_label": "true",
        "should_find": ["boil", "100"],
    },
    {
        "claim": "The Earth is approximately 4.5 billion years old.",
        "expected_label": "true",
        "should_find": ["Earth", "billion", "years"],
    },
    {
        "claim": "Albert Einstein failed mathematics in school.",
        "expected_label": "false",
        "should_find": ["Einstein"],
    },
]


def test_retriever(n_claims: int = 3, top_k: int = 5):
    """Run smoke test with n known claims."""
    from TRUST_agents.agents.web_search_retriever import WebSearchRetriever

    retriever = WebSearchRetriever()

    results = []
    claims_to_test = TEST_CLAIMS[:n_claims]

    for i, test_case in enumerate(claims_to_test, 1):
        claim = test_case["claim"]
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(claims_to_test)}: {claim}")
        print(f"{'='*70}")

        evidence = retriever.retrieve(claim, top_k=top_k)

        print(f"\n  Found {len(evidence)} evidence passages:")
        for j, ev in enumerate(evidence[:3], 1):
            score = ev.get("hybrid_score", 0)
            title = ev.get("title", "?")
            text_preview = ev.get("text", "")[:120]
            print(f"  [{j}] score={score:.4f} | {title}")
            print(f"      {text_preview}...")

        # Validate format
        format_ok = True
        if evidence:
            ev0 = evidence[0]
            required_keys = ["text", "title", "source", "url", "hybrid_score"]
            missing = [k for k in required_keys if k not in ev0]
            if missing:
                print(f"\n  [!] Missing keys: {missing}")
                format_ok = False
            else:
                print(f"\n  [OK] Evidence format OK")

            # Check for expected terms
            all_text = " ".join(e.get("text", "") for e in evidence).lower()
            for term in test_case.get("should_find", []):
                if term.lower() in all_text:
                    print(f"  [OK] Found expected term: '{term}'")
                else:
                    print(f"  [X] Missing expected term: '{term}'")
        else:
            print(f"\n  [!] No evidence retrieved!")
            format_ok = False

        results.append({
            "claim": claim,
            "n_evidence": len(evidence),
            "format_ok": format_ok,
            "top_score": evidence[0]["hybrid_score"] if evidence else 0.0,
        })

    # Test cache hit on second call
    print(f"\n{'='*70}")
    print("Testing cache (second call for first claim)...")
    print(f"{'='*70}")
    evidence2 = retriever.retrieve(claims_to_test[0]["claim"], top_k=top_k)
    print(f"  Second call returned {len(evidence2)} passages")

    if retriever._cache is not None:
        stats = retriever._cache.stats()
        print(f"  Cache stats: {stats}")
        if stats["hits"] > 0:
            print(f"  [OK] Cache hits working!")
        else:
            print(f"  [!] No cache hits detected")

    # Summary
    print(f"\n{'='*70}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "[OK]" if r["format_ok"] and r["n_evidence"] > 0 else "[X]"
        print(f"  {status} {r['claim'][:60]}... -> {r['n_evidence']} passages, top={r['top_score']:.4f}")

    passed = sum(1 for r in results if r["format_ok"] and r["n_evidence"] > 0)
    print(f"\n  {passed}/{len(results)} tests passed")

    return passed == len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test web retriever")
    parser.add_argument("--claims", type=int, default=3, help="Number of claims to test")
    parser.add_argument("--top-k", type=int, default=5, help="Evidence passages per claim")
    args = parser.parse_args()

    success = test_retriever(n_claims=args.claims, top_k=args.top_k)
    sys.exit(0 if success else 1)
