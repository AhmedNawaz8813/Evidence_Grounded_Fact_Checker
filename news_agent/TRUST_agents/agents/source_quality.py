# -*- coding: utf-8 -*-

"""
Source Quality Classifier — Phase 2C.

Classifies sources by type and assigns fixed quality scores.
"""

from __future__ import annotations

import re
from typing import Tuple
from urllib.parse import urlparse

# Source types and their default quality scores (0–1)
SOURCE_QUALITY_SCORES = {
    "wikipedia":    0.90,
    "government":   0.92,
    "academic":     0.93,
    "news_major":   0.80,
    "news_minor":   0.60,
    "factcheck":    0.88,
    "blog":         0.35,
    "social_media": 0.20,
    "unknown":      0.40,
}

# Domain → source type mappings
_GOVERNMENT_TLDS = {".gov", ".gov.uk", ".gov.au", ".gc.ca", ".europa.eu"}
_ACADEMIC_TLDS = {".edu", ".ac.uk", ".ac.jp"}
_ACADEMIC_DOMAINS = {"scholar.google.com", "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "jstor.org"}

_MAJOR_NEWS = {
    "nytimes.com", "washingtonpost.com", "bbc.com", "bbc.co.uk",
    "reuters.com", "apnews.com", "theguardian.com", "npr.org",
    "cnn.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com",
    "wsj.com", "economist.com", "ft.com", "bloomberg.com",
    "usatoday.com", "latimes.com", "politico.com", "thehill.com",
}

_FACTCHECK_DOMAINS = {
    "politifact.com", "factcheck.org", "snopes.com", "fullfact.org",
    "checkyourfact.com", "truthorfiction.com",
}

_SOCIAL_MEDIA = {
    "twitter.com", "x.com", "facebook.com", "reddit.com",
    "instagram.com", "tiktok.com", "youtube.com",
}


def classify_source(url: str) -> Tuple[str, float]:
    """
    Classify a URL into a source type and return (source_type, quality_score).

    Args:
        url: The source URL.

    Returns:
        (source_type, quality_score) tuple.
    """
    if not url:
        return "unknown", SOURCE_QUALITY_SCORES["unknown"]

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
        full_domain = parsed.netloc.lower()
    except Exception:
        return "unknown", SOURCE_QUALITY_SCORES["unknown"]

    # Wikipedia
    if "wikipedia.org" in domain:
        return "wikipedia", SOURCE_QUALITY_SCORES["wikipedia"]

    # Government
    for tld in _GOVERNMENT_TLDS:
        if domain.endswith(tld):
            return "government", SOURCE_QUALITY_SCORES["government"]

    # Academic
    for tld in _ACADEMIC_TLDS:
        if domain.endswith(tld):
            return "academic", SOURCE_QUALITY_SCORES["academic"]
    if domain in _ACADEMIC_DOMAINS:
        return "academic", SOURCE_QUALITY_SCORES["academic"]

    # Fact-checkers
    if domain in _FACTCHECK_DOMAINS:
        return "factcheck", SOURCE_QUALITY_SCORES["factcheck"]

    # Major news
    if domain in _MAJOR_NEWS:
        return "news_major", SOURCE_QUALITY_SCORES["news_major"]

    # Social media
    if domain in _SOCIAL_MEDIA:
        return "social_media", SOURCE_QUALITY_SCORES["social_media"]

    # Minor news / generic (has "news" in domain)
    if "news" in domain:
        return "news_minor", SOURCE_QUALITY_SCORES["news_minor"]

    # Blog detection
    if any(kw in domain for kw in ("blog", "medium.com", "substack.com", "wordpress.com", "tumblr.com")):
        return "blog", SOURCE_QUALITY_SCORES["blog"]

    return "unknown", SOURCE_QUALITY_SCORES["unknown"]
