# -*- coding: utf-8 -*-

"""
Retrieval Cache — SQLite-backed cache for web search results.

Avoids redundant API calls and enables reproducible re-runs.

Keyed by (query_hash, backend_type).
Stores: query, timestamp, source, URL, content, scores.
Supports TTL-based expiration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("TRUST_agents.retrieval_cache")

_DEFAULT_DB_PATH = "retrieval_index/retrieval_cache.db"
_DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


class RetrievalCache:
    """
    SQLite-backed cache for search results and fetched documents.

    Usage::

        cache = RetrievalCache()
        key = cache.make_key("some query", "wikipedia")
        hit = cache.get(key)
        if hit is None:
            results = backend.search(...)
            cache.put(key, results, query="some query", backend="wikipedia")
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._conn: Optional[sqlite3.Connection] = None

        self._hits = 0
        self._misses = 0

        self._init_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key         TEXT PRIMARY KEY,
                query       TEXT NOT NULL,
                backend     TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                data        TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        logger.info("[RetrievalCache] Initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(query: str, backend_type: str) -> str:
        """Compute a cache key from *query* and *backend_type*."""
        raw = f"{backend_type}::{query.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached results for *key*.

        Returns ``None`` on miss or if the entry has expired.
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT data, timestamp FROM cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            self._misses += 1
            logger.debug("[RetrievalCache] MISS key=%s…", key[:12])
            return None

        data_json, ts = row
        age = time.time() - ts
        if age > self.ttl_seconds:
            # Expired — delete and return miss
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            self._misses += 1
            logger.debug("[RetrievalCache] EXPIRED key=%s… (age=%.0fs)", key[:12], age)
            return None

        self._hits += 1
        logger.debug("[RetrievalCache] HIT key=%s… (age=%.0fs)", key[:12], age)
        return json.loads(data_json)

    def put(
        self,
        key: str,
        data: List[Dict[str, Any]],
        query: str = "",
        backend: str = "",
    ) -> None:
        """Store *data* under *key*."""
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cache (key, query, backend, timestamp, data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, query, backend, time.time(), json.dumps(data, ensure_ascii=False)),
        )
        self._conn.commit()
        logger.debug("[RetrievalCache] PUT key=%s… (%d items)", key[:12], len(data))

    def stats(self) -> Dict[str, int]:
        """Return hit/miss statistics."""
        return {"hits": self._hits, "misses": self._misses}

    def clear(self) -> None:
        """Delete all cached entries."""
        assert self._conn is not None
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()
        logger.info("[RetrievalCache] Cache cleared")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
