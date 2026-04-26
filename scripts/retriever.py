#!/usr/bin/env python3
"""
retriever.py
------------
Hybrid retriever combining BM25 (keyword) and vector (semantic) search.

Why hybrid?
- BM25 excels at exact part number queries: "RC0402JR-070RL"
- Vector excels at semantic queries: "low noise temp sensor for automotive"
- Hybrid beats both on mixed queries: "Yageo 10k resistor 0402"

The retriever is the backbone of the agent's search_catalog tool.

Usage (standalone benchmark):
    python scripts/retriever.py

Requirements:
    pip install rank-bm25 psycopg2-binary python-dotenv numpy
"""

import os
import re
import json
import time
import logging
import numpy as np
from dotenv import load_dotenv
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB", "partspilot"),
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

EMBED_MODEL = "all-MiniLM-L6-v2"
_embed_model = SentenceTransformer(EMBED_MODEL)
TOP_K       = 10   # number of results to return


# ---------------------------------------------------------------------------
# Tokenizer shared by BM25 indexing and query time
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Lowercase, split on non-alphanumeric chars, keep tokens >= 2 chars.
    Handles part numbers like RC0402JR-070RL by splitting on hyphens.
    """
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# BM25 index — built once at startup, held in memory
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Lightweight BM25 implementation over the parts catalog.
    Loads all parts from Postgres at startup and builds an in-memory index.
    For 1800 parts this takes <1 second and uses ~5MB RAM.
    """

    def __init__(self, conn):
        self.conn = conn
        self.parts = []        # list of dicts, one per part
        self.corpus = []       # tokenized documents
        self.idf = {}          # inverse document frequency per token
        self.k1 = 1.5          # BM25 tuning: term saturation
        self.b  = 0.75         # BM25 tuning: length normalization
        self._build()

    def _build_search_text(self, p: dict) -> str:
        """Same logic as embed.py — consistent text representation."""
        parts = []

        if p.get("manufacturer"):
            parts.append(p["manufacturer"])

        if p.get("manufacturer_part_number"):
            mpn = p["manufacturer_part_number"]
            parts.extend([mpn] * 3)  # exact part number lookup is a core use case

        if p.get("category"):
            parts.append(p["category"].replace("_", " "))

        if p.get("description"):
            desc = p["description"]
            parts.extend([desc] * 2)  # DigiKey's short description is high signal

        if p.get("detailed_description"):
            detail = p["detailed_description"]
            parts.extend([detail] * 2)  # contains searchable specs

        params = p.get("parameters") or {}
        if isinstance(params, dict):
            for k, v in params.items():
                if v and v != "-":
                    parts.append(f"{k} {v}")

        return " ".join(parts)

    def _build(self):
        log.info("Building BM25 index...")
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, digikey_part_number, manufacturer_part_number,
                       manufacturer, category, description, detailed_description,
                       unit_price_usd, stock_qty, warehouse, lead_time_days,
                       parameters
                FROM parts ORDER BY id
            """)
            self.parts = [dict(r) for r in cur.fetchall()]

        self.corpus = [tokenize(self._build_search_text(p)) for p in self.parts]

        # Compute IDF
        N = len(self.corpus)
        df = {}
        for doc in self.corpus:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1
        self.idf = {
            token: np.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in df.items()
        }

        self.avg_dl = np.mean([len(doc) for doc in self.corpus])
        log.info("BM25 index built: %d parts", len(self.parts))

    def search(self, query: str, top_k: int = TOP_K,
               category_filter: Optional[str] = None) -> list[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = np.zeros(len(self.corpus))
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for i, doc in enumerate(self.corpus):
                tf = doc.count(token)
                if tf == 0:
                    continue
                dl = len(doc)
                score = idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                )
                scores[i] += score

        # Apply category filter
        if category_filter:
            for i, p in enumerate(self.parts):
                if p.get("category") != category_filter:
                    scores[i] = 0.0

        top_indices = np.argsort(scores)[::-1][:top_k * 2]  # get extras for dedup
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                result = dict(self.parts[idx])
                result["bm25_score"] = float(scores[idx])
                result["retrieval_method"] = "bm25"
                results.append(result)
            if len(results) >= top_k:
                break

        return results


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

_model = None

# NEW
def get_embedding(text: str) -> list[float]:
    """Embed a query string using the local sentence transformer model."""
    return _embed_model.encode([text], convert_to_numpy=True)[0].tolist()


def vector_search(conn, query: str, top_k: int = TOP_K,
                  category_filter: Optional[str] = None) -> list[dict]:
    """
    Semantic search using pgvector cosine similarity.
    Embeds the query, then finds nearest neighbors in the parts table.
    """
    embedding = get_embedding(query)
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    category_clause = "AND category = %(category)s" if category_filter else ""

    sql = f"""
        SELECT id, digikey_part_number, manufacturer_part_number,
               manufacturer, category, description, detailed_description,
               unit_price_usd, stock_qty, warehouse, lead_time_days,
               parameters,
               1 - (embedding <=> %(embedding)s::vector) AS vector_score
        FROM parts
        WHERE embedding IS NOT NULL
        {category_clause}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(top_k)s
    """

    params = {"embedding": embedding_str, "top_k": top_k}
    if category_filter:
        params["category"] = category_filter

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        results = []
        for row in cur.fetchall():
            r = dict(row)
            r["retrieval_method"] = "vector"
            results.append(r)
        return results


# ---------------------------------------------------------------------------
# Hybrid search — reciprocal rank fusion
# ---------------------------------------------------------------------------

def hybrid_search(bm25_index: BM25Index, conn, query: str,
                  top_k: int = TOP_K,
                  category_filter: Optional[str] = None,
                  bm25_weight: float = 0.7,
                  vector_weight: float = 0.3) -> list[dict]:
    """
    Combines BM25 and vector search using Reciprocal Rank Fusion (RRF).

    RRF score = Σ 1 / (k + rank_i) for each retrieval method
    where k=60 is a smoothing constant.

    Why RRF instead of score normalization?
    - BM25 and vector scores are on completely different scales
    - Normalizing them requires knowing the score distribution in advance
    - RRF only needs the rank ordering, which is always comparable
    - RRF is proven to outperform score fusion in most benchmarks
    """
    RRF_K = 60

    # Run both searches in parallel (simplified: sequential here)
    t0 = time.time()
    bm25_results  = bm25_index.search(query, top_k=top_k * 2, category_filter=category_filter)
    bm25_time     = time.time() - t0

    t0 = time.time()
    vector_results = vector_search(conn, query, top_k=top_k * 2, category_filter=category_filter)
    vector_time    = time.time() - t0

    # Build RRF score map: part_id -> rrf_score
    rrf_scores: dict[int, float] = {}
    part_data:  dict[int, dict]  = {}

    for rank, result in enumerate(bm25_results):
        part_id = result["id"]
        rrf_scores[part_id] = rrf_scores.get(part_id, 0) + bm25_weight / (RRF_K + rank + 1)
        part_data[part_id]  = result

    for rank, result in enumerate(vector_results):
        part_id = result["id"]
        rrf_scores[part_id] = rrf_scores.get(part_id, 0) + vector_weight / (RRF_K + rank + 1)
        if part_id not in part_data:
            part_data[part_id] = result

    # Sort by RRF score
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for part_id, rrf_score in ranked:
        r = dict(part_data[part_id])
        r["rrf_score"]         = rrf_score
        r["retrieval_method"]  = "hybrid"
        r["bm25_time_ms"]      = round(bm25_time * 1000, 1)
        r["vector_time_ms"]    = round(vector_time * 1000, 1)
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Main retriever class — this is what the agent imports
# ---------------------------------------------------------------------------

class PartRetriever:
    """
    The interface the agent uses. Wraps hybrid search with a clean API.
    Initialize once at app startup, reuse for every query.
    """

    def __init__(self):
        self.conn  = psycopg2.connect(**DB_CONFIG)
        self.bm25  = BM25Index(self.conn)
        log.info("PartRetriever ready.")

    def search(self, query: str, top_k: int = TOP_K,
               category: Optional[str] = None,
               in_stock_only: bool = False) -> list[dict]:
        """
        Main search method. Returns top_k parts ranked by hybrid score.

        Args:
            query:         Natural language or part number query
            top_k:         Number of results (default 10)
            category:      Optional filter: resistors, capacitors, sensors, etc.
            in_stock_only: If True, only return parts with stock_qty > 0
        """
        results = hybrid_search(
            self.bm25, self.conn, query,
            top_k=top_k * 2 if in_stock_only else top_k,
            category_filter=category,
        )

        if in_stock_only:
            results = [r for r in results if r.get("stock_qty", 0) > 0][:top_k]

        # Clean up internal scoring fields before returning to agent
        for r in results:
            r.pop("bm25_score", None)
            r.pop("vector_score", None)

        return results

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Benchmark — run this to generate the comparison table for your README
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    ("RC0402JR-070RL",                          "RC0402JR-070RL",           "exact_part_number"),
    ("0 ohm jumper resistor 0402",              "RC0402JR-070RL",           "spec_description"),
    ("Yageo chip resistor",                     "RC0402JR-070RL",           "manufacturer_keyword"),
    ("Samsung ceramic capacitor 100nF 10V",     "CL05B104KP5NNNC",          "spec_description"),
    ("NTC thermistor 10k temperature sensor",   "NTCG103JF103FT1",          "semantic"),
    ("Microchip 8bit microcontroller SOIC",     "PIC12F1571T-I/SN",         "mcu_keyword"),
    ("PIC16 flash microcontroller DIP",         "PIC16F684-I/P",            "mcu_exact"),
    ("dc dc converter 3.3V power module",       "XCL206B333CR-G",           "power_semantic"),
    ("low power sensor 0402 package",           None,                       "semantic_general"),
    ("proximity sensor automotive grade",       None,                       "application_query"),
]


def run_benchmark(bm25_index: BM25Index, conn):
    """
    Runs all benchmark queries against BM25-only, vector-only, and hybrid.
    Prints a comparison table suitable for your README.
    """
    print("\n" + "=" * 90)
    print(f"{'Query':<42} {'Type':<20} {'BM25':>6} {'Vector':>8} {'Hybrid':>8}")
    print("=" * 90)

    bm25_hits, vector_hits, hybrid_hits = 0, 0, 0
    bm25_times, vector_times, hybrid_times = [], [], []

    for query, expected_mpn, qtype in BENCHMARK_QUERIES:
        # BM25 only
        t0 = time.time()
        bm25_res = bm25_index.search(query, top_k=5)
        bm25_t = (time.time() - t0) * 1000
        bm25_times.append(bm25_t)

        # Vector only
        t0 = time.time()
        vec_res = vector_search(conn, query, top_k=5)
        vec_t = (time.time() - t0) * 1000
        vector_times.append(vec_t)

        # Hybrid
        t0 = time.time()
        hyb_res = hybrid_search(bm25_index, conn, query, top_k=5)
        hyb_t = (time.time() - t0) * 1000
        hybrid_times.append(hyb_t)

        if expected_mpn:
            bm25_hit   = any(r["manufacturer_part_number"] == expected_mpn for r in bm25_res)
            vector_hit = any(r["manufacturer_part_number"] == expected_mpn for r in vec_res)
            hybrid_hit = any(r["manufacturer_part_number"] == expected_mpn for r in hyb_res)
            bm25_hits   += int(bm25_hit)
            vector_hits += int(vector_hit)
            hybrid_hits += int(hybrid_hit)
            b = "✓" if bm25_hit else "✗"
            v = "✓" if vector_hit else "✗"
            h = "✓" if hybrid_hit else "✗"
        else:
            b = v = h = "-"

        print(f"{query[:40]:<42} {qtype[:18]:<20} {b:>6} {v:>8} {h:>8}")

    n_graded = sum(1 for _, mpn, _ in BENCHMARK_QUERIES if mpn)
    print("=" * 90)
    print(f"{'Recall@5 (graded queries)':<42} {'':20} "
          f"{bm25_hits}/{n_graded}  "
          f"{vector_hits}/{n_graded}     "
          f"{hybrid_hits}/{n_graded}")
    print(f"{'Avg latency (ms)':<42} {'':20} "
          f"{np.mean(bm25_times):>5.0f}   "
          f"{np.mean(vector_times):>7.0f}   "
          f"{np.mean(hybrid_times):>7.0f}")
    print("=" * 90)
    print()
    print("Note: copy this table into your README as the retrieval comparison section.")
    print()


def main():
    """Run benchmark when called directly."""
    conn = psycopg2.connect(**DB_CONFIG)

    # Check embeddings exist
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM parts WHERE embedding IS NOT NULL")
        embedded = cur.fetchone()[0]

    if embedded == 0:
        print("\n❌  No embeddings found. Run scripts/embed.py first.\n")
        conn.close()
        return

    log.info("Embeddings found for %d parts. Running benchmark...", embedded)

    bm25 = BM25Index(conn)

    # Demo: show results for a few sample queries
    demo_queries = [
        "RC0402JR-070RL",
        "temperature sensor automotive 0402",
        "10k resistor thick film",
    ]

    for q in demo_queries:
        print(f"\n{'─'*60}")
        print(f"Query: '{q}'")
        print(f"{'─'*60}")
        results = hybrid_search(bm25, conn, q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['category']}] {r['manufacturer']} {r['manufacturer_part_number']}")
            print(f"     {r['description']}")
            print(f"     Stock: {r['stock_qty']} | Price: ${r['unit_price_usd']} | RRF: {r['rrf_score']:.4f}")

    run_benchmark(bm25, conn)
    conn.close()


if __name__ == "__main__":
    main()
