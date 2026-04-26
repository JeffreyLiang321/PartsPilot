#!/usr/bin/env python3
"""
embed.py
--------
Computes sentence-transformer embeddings for all parts
and stores them in the parts.embedding column (pgvector).

Uses all-MiniLM-L6-v2 — free, local, no API key needed.
Produces 384-dimensional vectors.

Run this ONCE after ingest.py. Takes ~2 minutes for 1,800 parts on CPU.

Usage:
    python scripts/embed.py

Requirements:
    pip install sentence-transformers psycopg2-binary python-dotenv
"""

import os
import logging
from dotenv import load_dotenv
import psycopg2
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

EMBED_MODEL = "all-MiniLM-L6-v2"   # 384 dims, free, local, fast
BATCH_SIZE  = 64


def get_model() -> SentenceTransformer:
    """Load the model once — cached after first call."""
    log.info("Loading sentence transformer model (downloads ~80MB on first run)...")
    model = SentenceTransformer(EMBED_MODEL)
    log.info("Model loaded.")
    return model


def build_search_text(row: dict) -> str:
    """
    Builds the text string we embed for each part.
    Combines all fields useful for semantic search.
    Must be identical to build_search_text in retriever.py.
    """
    parts_text = []

    if row.get("manufacturer"):
        parts_text.append(row["manufacturer"])
    if row.get("manufacturer_part_number"):
        parts_text.append(row["manufacturer_part_number"])
    if row.get("category"):
        parts_text.append(row["category"].replace("_", " "))
    if row.get("description"):
        parts_text.append(row["description"])
    if row.get("detailed_description"):
        parts_text.append(row["detailed_description"])

    params = row.get("parameters") or {}
    if isinstance(params, dict):
        for k, v in params.items():
            if v and v != "-":
                parts_text.append(f"{k}: {v}")

    return " | ".join(parts_text)


def get_parts_without_embeddings(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, manufacturer, manufacturer_part_number, category,
                   description, detailed_description, parameters
            FROM parts
            WHERE embedding IS NULL
            ORDER BY id
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def embed_batch(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Encode a batch of texts locally using sentence transformers."""
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def store_embeddings(conn, id_embedding_pairs: list[tuple]):
    """Write computed embeddings into the parts table."""
    with conn.cursor() as cur:
        for part_id, embedding in id_embedding_pairs:
            cur.execute(
                "UPDATE parts SET embedding = %s WHERE id = %s",
                (embedding, part_id)
            )
    conn.commit()


def create_vector_index(conn):
    """Create IVFFlat index for fast approximate nearest-neighbor search."""
    log.info("Creating pgvector IVFFlat index...")
    with conn.cursor() as cur:
        # lists = ~sqrt(num_rows), appropriate for 1800 parts
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_parts_embedding
            ON parts USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 30);
        """)
    conn.commit()
    log.info("Vector index created.")


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    # Verify the embedding column exists and is the right dimension
    with conn.cursor() as cur:
        cur.execute("""
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = 'parts'::regclass AND attname = 'embedding'
        """)
        row = cur.fetchone()
        if row is None:
            print("\n❌  No embedding column found. Run this first:\n")
            print("docker exec -it partspilot_db psql -U postgres -d partspilot -c \"")
            print("DROP INDEX IF EXISTS idx_parts_embedding;")
            print("ALTER TABLE parts DROP COLUMN IF EXISTS embedding;")
            print("ALTER TABLE parts ADD COLUMN embedding vector(384);\"")
            conn.close()
            return

    parts = get_parts_without_embeddings(conn)
    if not parts:
        log.info("All parts already have embeddings. Nothing to do.")
        create_vector_index(conn)
        conn.close()
        return

    log.info("Embedding %d parts locally (no API calls)...", len(parts))

    model = get_model()
    total_embedded = 0

    for i in range(0, len(parts), BATCH_SIZE):
        batch = parts[i: i + BATCH_SIZE]
        texts = [build_search_text(p) for p in batch]

        try:
            embeddings = embed_batch(model, texts)
            pairs = [(p["id"], emb) for p, emb in zip(batch, embeddings)]
            store_embeddings(conn, pairs)
            total_embedded += len(batch)
            log.info("  Embedded %d/%d parts...", total_embedded, len(parts))
        except Exception as e:
            log.error("  Batch %d failed: %s", i // BATCH_SIZE + 1, e)

    log.info("Embeddings complete: %d parts", total_embedded)
    create_vector_index(conn)
    conn.close()
    log.info("✅  Done. Ready to run the hybrid retriever.")


if __name__ == "__main__":
    main()
