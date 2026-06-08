"""
kg_query.py — Fast lookup utilities for the India Knowledge Graph.
Used by the NLP/inference engine to resolve entity names → hashtags.
"""

import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "india_kg.db"


def _conn(db_path: Optional[Path] = None):
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────
# Core lookup — name → hashtags
# ─────────────────────────────────────────────

def get_hashtags_for_name(name: str, db_path: Optional[Path] = None) -> list[str]:
    """
    Given a person/entity name, return its hashtags.
    e.g. "Virat Kohli" → ["cricket", "indiancricket", "bcci", "viratkohli"]
    """
    conn = _conn(db_path)
    cur = conn.cursor()

    # exact match first
    cur.execute("""
        SELECT e.wikidata_id, e.domain, e.sub_domain
        FROM entities e
        WHERE LOWER(e.name) = LOWER(?)
        LIMIT 1
    """, (name,))
    row = cur.fetchone()

    # fallback: partial match
    if not row:
        cur.execute("""
            SELECT e.wikidata_id, e.domain, e.sub_domain
            FROM entities e
            WHERE LOWER(e.name) LIKE LOWER(?)
            LIMIT 1
        """, (f"%{name}%",))
        row = cur.fetchone()

    if not row:
        conn.close()
        return []

    wid = row["wikidata_id"]

    cur.execute("""
        SELECT tag FROM tags WHERE entity_id = ?
    """, (wid,))
    tags = [r["tag"] for r in cur.fetchall()]

    conn.close()
    return tags


def get_entity_info(name: str, db_path: Optional[Path] = None) -> Optional[dict]:
    """
    Full entity record for a given name.
    Returns dict with name, domain, sub_domain, sensitivity, wikidata_id, tags.
    """
    conn = _conn(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT e.wikidata_id, e.name, e.domain, e.sub_domain,
               e.sensitivity, e.entity_type
        FROM entities e
        WHERE LOWER(e.name) = LOWER(?)
        LIMIT 1
    """, (name,))
    row = cur.fetchone()

    if not row:
        cur.execute("""
            SELECT e.wikidata_id, e.name, e.domain, e.sub_domain,
                   e.sensitivity, e.entity_type
            FROM entities e
            WHERE LOWER(e.name) LIKE LOWER(?)
            LIMIT 1
        """, (f"%{name}%",))
        row = cur.fetchone()

    if not row:
        conn.close()
        return None

    wid = row["wikidata_id"]
    cur.execute("SELECT tag, weight FROM tags WHERE entity_id = ?", (wid,))
    tags = [r["tag"] for r in cur.fetchall()]

    result = {
        "wikidata_id": wid,
        "name":        row["name"],
        "domain":      row["domain"],
        "sub_domain":  row["sub_domain"],
        "sensitivity": row["sensitivity"],
        "entity_type": row["entity_type"],
        "tags":        tags,
    }
    conn.close()
    return result


def get_relationships(wikidata_id: str, db_path: Optional[Path] = None) -> list[dict]:
    """
    Return all relationships for an entity.
    e.g. [{"predicate": "PLAYS_FOR", "object": "Chennai Super Kings"}, ...]
    """
    conn = _conn(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT predicate, object_id
        FROM relationships
        WHERE subject_id = ?
    """, (wikidata_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"predicate": r["predicate"], "object": r["object_id"]} for r in rows]


def search_entities(query: str, domain: Optional[str] = None,
                    limit: int = 10, db_path: Optional[Path] = None) -> list[dict]:
    """
    Search entities by partial name. Optionally filter by domain.
    Used for testing and debugging.
    """
    conn = _conn(db_path)
    cur = conn.cursor()

    if domain:
        cur.execute("""
            SELECT name, domain, sub_domain, wikidata_id
            FROM entities
            WHERE LOWER(name) LIKE LOWER(?) AND domain = ?
            ORDER BY name
            LIMIT ?
        """, (f"%{query}%", domain, limit))
    else:
        cur.execute("""
            SELECT name, domain, sub_domain, wikidata_id
            FROM entities
            WHERE LOWER(name) LIKE LOWER(?)
            ORDER BY name
            LIMIT ?
        """, (f"%{query}%", limit))

    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_db_stats(db_path: Optional[Path] = None) -> dict:
    """
    Quick summary of what's in the DB.
    """
    conn = _conn(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as c FROM entities")
    total = cur.fetchone()["c"]

    cur.execute("SELECT domain, COUNT(*) as c FROM entities GROUP BY domain ORDER BY c DESC")
    by_domain = {r["domain"]: r["c"] for r in cur.fetchall()}

    cur.execute("SELECT COUNT(*) as c FROM tags")
    total_tags = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) as c FROM relationships")
    total_rels = cur.fetchone()["c"]

    cur.execute("""
        SELECT category, entities_added, fetch_time, status
        FROM fetch_log
        ORDER BY fetch_time DESC
        LIMIT 10
    """)
    recent_fetches = [dict(r) for r in cur.fetchall()]

    conn.close()
    return {
        "total_entities":    total,
        "by_domain":         by_domain,
        "total_tags":        total_tags,
        "total_relationships": total_rels,
        "recent_fetches":    recent_fetches,
    }


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    stats = get_db_stats()
    print("\n=== DB STATS ===")
    print(f"Total entities   : {stats['total_entities']}")
    print(f"Total tags       : {stats['total_tags']}")
    print(f"Total relations  : {stats['total_relationships']}")
    print(f"\nBy domain:")
    for domain, count in stats["by_domain"].items():
        print(f"  {domain:<20} {count}")

    print("\n=== SAMPLE LOOKUPS ===")
    for name in ["Virat Kohli", "Sachin Tendulkar", "Narendra Modi", "Rahul Gandhi"]:
        tags = get_hashtags_for_name(name)
        print(f"  {name:<25} → {tags}")

    print("\n=== SEARCH TEST ===")
    results = search_entities("Dhoni", domain="Sports")
    for r in results:
        print(f"  {r['name']:<25} {r['sub_domain']}")
