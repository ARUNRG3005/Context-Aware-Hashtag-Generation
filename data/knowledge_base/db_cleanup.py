"""
db_cleanup.py — One-time cleanup of india_kg.db before running Round 2+
Fixes:
  1. Duplicate tags (same entity_id+tag stored many times)
  2. TitleCase tags → lowercase (normalise format)
  3. Adds UNIQUE constraint on (entity_id, tag) so it never happens again
Run once: python db_cleanup.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "india_kg.db"


def cleanup():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── Step 1: count before ──────────────────────────────
    cur.execute("SELECT COUNT(*) FROM tags")
    before = cur.fetchone()[0]
    print(f"Tags before cleanup : {before}")

    # ── Step 2: lowercase all existing tags ───────────────
    cur.execute("UPDATE tags SET tag = LOWER(tag)")
    conn.commit()
    print("  ✓ All tags lowercased")

    # ── Step 3: remove duplicate (entity_id, tag) pairs ──
    # Keep only the row with the lowest rowid for each unique pair
    cur.execute("""
        DELETE FROM tags
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM tags
            GROUP BY entity_id, tag
        )
    """)
    conn.commit()
    removed = before - cur.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
    print(f"  ✓ Removed {removed} duplicate tag rows")

    # ── Step 4: add UNIQUE constraint so it can't happen again ──
    # SQLite can't ALTER to add unique constraints directly,
    # so we rebuild the table with a proper constraint.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tags_new (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT,
            tag       TEXT,
            weight    REAL DEFAULT 1.0,
            FOREIGN KEY (entity_id) REFERENCES entities(id),
            UNIQUE(entity_id, tag)
        )
    """)
    cur.execute("""
        INSERT OR IGNORE INTO tags_new (entity_id, tag, weight)
        SELECT entity_id, tag, weight FROM tags
    """)
    cur.execute("DROP TABLE tags")
    cur.execute("ALTER TABLE tags_new RENAME TO tags")
    conn.commit()
    print("  ✓ tags table rebuilt with UNIQUE(entity_id, tag) constraint")

    # ── Step 5: final count ───────────────────────────────
    cur.execute("SELECT COUNT(*) FROM tags")
    after = cur.fetchone()[0]
    print(f"\nTags after cleanup  : {after}")
    print(f"Removed             : {before - after} duplicate rows")

    # ── Step 6: verify a known entity ────────────────────
    print("\n=== VERIFICATION ===")
    for name in ["Virat Kohli", "Narendra Modi", "Sachin Tendulkar"]:
        cur.execute("""
            SELECT t.tag FROM tags t
            JOIN entities e ON t.entity_id = e.wikidata_id
            WHERE LOWER(e.name) = LOWER(?)
        """, (name,))
        tags = [r[0] for r in cur.fetchall()]
        print(f"  {name:<25} → {tags}")

    # ── Step 7: domain summary ────────────────────────────
    print("\n=== DB SUMMARY ===")
    cur.execute("SELECT COUNT(*) FROM entities")
    print(f"  Entities   : {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM tags")
    print(f"  Tags       : {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM relationships")
    print(f"  Relations  : {cur.fetchone()[0]}")
    print()
    cur.execute("SELECT domain, COUNT(*) c FROM entities GROUP BY domain ORDER BY c DESC")
    for r in cur.fetchall():
        print(f"  {r[0]:<20} {r[1]}")

    conn.close()
    print("\nCleanup complete. Ready for Round 2.")


if __name__ == "__main__":
    cleanup()
