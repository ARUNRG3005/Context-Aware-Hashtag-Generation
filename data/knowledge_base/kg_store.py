"""
kg_store.py — SQLite storage for India Knowledge Graph.
Schema matches kg_fetcher.py exactly.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "india_kg.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            wikidata_id   TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            aliases       TEXT DEFAULT '',
            entity_type   TEXT DEFAULT 'PERSON',
            domain        TEXT,
            sub_domain    TEXT,
            sensitivity   INTEGER DEFAULT 0,
            last_updated  TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id   TEXT NOT NULL,
            tag         TEXT NOT NULL,
            weight      REAL DEFAULT 1.0,
            UNIQUE(entity_id, tag),
            FOREIGN KEY (entity_id) REFERENCES entities(wikidata_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id  TEXT NOT NULL,
            predicate   TEXT NOT NULL,
            object_id   TEXT NOT NULL,
            UNIQUE(subject_id, predicate, object_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            fetch_time       TEXT,
            category         TEXT,
            entities_added   INTEGER DEFAULT 0,
            entities_updated INTEGER DEFAULT 0,
            status           TEXT,
            error_message    TEXT
        )
    """)

    # Indexes for fast lookup
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tags_entity ON tags(entity_id)")

    conn.commit()
    conn.close()
    print("Database initialized.")


def upsert_entity(entity: dict):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO entities
            (wikidata_id, name, aliases, entity_type, domain, sub_domain, sensitivity, last_updated)
        VALUES
            (:wikidata_id, :name, :aliases, :entity_type, :domain, :sub_domain, :sensitivity, :last_updated)
        ON CONFLICT(wikidata_id) DO UPDATE SET
            name         = excluded.name,
            domain       = excluded.domain,
            sub_domain   = excluded.sub_domain,
            last_updated = excluded.last_updated
    """, entity)
    conn.commit()
    conn.close()


def upsert_tags(entity_id: str, tags: list):
    conn = get_connection()
    cur = conn.cursor()
    for tag in tags:
        if tag and isinstance(tag, str):
            cur.execute("""
                INSERT OR IGNORE INTO tags (entity_id, tag, weight)
                VALUES (?, ?, 1.0)
            """, (entity_id, tag.strip().lower()))
    conn.commit()
    conn.close()


def upsert_relationship(subject_id: str, predicate: str, object_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO relationships (subject_id, predicate, object_id)
        VALUES (?, ?, ?)
    """, (subject_id, predicate, object_id))
    conn.commit()
    conn.close()


def log_fetch(category: str, added: int, updated: int, status: str, error: str = None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO fetch_log (fetch_time, category, entities_added, entities_updated, status, error_message)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), category, added, updated, status, error))
    conn.commit()
    conn.close()


def get_last_fetch(category: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT fetch_time FROM fetch_log
        WHERE category = ? AND status = 'success'
        ORDER BY fetch_time DESC LIMIT 1
    """, (category,))
    row = cur.fetchone()
    conn.close()
    return row["fetch_time"] if row else None


if __name__ == "__main__":
    initialize_db()