import sqlite3
c = sqlite3.connect('data/knowledge_base/india_kg.db')

print('=== TABLES ===')
print(c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())

for table in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
    t = table[0]
    print(f'\n=== {t.upper()} SCHEMA ===')
    print(c.execute(f"PRAGMA table_info({t})").fetchall())
    print(f'=== {t.upper()} COUNT ===')
    print(c.execute(f"SELECT COUNT(*) FROM {t}").fetchone())
    print(f'=== {t.upper()} SAMPLE (10 rows) ===')
    for row in c.execute(f"SELECT * FROM {t} LIMIT 10").fetchall():
        print(row)