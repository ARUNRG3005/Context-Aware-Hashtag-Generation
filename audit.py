import os, sqlite3, pandas as pd, json
from collections import Counter

BASE = r'D:\hashtag-generator'
KG   = os.path.join(BASE, 'data', 'knowledge_base', 'india_kg.db')
PROC = os.path.join(BASE, 'data', 'processed')
RAW  = os.path.join(BASE, 'data', 'raw')

print('=' * 60)
print('FULL AUDIT — India Hashtag Generator Project')
print('=' * 60)

# --- KG STATS ---
conn = sqlite3.connect(KG)
cur  = conn.cursor()

cur.execute('SELECT COUNT(*) FROM entities')
total_entities = cur.fetchone()[0]

cur.execute('SELECT COUNT(*) FROM tags')
total_tags = cur.fetchone()[0]

cur.execute('SELECT domain, COUNT(*) as c FROM entities WHERE domain IS NOT NULL GROUP BY domain ORDER BY c DESC')
domains = cur.fetchall()

cur.execute('SELECT sub_domain, COUNT(*) as c FROM entities WHERE sub_domain IS NOT NULL GROUP BY sub_domain ORDER BY c DESC LIMIT 20')
subdomains = cur.fetchall()

cur.execute('SELECT tag, COUNT(*) as c FROM tags GROUP BY tag ORDER BY c DESC LIMIT 20')
top_tags = cur.fetchall()

cur.execute("SELECT COUNT(DISTINCT entity_id) FROM tags")
tagged_entities = cur.fetchone()[0]

conn.close()

print()
print('=== KNOWLEDGE GRAPH ===')
print(f'  Total entities:        {total_entities:,}')
print(f'  Entities with tags:    {tagged_entities:,}')
print(f'  Total tag mappings:    {total_tags:,}')
print()
print('  By Domain:')
for d, c in domains:
    bar = chr(9608) * (c // 50)
    print(f'    {d:<20} {c:>6}  {bar}')

print()
print('  Top 20 Sub-Domains:')
for sd, c in subdomains:
    print(f'    {sd:<30} {c:>5}')

print()
print('  Top 20 Tags in KG:')
for tag, c in top_tags:
    print(f'    #{tag:<30} {c:>5}')

# --- DATASET STATS ---
print()
print('=== PROCESSED DATASET ===')
for split in ['train', 'val', 'test']:
    path = os.path.join(PROC, f'{split}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f'  {split}.csv : {len(df):>6} rows  ({size_mb:.1f} MB)')
        if split == 'train':
            print(f'    Domains: {dict(df["domain"].value_counts())}')
            # Unique labels
            all_labels = []
            for l in df['labels']:
                all_labels.extend(str(l).split('|'))
            unique = set(all_labels)
            counts = Counter(all_labels).most_common(25)
            print(f'    Unique hashtags: {len(unique)}')
            print(f'    Top 25 hashtags:')
            for tag, c in counts:
                print(f'      #{tag:<35} {c:>6}')

# --- RAW FILES ---
print()
print('=== RAW DATA FILES (D: Drive) ===')
total_size = 0
for f in sorted(os.listdir(RAW)):
    if f.endswith('.csv'):
        size = os.path.getsize(os.path.join(RAW, f))
        total_size += size
        print(f'  {f:<45} {size//1024:>8} KB')
print(f'  TOTAL:                                            {total_size//1024//1024:>5} MB')
