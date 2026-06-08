import sqlite3
import json
import os

DB_PATH = os.path.join("data", "knowledge_base", "india_kg.db")
OUT_PATH = os.path.join("frontend", "kg_data.json")

def extract():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get counts grouped by domain and subdomain
    c.execute("""
        SELECT domain, sub_domain, COUNT(name) 
        FROM entities 
        GROUP BY domain, sub_domain
    """)
    rows = c.fetchall()
    
    # Build tree
    tree = {"name": "India KG", "children": []}
    
    # domain -> list of subdomains
    domain_map = {}
    for domain, subdomain, count in rows:
        if domain not in domain_map:
            domain_map[domain] = []
        domain_map[domain].append({"name": subdomain, "value": count, "isLeaf": True})
        
    for domain, subdomains in domain_map.items():
        tree["children"].append({
            "name": domain,
            "children": subdomains
        })
        
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2)
        
    print(f"Extracted {len(rows)} subdomains to {OUT_PATH}")

if __name__ == "__main__":
    extract()
