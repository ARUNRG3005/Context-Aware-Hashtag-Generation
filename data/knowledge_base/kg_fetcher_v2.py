import os
import sys
import json
import sqlite3
import requests
import time
from urllib.parse import quote
from datetime import datetime

# Path setup
BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE, "data", "knowledge_base", "india_kg.db")

WIKI_CATEGORIES = {
    "Sports": [
        ("Athletics", "Category:Indian_athletes"),
        ("Athletics", "Category:Indian_cricketers"),
        ("Athletics", "Category:Indian_footballers"),
        ("Athletics", "Category:Indian_badminton_players"),
        ("Athletics", "Category:Indian_tennis_players"),
        ("Athletics", "Category:Olympic_medalists_for_India")
    ],
    "Politics": [
        ("Political Party", "Category:Political_parties_in_India"),
        ("Indian Politics", "Category:Chief_ministers_of_Indian_states"),
        ("Indian Politics", "Category:Prime_Ministers_of_India"),
        ("Elections", "Category:Elections_in_India")
    ],
    "Business": [
        ("Indian Business", "Category:Companies_of_India"),
        ("Indian Business", "Category:Indian_billionaires"),
        ("Indian Business", "Category:Indian_businesspeople"),
        ("Indian Business", "Category:Indian_chief_executives"),
        ("Indian Business", "Category:Government-owned_companies_of_India")
    ],
    "Science": [
        ("Indian Science", "Category:Indian_scientists"),
        ("Indian Science", "Category:Indian_Space_Research_Organisation"),
        ("Indian Science", "Category:Indian_physicists"),
        ("Indian Science", "Category:Indian_mathematicians")
    ],
    "Entertainment": [
        ("Bollywood", "Category:Hindi-language_films"),
        ("Tollywood", "Category:Telugu-language_films"),
        ("Kollywood", "Category:Tamil-language_films"),
        ("Mollywood", "Category:Malayalam-language_films"),
        ("Sandalwood", "Category:Kannada-language_films"),
        ("Indian Cinema", "Category:Indian_film_directors")
    ],
    "Places": [
        ("Indian City", "Category:Cities_in_India"),
        ("Indian State", "Category:States_and_union_territories_of_India")
    ],
    "Health": [
        ("Hospitals", "Category:Hospitals_in_India"),
        ("Medical Institutions", "Category:Medical_colleges_in_India"),
        ("Diseases", "Category:Endemic_diseases_in_India"),
        ("Public Health", "Category:Health_in_India")
    ],
    "Education": [
        ("Universities", "Category:Universities_in_India"),
        ("Institutes", "Category:Indian_Institutes_of_Technology"),
        ("Institutes", "Category:Indian_Institutes_of_Management"),
        ("Education Boards", "Category:Boards_of_education_in_India")
    ],
    "Environment": [
        ("National Parks", "Category:National_parks_of_India"),
        ("Rivers", "Category:Rivers_of_India"),
        ("Wildlife", "Category:Fauna_of_India"),
        ("Climate Issues", "Category:Climate_change_in_India")
    ],
    "Crime": [
        ("Indian Law", "Category:Indian_criminal_law"),
        ("Law Enforcement", "Category:Law_enforcement_agencies_of_India"),
        ("Social Issues", "Category:Social_issues_in_India"),
        ("Controversies", "Category:Scandals_in_India")
    ],
    "LawAndJustice": [
        ("Supreme Court", "Category:Supreme_Court_of_India_cases"),
        ("Judges", "Category:Indian_judges"),
        ("Law", "Category:Law_of_India")
    ],
    "Defense": [
        ("Armed Forces", "Category:Military_of_India"),
        ("Army Regiments", "Category:Regiments_of_the_Indian_Army"),
        ("Missiles", "Category:Missiles_of_India")
    ],
    "Startups": [
        ("Unicorns", "Category:Unicorn_startup_companies_of_India"),
        ("Tech Startups", "Category:Technology_companies_of_India")
    ],
    "Disasters": [
        ("Monsoons", "Category:Monsoons_in_India"),
        ("Natural Disasters", "Category:Natural_disasters_in_India")
    ],
    "Geopolitics": [
        ("Foreign Policy", "Category:Foreign_relations_of_India"),
        ("Treaties", "Category:Treaties_of_India")
    ],
    "Economy": [
        ("Indian Economy", "Category:Economy_of_India"),
        ("PSU", "Category:Public_sector_undertakings_of_India"),
        ("Banks", "Category:Banks_of_India")
    ],
    "ArtsAndCulture": [
        ("Festivals", "Category:Festivals_in_India"),
        ("Literature", "Category:Indian_literature"),
        ("Heritage", "Category:World_Heritage_Sites_in_India")
    ],
    "Infrastructure": [
        ("Highways", "Category:National_Highways_in_India"),
        ("Transit", "Category:Rapid_transit_in_India")
    ]
}

def fetch_category_members(category_name, max_pages=3000):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_name,
        "cmlimit": "max",
        "format": "json"
    }
    
    members = []
    retries = 3
    delay = 1
    
    headers = {"User-Agent": "HashtagGeneratorBot/1.0 (contact@example.com)"}
    
    while True:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 429:
                raise Exception("429 Too Many Requests")
            data = response.json()
            if 'query' in data and 'categorymembers' in data['query']:
                for member in data['query']['categorymembers']:
                    if member['ns'] == 0:  # Only main namespace
                        members.append(member['title'])
                        if len(members) >= max_pages:
                            return members
            
            if 'continue' in data:
                params.update(data['continue'])
            else:
                break
        except Exception as e:
            print(f"Error fetching {category_name}: {e}")
            retries -= 1
            if retries <= 0:
                break
            time.sleep(delay)
            delay *= 2
    return members

def make_entity(wid, name, domain, sub_domain, sensitivity=0, entity_type="PERSON", aliases=""):
    return {
        "wikidata_id":  wid,
        "name":         name,
        "domain":       domain,
        "sub_domain":   sub_domain,
        "sensitivity":  sensitivity,
        "entity_type":  entity_type,
        "aliases":      aliases
    }

def main():
    print("=" * 60)
    print("KG FETCHER V2 — Aggressive Scale")
    print("=" * 60)
    
    # Track existing to avoid duplicates
    existing = set()
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT name FROM entities")
            existing = set([row[0] for row in c.fetchall()])
            conn.close()
        except:
            pass

    entities_to_insert = []
    count = 0
    
    for domain, cats in WIKI_CATEGORIES.items():
        print(f"\nProcessing Domain: {domain}")
        for sub_domain, cat_name in cats:
            print(f"  Fetching: {cat_name} -> {sub_domain}")
            members = fetch_category_members(cat_name)
            added = 0
            for name in members:
                # Clean name (remove parentheses)
                clean_name = name.split("(")[0].strip()
                if clean_name not in existing:
                    # Basic sensitivity logic
                    sens = 0
                    if domain == "Crime": sens = 2
                    if domain == "Politics": sens = 1
                    
                    ent = make_entity(
                        wid=f"Q{len(existing)+count}", # fake ID
                        name=clean_name,
                        domain=domain,
                        sub_domain=sub_domain,
                        sensitivity=sens,
                        entity_type="ORG" if domain in ["Business", "Education", "Health"] else "PERSON"
                    )
                    entities_to_insert.append(ent)
                    existing.add(clean_name)
                    count += 1
                    added += 1
            print(f"    ✓ Added {added} new entities")
            time.sleep(0.5)

    if not entities_to_insert:
        print("\nNo new entities to add.")
        return

    print(f"\nTotal new entities to inject: {len(entities_to_insert)}")
    
    from kg_store import upsert_entity, upsert_tags
    from datetime import datetime
    
    for idx, ent in enumerate(entities_to_insert):
        # Auto-generate tags
        tags = ["india", domain.lower().replace(" ", "")]
        if sub_domain:
            tags.append(sub_domain.lower().replace(" ", ""))
            
        if domain == "Crime":
            tags.append("crime")
        if domain == "Sports":
            tags.append("sports")
            tags.append("athlete")
        if domain == "Entertainment":
            tags.append("indiancinema")
            
        # Clean tags
        clean_tags = [t for t in tags if t]
        
        ent["last_updated"] = datetime.now().isoformat()
        upsert_entity(ent)
        upsert_tags(ent["wikidata_id"], clean_tags)
        
        if idx % 1000 == 0 and idx > 0:
            print(f"  Committed {idx} / {len(entities_to_insert)}")

    print(f"✓ Successfully expanded KG. ({count} new entities)")

if __name__ == "__main__":
    main()
