"""
kg_fetcher.py — Wikidata SPARQL fetcher for India Knowledge Graph
Rounds 1-6 : Cricketers, Politicians, Actors, Business, Athletes, Scientists
Round 7    : IPL Teams + Cricket Boards
Round 8    : Political Parties
Round 9    : Indian States + Major Cities
Round 10   : Major Indian Events (elections, tournaments, etc.)
"""

import time

import requests
from datetime import datetime
from kg_store import upsert_entity, upsert_tags, upsert_relationship, log_fetch

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "IndiaHashtagKG/1.0 (hashtag-generator project)"}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def val(row, key):
    return row.get(key, {}).get("value", "").strip()


def run_sparql(query, retries=3, delay=5):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                WIKIDATA_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("results", {}).get("bindings", [])
            elif r.status_code == 429:
                wait = delay * attempt * 2
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    HTTP {r.status_code} on attempt {attempt}")
                time.sleep(delay)
        except Exception as e:
            print(f"    Error on attempt {attempt}: {e}")
            time.sleep(delay)
    return []


def make_entity(wid, name, domain, sub_domain,
                sensitivity=0, entity_type="PERSON", aliases=""):
    return {
        "wikidata_id":  wid,
        "name":         name,
        "aliases":      aliases,
        "entity_type":  entity_type,
        "domain":       domain,
        "sub_domain":   sub_domain,
        "sensitivity":  sensitivity,
        "last_updated": datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────
# IPL team → hashtag mapping
# ─────────────────────────────────────────────

IPL_TEAM_TAGS = {
    "Chennai Super Kings":      ["csk", "chennaIsuperkings", "yellove", "ipl"],
    "Mumbai Indians":           ["mi", "mumbaiindians", "paltan", "ipl"],
    "Royal Challengers Bengaluru": ["rcb", "royalchallengersbengaluru", "playbold", "ipl"],
    "Kolkata Knight Riders":    ["kkr", "kolkataknightriders", "korbolorbojeebo", "ipl"],
    "Sunrisers Hyderabad":      ["srh", "sunrisershyderabad", "orangearmy", "ipl"],
    "Delhi Capitals":           ["dc", "delhicapitals", "yehai", "ipl"],
    "Rajasthan Royals":         ["rr", "rajasthanroyals", "hallabol", "ipl"],
    "Punjab Kings":             ["pbks", "punjabkings", "sadda", "ipl"],
    "Lucknow Super Giants":     ["lsg", "lucknowsupergiants", "abapunjab", "ipl"],
    "Gujarat Titans":           ["gt", "gujarattitans", "aavade", "ipl"],
}

PARTY_TAGS = {
    "Bharatiya Janata Party":   ["bjp", "narendramodi", "politics", "india"],
    "Indian National Congress": ["congress", "inc", "rahulgandhi", "politics", "india"],
    "Aam Aadmi Party":          ["aap", "arvindkejriwal", "politics", "india"],
    "Trinamool Congress":       ["tmc", "mamatabanerjee", "bengalpolitics", "politics"],
    "Dravida Munnetra Kazhagam":["dmk", "mkstalin", "tamilnadupolitics", "politics"],
    "Samajwadi Party":          ["sp", "akhileshyadav", "uppolitics", "politics"],
    "Shiv Sena":                ["shivsena", "maharashtrapolitics", "politics"],
    "Bahujan Samaj Party":      ["bsp", "mayawati", "dalitpolitics", "politics"],
}


# ─────────────────────────────────────────────
# 1. Cricketers (richer tags)
# ─────────────────────────────────────────────

def fetch_cricketers():
    print("[1/10] Fetching Indian cricketers...")
    query = """
    SELECT DISTINCT ?player ?playerLabel ?teamLabel ?wikidataId WHERE {
      ?player wdt:P106 wd:Q12299841 .
      ?player wdt:P27  wd:Q668 .
      OPTIONAL { ?player wdt:P54 ?team . }
      BIND(STRAFTER(STR(?player), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 1000
    """
    rows = run_sparql(query)
    count = 0
    for row in rows:
        name = val(row, "playerLabel")
        wid  = val(row, "wikidataId")
        team = val(row, "teamLabel")
        if not name or name.startswith("Q"):
            continue
        upsert_entity(make_entity(wid, name, "Sports", "Cricket"))
        tags = ["cricket", "indiancricket", "bcci", "teamindia"]
        # add IPL team tag if we recognise it
        for ipl_name, ipl_tags in IPL_TEAM_TAGS.items():
            if team and ipl_name.lower() in team.lower():
                tags += ipl_tags
                break
        # add name-based hashtag (e.g. "viratkohli")
        name_tag = name.lower().replace(" ", "")
        tags.append(name_tag)
        upsert_tags(wid, list(set(tags)))
        if team:
            upsert_relationship(wid, "PLAYS_FOR", team)
        count += 1
    log_fetch("cricketers", added=count, updated=0, status="success")
    print(f"  ✓ {count} cricketers")
    return count


# ─────────────────────────────────────────────
# 2. Politicians (richer tags)
# ─────────────────────────────────────────────

def fetch_politicians():
    print("[2/10] Fetching Indian politicians...")
    query = """
    SELECT DISTINCT ?person ?personLabel ?partyLabel ?positionLabel ?wikidataId WHERE {
      {
        ?person wdt:P106 wd:Q82955 .
      } UNION {
        ?person wdt:P106 wd:Q1353186 .
      }
      ?person wdt:P27 wd:Q668 .
      OPTIONAL { ?person wdt:P102 ?party . }
      OPTIONAL { ?person wdt:P39  ?position . }
      BIND(STRAFTER(STR(?person), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 1000
    """
    rows = run_sparql(query)
    count = 0
    for row in rows:
        name     = val(row, "personLabel")
        wid      = val(row, "wikidataId")
        party    = val(row, "partyLabel")
        position = val(row, "positionLabel")
        if not name or name.startswith("Q"):
            continue
        upsert_entity(make_entity(wid, name, "Politics", "Indian Politics"))
        tags = ["politics", "india", "government"]
        # add party-specific tags
        for party_name, party_tags in PARTY_TAGS.items():
            if party and party_name.lower() in party.lower():
                tags += party_tags
                break
        if position and not position.startswith("Q"):
            pos_tag = position.lower().replace(" ", "").replace("of", "")
            tags.append(pos_tag)
        name_tag = name.lower().replace(" ", "")
        tags.append(name_tag)
        upsert_tags(wid, list(set(tags)))
        if party:
            upsert_relationship(wid, "MEMBER_OF", party)
        if position:
            upsert_relationship(wid, "HOLDS_POSITION", position)
        count += 1
    log_fetch("politicians", added=count, updated=0, status="success")
    print(f"  ✓ {count} politicians")
    return count


# ─────────────────────────────────────────────
# 3. Actors (Bollywood / regional cinema)
# ─────────────────────────────────────────────

def fetch_actors():
    print("[3/10] Fetching Indian actors...")
    query = """
    SELECT DISTINCT ?person ?personLabel ?wikidataId ?languageLabel WHERE {
      {
        ?person wdt:P106 wd:Q33999 .
      } UNION {
        ?person wdt:P106 wd:Q10800557 .
      }
      ?person wdt:P27 wd:Q668 .
      OPTIONAL { ?person wdt:P1412 ?language . }
      BIND(STRAFTER(STR(?person), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 1000
    """
    rows = run_sparql(query)

    LANG_MAP = {
        "hindi":  ("Entertainment", "Bollywood",  ["bollywood", "indiancinema", "hindifilm"]),
        "tamil":  ("Entertainment", "Kollywood",  ["kollywood", "tamilcinema", "tamilfilm"]),
        "telugu": ("Entertainment", "Tollywood",  ["tollywood", "telugucinema", "telugufilm"]),
        "malayalam": ("Entertainment", "Mollywood", ["mollywood", "malayalamcinema"]),
        "kannada": ("Entertainment", "Sandalwood", ["sandalwood", "kannadacinema"]),
        "bengali": ("Entertainment", "Bollywood",  ["bengalicinema", "tollywood"]),
    }

    count = 0
    for row in rows:
        name = val(row, "personLabel")
        wid  = val(row, "wikidataId")
        lang = val(row, "languageLabel").lower()
        if not name or name.startswith("Q"):
            continue

        domain, sub, base_tags = "Entertainment", "Indian Cinema", ["indiancinema", "actor"]
        for key, (d, s, t) in LANG_MAP.items():
            if key in lang:
                domain, sub, base_tags = d, s, t
                break

        upsert_entity(make_entity(wid, name, domain, sub))
        name_tag = name.lower().replace(" ", "")
        upsert_tags(wid, list(set(base_tags + [name_tag])))
        count += 1

    log_fetch("actors", added=count, updated=0, status="success")
    print(f"  ✓ {count} actors")
    return count


# ─────────────────────────────────────────────
# 4. Business Figures
# ─────────────────────────────────────────────

def fetch_business():
    print("[4/10] Fetching Indian business figures...")
    query = """
    SELECT DISTINCT ?person ?personLabel ?wikidataId ?employerLabel WHERE {
      {
        ?person wdt:P106 wd:Q43845 .
      } UNION {
        ?person wdt:P106 wd:Q484876 .
      } UNION {
        ?person wdt:P106 wd:Q131524 .
      }
      ?person wdt:P27 wd:Q668 .
      OPTIONAL { ?person wdt:P108 ?employer . }
      BIND(STRAFTER(STR(?person), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 1000
    """
    rows = run_sparql(query)
    count = 0
    for row in rows:
        name     = val(row, "personLabel")
        wid      = val(row, "wikidataId")
        employer = val(row, "employerLabel")
        if not name or name.startswith("Q"):
            continue
        upsert_entity(make_entity(wid, name, "Business", "Indian Business"))
        tags = ["business", "india", "entrepreneur"]
        if employer and not employer.startswith("Q"):
            emp_tag = employer.lower().replace(" ", "")
            tags.append(emp_tag)
        name_tag = name.lower().replace(" ", "")
        tags.append(name_tag)
        upsert_tags(wid, list(set(tags)))
        if employer:
            upsert_relationship(wid, "WORKS_FOR", employer)
        count += 1
    log_fetch("business", added=count, updated=0, status="success")
    print(f"  ✓ {count} business figures")
    return count


# ─────────────────────────────────────────────
# 5. Athletes (non-cricket)
# ─────────────────────────────────────────────

SPORT_TAG_MAP = {
    "football":   ["football", "indianfootball", "aiff"],
    "badminton":  ["badminton", "indianbadminton", "bai"],
    "hockey":     ["hockey", "indianhockey", "fieldhockey"],
    "wrestling":  ["wrestling", "wwe", "indianwrestling"],
    "boxing":     ["boxing", "indianboxing"],
    "chess":      ["chess", "fide", "indianchess"],
    "kabaddi":    ["kabaddi", "pkl", "prokabaddi", "indiankabaddi"],
    "athletics":  ["athletics", "indianathletics"],
    "tennis":     ["tennis", "indiantennis"],
    "shooting":   ["shooting", "indianshooting"],
}

def fetch_athletes():
    print("[5/10] Fetching Indian athletes (non-cricket)...")
    query = """
    SELECT DISTINCT ?person ?personLabel ?sportLabel ?wikidataId WHERE {
      ?person wdt:P106 wd:Q2066131 .
      ?person wdt:P27  wd:Q668 .
      OPTIONAL { ?person wdt:P641 ?sport . }
      BIND(STRAFTER(STR(?person), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 1000
    """
    rows = run_sparql(query)
    count = 0
    for row in rows:
        name  = val(row, "personLabel")
        wid   = val(row, "wikidataId")
        sport = val(row, "sportLabel").lower()
        if not name or name.startswith("Q"):
            continue
        sub = sport.title() if sport and not sport.startswith("q") else "Athletics"
        upsert_entity(make_entity(wid, name, "Sports", sub))
        tags = ["sports", "india", "athlete"]
        for key, sport_tags in SPORT_TAG_MAP.items():
            if key in sport:
                tags += sport_tags
                break
        else:
            if sport and not sport.startswith("q"):
                tags.append(sport.replace(" ", ""))
        name_tag = name.lower().replace(" ", "")
        tags.append(name_tag)
        upsert_tags(wid, list(set(tags)))
        count += 1
    log_fetch("athletes", added=count, updated=0, status="success")
    print(f"  ✓ {count} athletes")
    return count


# ─────────────────────────────────────────────
# 6. Scientists / Academics
# ─────────────────────────────────────────────

def fetch_scientists():
    print("[6/10] Fetching Indian scientists...")
    query = """
    SELECT DISTINCT ?person ?personLabel ?wikidataId ?fieldLabel WHERE {
      {
        ?person wdt:P106 wd:Q901 .
      } UNION {
        ?person wdt:P106 wd:Q170790 .
      } UNION {
        ?person wdt:P106 wd:Q169470 .
      } UNION {
        ?person wdt:P106 wd:Q593644 .
      }
      ?person wdt:P27 wd:Q668 .
      OPTIONAL { ?person wdt:P101 ?field . }
      BIND(STRAFTER(STR(?person), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 500
    """
    rows = run_sparql(query)
    count = 0
    for row in rows:
        name  = val(row, "personLabel")
        wid   = val(row, "wikidataId")
        field = val(row, "fieldLabel")
        if not name or name.startswith("Q"):
            continue
        upsert_entity(make_entity(wid, name, "Science", "Indian Science"))
        tags = ["science", "india", "research", "isro", "education"]
        if field and not field.startswith("Q"):
            tags.append(field.lower().replace(" ", ""))
        upsert_tags(wid, list(set(tags)))
        count += 1
    log_fetch("scientists", added=count, updated=0, status="success")
    print(f"  ✓ {count} scientists")
    return count


# ─────────────────────────────────────────────
# 7. IPL Teams + Cricket boards (ORGANIZATIONS)
# ─────────────────────────────────────────────

def fetch_ipl_teams():
    print("[7/10] Fetching IPL teams and cricket organizations...")
    query = """
    SELECT DISTINCT ?org ?orgLabel ?wikidataId WHERE {
      {
        ?org wdt:P31 wd:Q847017 .          # sports club
        ?org wdt:P17 wd:Q668 .
        ?org wdt:P641 wd:Q1040 .           # cricket
      } UNION {
        ?org wdt:P31 wd:Q4830453 .         # business
        ?org wdt:P17 wd:Q668 .
        ?org wdt:P641 wd:Q1040 .
      }
      BIND(STRAFTER(STR(?org), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 200
    """
    rows = run_sparql(query)
    count = 0
    for row in rows:
        name = val(row, "orgLabel")
        wid  = val(row, "wikidataId")
        if not name or name.startswith("Q"):
            continue
        upsert_entity(make_entity(wid, name, "Sports", "Cricket", entity_type="ORGANIZATION"))
        tags = ["cricket", "ipl", "bcci"]
        # assign team-specific tags
        for team_name, team_tags in IPL_TEAM_TAGS.items():
            if team_name.lower() in name.lower() or name.lower() in team_name.lower():
                tags += team_tags
                break
        org_tag = name.lower().replace(" ", "")
        tags.append(org_tag)
        upsert_tags(wid, list(set(tags)))
        count += 1
    log_fetch("ipl_teams", added=count, updated=0, status="success")
    print(f"  ✓ {count} cricket organizations")
    return count


# ─────────────────────────────────────────────
# 8. Political Parties
# ─────────────────────────────────────────────

def fetch_political_parties():
    print("[8/10] Fetching Indian political parties...")
    query = """
    SELECT DISTINCT ?party ?partyLabel ?wikidataId WHERE {
      ?party wdt:P31 wd:Q7278 .            # political party
      ?party wdt:P17 wd:Q668 .
      BIND(STRAFTER(STR(?party), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 200
    """
    rows = run_sparql(query)
    count = 0
    for row in rows:
        name = val(row, "partyLabel")
        wid  = val(row, "wikidataId")
        if not name or name.startswith("Q"):
            continue
        upsert_entity(make_entity(wid, name, "Politics", "Political Party",
                                  entity_type="ORGANIZATION"))
        tags = ["politics", "india", "elections"]
        for pname, ptags in PARTY_TAGS.items():
            if pname.lower() in name.lower() or name.lower() in pname.lower():
                tags += ptags
                break
        org_tag = name.lower().replace(" ", "")
        tags.append(org_tag)
        upsert_tags(wid, list(set(tags)))
        count += 1
    log_fetch("political_parties", added=count, updated=0, status="success")
    print(f"  ✓ {count} political parties")
    return count


# ─────────────────────────────────────────────
# 9. Indian States + Major Cities (PLACES)
# ─────────────────────────────────────────────

STATE_TAGS = {
    "Maharashtra":   ["maharashtra", "mumbai", "pune"],
    "Tamil Nadu":    ["tamilnadu", "chennai", "tamilpolitics"],
    "Uttar Pradesh": ["uttarpradesh", "lucknow", "upelections"],
    "West Bengal":   ["westbengal", "kolkata", "bengalpolitics"],
    "Karnataka":     ["karnataka", "bengaluru", "bangalorenews"],
    "Kerala":        ["kerala", "thiruvananthapuram", "godsowncountry"],
    "Gujarat":       ["gujarat", "ahmedabad", "vibrantgujarat"],
    "Rajasthan":     ["rajasthan", "jaipur"],
    "Delhi":         ["delhi", "newdelhi", "dilliki"],
    "Punjab":        ["punjab", "chandigarh", "punjabpolitics"],
    "Telangana":     ["telangana", "hyderabad", "tspolities"],
    "Andhra Pradesh":["andhrapradesh", "vijayawada", "apnews"],
    "Bihar":         ["bihar", "patna", "biharpolitics"],
    "Madhya Pradesh":["madhyapradesh", "bhopal", "mppolitics"],
    "Jharkhand":     ["jharkhand", "ranchi"],
    "Odisha":        ["odisha", "bhubaneswar"],
    "Assam":         ["assam", "guwahati", "northeast"],
    "Manipur":       ["manipur", "imphal", "northeast"],
}

def fetch_places():
    print("[9/10] Fetching Indian states and cities...")
    # States
    query_states = """
    SELECT DISTINCT ?place ?placeLabel ?wikidataId WHERE {
      ?place wdt:P31 wd:Q1467840 .         # state of India
      BIND(STRAFTER(STR(?place), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 50
    """
    # Cities
    query_cities = """
    SELECT DISTINCT ?place ?placeLabel ?wikidataId WHERE {
      ?place wdt:P31 wd:Q1549591 .         # big city
      ?place wdt:P17 wd:Q668 .
      BIND(STRAFTER(STR(?place), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 200
    """
    count = 0
    for rows, sub_domain in [(run_sparql(query_states), "Indian State"),
                             (run_sparql(query_cities), "Indian City")]:
        for row in rows:
            name = val(row, "placeLabel")
            wid  = val(row, "wikidataId")
            if not name or name.startswith("Q"):
                continue
            upsert_entity(make_entity(wid, name, "Places", sub_domain,
                                      entity_type="PLACE"))
            tags = ["india", name.lower().replace(" ", "")]
            for state_name, state_tags in STATE_TAGS.items():
                if state_name.lower() in name.lower() or name.lower() in state_name.lower():
                    tags += state_tags
                    break
            upsert_tags(wid, list(set(tags)))
            count += 1
        time.sleep(2)

    log_fetch("places", added=count, updated=0, status="success")
    print(f"  ✓ {count} places")
    return count


# ─────────────────────────────────────────────
# 10. Major Indian Events
# ─────────────────────────────────────────────

def fetch_events():
    print("[10/10] Fetching major Indian events...")
    query = """
    SELECT DISTINCT ?event ?eventLabel ?wikidataId WHERE {
      {
        ?event wdt:P31 wd:Q13406463 .       # sports event
      } UNION {
        ?event wdt:P31 wd:Q40231 .          # election
      } UNION {
        ?event wdt:P31 wd:Q1158803 .        # cricket tournament
      }
      ?event wdt:P17 wd:Q668 .
      BIND(STRAFTER(STR(?event), "entity/") AS ?wikidataId)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 200
    """
    rows = run_sparql(query)
    count = 0
    for row in rows:
        name = val(row, "eventLabel")
        wid  = val(row, "wikidataId")
        if not name or name.startswith("Q"):
            continue
        # classify by name keywords
        name_l = name.lower()
        if any(k in name_l for k in ["ipl", "cricket", "t20", "odi", "test match"]):
            domain, sub = "Sports", "Cricket"
            tags = ["cricket", "ipl", "bcci"]
        elif any(k in name_l for k in ["election", "poll", "lok sabha", "assembly"]):
            domain, sub = "Politics", "Elections"
            tags = ["elections", "india", "democracy", "loksabha"]
        else:
            domain, sub = "Events", "India Event"
            tags = ["india", "event"]
        upsert_entity(make_entity(wid, name, domain, sub, entity_type="EVENT"))
        event_tag = name_l.replace(" ", "")
        tags.append(event_tag)
        upsert_tags(wid, list(set(tags)))
        count += 1
    log_fetch("events", added=count, updated=0, status="success")
    print(f"  ✓ {count} events")
    return count


# ─────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────

def run_round_1():
    """Round 1 — Cricketers + Politicians only"""
    print("=" * 50)
    print("ROUND 1 — Cricketers + Politicians")
    print("=" * 50)
    total = 0
    total += fetch_cricketers(); time.sleep(2)
    total += fetch_politicians()
    print(f"\nRound 1 complete — {total} total entities fetched.")
    return total


def run_round_2():
    """Round 2 — Actors + Business + Athletes + Scientists"""
    print("=" * 50)
    print("ROUND 2 — Actors, Business, Athletes, Scientists")
    print("=" * 50)
    total = 0
    total += fetch_actors();      time.sleep(2)
    total += fetch_business();    time.sleep(2)
    total += fetch_athletes();    time.sleep(2)
    total += fetch_scientists()
    print(f"\nRound 2 complete — {total} total entities fetched.")
    return total


def run_round_3():
    """Round 3 — Organizations, Places, Events"""
    print("=" * 50)
    print("ROUND 3 — IPL Teams, Parties, Places, Events")
    print("=" * 50)
    total = 0
    total += fetch_ipl_teams();          time.sleep(2)
    total += fetch_political_parties();  time.sleep(2)
    total += fetch_places();             time.sleep(2)
    total += fetch_events()
    print(f"\nRound 3 complete — {total} total entities fetched.")
    return total


def run_all():
    """All 10 categories — full knowledge graph"""
    print("=" * 50)
    print("FULL FETCH — All 10 Categories")
    print("=" * 50)
    total = 0
    total += fetch_cricketers();         time.sleep(2)
    total += fetch_politicians();        time.sleep(2)
    total += fetch_actors();             time.sleep(2)
    total += fetch_business();           time.sleep(2)
    total += fetch_athletes();           time.sleep(2)
    total += fetch_scientists();         time.sleep(2)
    total += fetch_ipl_teams();          time.sleep(2)
    total += fetch_political_parties();  time.sleep(2)
    total += fetch_places();             time.sleep(2)
    total += fetch_events()
    print(f"\nFull fetch complete — {total} total entities fetched.")
    return total


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"

    if arg == "all":
        run_all()
    elif arg == "2":
        run_round_2()
    elif arg == "3":
        run_round_3()
    else:
        run_round_1()
