"""
preprocessor_v2.py — Combines all data sources and uses KG for label enrichment
"""

import os
import re
import sqlite3
import pandas as pd
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(__file__))
RAW_DIR      = os.path.join(BASE, "data", "raw")
PROCESSED    = os.path.join(BASE, "data", "processed")
KG_DB        = os.path.join(BASE, "data", "knowledge_base", "india_kg.db")
os.makedirs(PROCESSED, exist_ok=True)

SCRAPED_CSV  = os.path.join(RAW_DIR, "scraped_articles.csv")
AG_TRAIN_CSV = os.path.join(RAW_DIR, "ag_news_train.csv")
AG_TEST_CSV  = os.path.join(RAW_DIR, "ag_news_test.csv")
BBC_CSV      = os.path.join(RAW_DIR, "bbc_news.csv")

# New bulk HF CSVs
HF_YAHOO     = os.path.join(RAW_DIR, "hf_yahoo.csv")
HF_CNN       = os.path.join(RAW_DIR, "hf_cnn_full.csv")
HF_CNN_OLD   = os.path.join(RAW_DIR, "hf_cnn_old.csv")
HF_CNN_DM    = os.path.join(RAW_DIR, "hf_cnn_dailymail.csv")
HF_BBC_ALL   = os.path.join(RAW_DIR, "hf_bbc_alltime.csv")
HF_HUFFPOST  = os.path.join(RAW_DIR, "hf_huffpost.csv")
HF_XSUM      = os.path.join(RAW_DIR, "hf_xsum.csv")
HF_CCNEWS    = os.path.join(RAW_DIR, "hf_cc_news.csv")
HF_MULTINEWS = os.path.join(RAW_DIR, "hf_multinews.csv")
HF_ARGILLA   = os.path.join(RAW_DIR, "hf_argilla_news.csv")
HF_FINANCE   = os.path.join(RAW_DIR, "hf_indian_financial_news.csv")
HF_SENTIMENT = os.path.join(RAW_DIR, "hf_indian_sentiment.csv")
HF_KDAVE     = os.path.join(RAW_DIR, "hf_kdave_finance.csv")

MAX_PER_DOMAIN = 20000

# ── KG Loading for Fast In-Memory Lookup ──────────────────────────────────
KG_ENTITIES = {}  # lower_name -> {tags, domain}

def load_kg():
    print("Loading Knowledge Graph into memory...")
    if not os.path.exists(KG_DB):
        print("  WARNING: india_kg.db not found!")
        return
    conn = sqlite3.connect(KG_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT e.name, e.domain, t.tag 
        FROM entities e 
        LEFT JOIN tags t ON e.wikidata_id = t.entity_id
    """)
    rows = cur.fetchall()
    count = 0
    for row in rows:
        name, domain, tag = row
        lname = name.lower()
        if len(lname) < 4: 
            continue # skip very short names to avoid false positives
        if lname not in KG_ENTITIES:
            KG_ENTITIES[lname] = {"domain": domain, "tags": set()}
            count += 1
        if tag:
            KG_ENTITIES[lname]["tags"].add(tag)
    conn.close()
    print(f"  OK Loaded {count} entities into memory for fast scanning.")

def kg_boost(text: str, base_tags: list) -> tuple:
    """Scans text for KG entities and returns enriched (domain, tags)."""
    tl = text.lower()
    extra_tags = set(base_tags)
    best_domain = None
    
    words = tl.split()
    candidates = set()
    for i, w in enumerate(words):
        clean = w.strip(".,!?;:()")
        if len(clean) >= 4:
            candidates.add(clean)
        if i + 1 < len(words):
            bg = f"{clean} {words[i+1].strip('.,!?;:()')}"
            if len(bg) >= 5:
                candidates.add(bg)
        if i + 2 < len(words):
            tg = f"{clean} {words[i+1].strip('.,!?;:()')} {words[i+2].strip('.,!?;:()')}"
            if len(tg) >= 8:
                candidates.add(tg)
                
    for candidate in candidates:
        if candidate in KG_ENTITIES:
            extra_tags.update(KG_ENTITIES[candidate]["tags"])
            if not best_domain and KG_ENTITIES[candidate]["domain"] and KG_ENTITIES[candidate]["domain"] != "General":
                best_domain = KG_ENTITIES[candidate]["domain"]
                
    return best_domain, list(extra_tags)


# ── Relationship label extraction (mirrors predictor Layer 5) ─────────────
IPL_TEAM_MAP = {
    "gujarat titans": "GT", "gt": "GT",
    "rajasthan royals": "RR", "rr": "RR",
    "mumbai indians": "MI", "mi": "MI",
    "chennai super kings": "CSK", "csk": "CSK",
    "royal challengers bengaluru": "RCB", "royal challengers bangalore": "RCB", "rcb": "RCB",
    "kolkata knight riders": "KKR", "kkr": "KKR",
    "delhi capitals": "DC", "dc": "DC",
    "punjab kings": "PBKS", "pbks": "PBKS",
    "sunrisers hyderabad": "SRH", "srh": "SRH",
    "lucknow super giants": "LSG", "lsg": "LSG",
}

CRICKET_OPP_MAP = {
    "australia": "Aus", "australian": "Aus",
    "england": "Eng", "pakistan": "Pak", "pakistani": "Pak",
    "south africa": "SA", "new zealand": "NZ", "west indies": "WI",
    "sri lanka": "SL", "sri lankan": "SL", "bangladesh": "Ban",
    "afghanistan": "Afg", "zimbabwe": "Zim", "ireland": "Ire",
}

POL_PARTY_MAP = {
    "bjp": "BJP", "bharatiya janata party": "BJP",
    "congress": "INC", "indian national congress": "INC", "inc": "INC",
    "aap": "AAP", "aam aadmi party": "AAP",
    "tmc": "TMC", "trinamool": "TMC",
    "sp": "SP", "samajwadi": "SP",
    "bsp": "BSP", "dmk": "DMK", "nda": "NDA",
}

MATCH_SIGNALS   = ["vs", "versus", "against", "beat", "defeated", "won", "lost",
                   "match", "final", "qualifier", "eliminator", "clash", "face off"]
ELECTION_SIGS   = ["election", "poll", "vote", "campaign", "constituency", "lok sabha",
                   "assembly", "by-election"]
BUSINESS_SIGS   = ["acquires", "buys", "takes over", "merges", "acquisition", "merger",
                   "stake in", "buyout"]


def extract_relationship_labels(text: str, base_labels: list) -> list:
    """
    Extract compound relationship labels and REAL sensitive keywords from text.
    """
    tl   = text.lower()
    extra = list(base_labels)
    has_match    = any(s in tl for s in MATCH_SIGNALS)
    has_election = any(s in tl for s in ELECTION_SIGS)
    has_business = any(s in tl for s in BUSINESS_SIGS)

    # ── IPL team vs IPL team ─────────────────────────────────────────
    found_ipl = []
    for kw, code in IPL_TEAM_MAP.items():
        if kw in tl and code not in found_ipl:
            found_ipl.append(code)
    if len(found_ipl) >= 2 and has_match:
        t1, t2 = found_ipl[0], found_ipl[1]
        extra += [f"{t1}vs{t2}", "IPLMatch", "IPL", "Cricket", "IndianCricket"]

    # ── India vs opponent cricket ────────────────────────────────────
    if ("india" in tl or "team india" in tl) and has_match:
        for kw, code in CRICKET_OPP_MAP.items():
            if kw in tl:
                extra += [f"IndVs{code}", "Cricket", "IndianCricket", "TeamIndia"]

    # ── Political party vs party ─────────────────────────────────────
    found_parties = []
    for kw, code in POL_PARTY_MAP.items():
        if kw in tl and code not in found_parties:
            found_parties.append(code)
    if len(found_parties) >= 2 and has_election:
        p1, p2 = found_parties[0], found_parties[1]
        extra += [f"{p1}vs{p2}", "Elections", "IndianPolitics", "Democracy"]

    # ── Business acquisition ─────────────────────────────────────────
    if has_business and "india" in tl:
        extra += ["Acquisition", "IndianBusiness", "Business"]
        
    # ── REAL SENSITIVE DATA EXTRACTION ───────────────────────────────
    # We pull these directly from the real CNN/BBC/XSum articles
    if "rape" in tl or "sexual assault" in tl:
        extra += ["GenderViolence", "Rape", "Crime", "SocialIssues"]
    if "honour killing" in tl or "honor killing" in tl:
        extra += ["HonourKilling", "GenderViolence", "Crime", "SocialIssues"]
    if "riot" in tl or "communal violence" in tl or "mob lynching" in tl:
        extra += ["Riots", "CommunalViolence", "SocialUnrest", "SocialIssues"]
    if "scam" in tl or "bribe" in tl or "cbi raid" in tl or "enforcement directorate" in tl:
        extra += ["Corruption", "Scam", "Crime"]
    if "dowry death" in tl or "acid attack" in tl:
        extra += ["GenderViolence", "Crime", "SocialIssues"]

    # ── NEW BROAD DOMAINS (Phase 3) ──────────────────────────────────
    if "supreme court" in tl or "high court" in tl or "chief justice" in tl or "verdict" in tl or "pil " in tl:
        extra += ["LawAndJustice", "IndianLaw", "SupremeCourt"]
    if "nhai" in tl or "metro rail" in tl or "highway" in tl or "smart city" in tl or "vande bharat" in tl:
        extra += ["Infrastructure", "Development", "SmartCities"]
    if "indian army" in tl or "border security" in tl or "drdo" in tl or "indian navy" in tl or "air force" in tl:
        extra += ["Defense", "NationalSecurity", "IndianArmy"]
    if "rbi " in tl or "inflation" in tl or "union budget" in tl or "sensex" in tl or "nifty" in tl or "gdp" in tl:
        extra += ["Economy", "IndianEconomy", "StockMarket"]
    if "startup" in tl or "fintech" in tl or "unicorn" in tl or "upi " in tl or "digital payments" in tl:
        extra += ["Startups", "DigitalIndia", "UPI"]
    if "festival" in tl or "diwali" in tl or "holi " in tl or "durga puja" in tl or "heritage" in tl or "literature" in tl:
        extra += ["ArtsAndCulture", "IndianCulture", "Heritage"]
    if "monsoon" in tl or "cyclone" in tl or "heatwave" in tl or "earthquake" in tl or "landslide" in tl:
        extra += ["Disasters", "Weather", "NaturalDisaster"]
    if "foreign policy" in tl or "brics" in tl or "g20" in tl or "bilateral" in tl or "diplomacy" in tl or "jaishankar" in tl:
        extra += ["Geopolitics", "ForeignPolicy", "Diplomacy"]

    return list(set(extra))

# ── Existing Mappings & Keyword Logic (Kept mostly as is) ─────────────────

AG_LABEL_MAP = {
    0: {"domain": "Politics",   "tags": ["WorldNews", "International"]},
    1: {"domain": "Sports",     "tags": ["Sports"]},
    2: {"domain": "Business",   "tags": ["Business", "Economy"]},
    3: {"domain": "Technology", "tags": ["Technology", "Digital"]},
}

AG_KEYWORD_TAGS = {
    "cricket":     ["Cricket", "IndianCricket", "BCCI"],
    "ipl":         ["IPL", "Cricket", "IndianCricket"],
    "modi":        ["BJP", "IndianPolitics", "NarendraModi"],
    "bjp":         ["BJP", "IndianPolitics"],
    "congress":    ["INC", "IndianPolitics"],
    "election":    ["Elections", "IndianPolitics", "Democracy"],
    "parliament":  ["Parliament", "IndianPolitics", "LokSabha"],
    "sensex":      ["StockMarket", "Sensex", "IndianEconomy"],
    "nifty":       ["StockMarket", "Nifty", "IndianEconomy"],
    "isro":        ["ISRO", "Space", "IndiaInSpace"],
    "chandrayaan": ["ISRO", "Space", "IndiaInSpace", "Chandrayaan"],
    "gaganyaan":   ["ISRO", "Space", "IndiaInSpace"],
    "startup":     ["Startup", "IndianStartup", "Entrepreneurship"],
    "reliance":    ["Reliance", "IndianBusiness"],
    "tata":        ["TataGroup", "IndianBusiness"],
    "infosys":     ["Infosys", "IT", "Technology"],
    "wipro":       ["Wipro", "IT", "Technology"],
    "bollywood":   ["Bollywood", "IndianCinema"],
    "covid":       ["COVID19", "Health", "Pandemic"],
    "corona":      ["COVID19", "Health", "Pandemic"],
    "farmer":      ["FarmerIssues", "Agriculture", "RuralIndia"],
    "kashmir":     ["Kashmir", "IndianPolitics"],
    "army":        ["IndianArmy", "Defence", "India"],
    "rupee":       ["IndianEconomy", "Currency", "RBI"],
    "rbi":         ["RBI", "IndianEconomy", "MonetaryPolicy"],
    "india":       ["India"],
    "indian":      ["India"],
}

BBC_SECTION_MAP = {
    "sport":         ("Sports",        ["Sports"]),
    "cricket":       ("Sports",        ["Cricket", "Sports"]),
    "football":      ("Sports",        ["Football", "Sports"]),
    "health":        ("Health",        ["Health", "PublicHealth"]),
    "science":       ("Science",       ["Science", "Research"]),
    "technology":    ("Technology",    ["Technology", "Digital"]),
    "tech":          ("Technology",    ["Technology", "Digital"]),
    "business":      ("Business",      ["Business", "Economy"]),
    "entertainment": ("Entertainment", ["Entertainment"]),
    "arts":          ("Entertainment", ["Entertainment", "Arts"]),
    "culture":       ("Entertainment", ["Entertainment", "Culture"]),
    "film":          ("Entertainment", ["Entertainment", "Cinema"]),
    "music":         ("Entertainment", ["Entertainment", "Music"]),
    "world":         ("Politics",      ["WorldNews", "International"]),
    "politics":      ("Politics",      ["Politics"]),
    "news":          ("Politics",      ["News", "Politics"]),
    "uk":            ("Politics",      ["UK", "WorldNews"]),
    "education":     ("Education",     ["Education"]),
    "environment":   ("Environment",   ["Environment", "ClimateChange"]),
    "climate":       ("Environment",   ["ClimateChange", "Environment"]),
    "nature":        ("Environment",   ["Environment", "Nature"]),
    "weather":       ("Environment",   ["Weather", "Environment"]),
    "india":         ("Politics",      ["India", "IndianNews"]),
    "asia":          ("Politics",      ["Asia", "WorldNews"]),
    "crime":         ("Crime",         ["Crime", "LawAndOrder"]),
    "law":           ("Crime",         ["Law", "Justice"]),
}

DOMAIN_HASHTAG_MAP = {
    ("Sports",       "Cricket"):        ["Cricket", "IndianCricket", "BCCI"],
    ("Sports",       "Football"):       ["Football", "IndianFootball"],
    ("Sports",       "Kabaddi"):        ["Kabaddi", "ProKabaddi"],
    ("Sports",       "Athletics"):      ["Sports", "IndianSports", "Athletics"],
    ("Sports",       "General"):        ["Sports", "IndianSports"],
    ("Politics",     "Elections"):      ["Elections", "IndianPolitics", "Democracy"],
    ("Politics",     "Parliament"):     ["Parliament", "IndianPolitics", "LokSabha"],
    ("Politics",     "CentralGovt"):    ["IndianPolitics", "Government", "India"],
    ("Politics",     "StatePolitics"):  ["StatePolitics", "IndianPolitics"],
    ("Politics",     "General"):        ["IndianPolitics", "Politics", "India"],
    ("Crime",        "SexualViolence"): ["GenderViolence", "Justice", "Crime", "India"],
    ("Crime",        "Murder"):         ["Crime", "LawAndOrder", "India"],
    ("Crime",        "MobViolence"):    ["CommunalViolence", "Crime", "India"],
    ("Crime",        "Corruption"):     ["Corruption", "Crime", "India"],
    ("Crime",        "General"):        ["Crime", "LawAndOrder", "India"],
    ("SocialIssues", "Casteism"):       ["Casteism", "DalitRights", "SocialJustice"],
    ("SocialIssues", "Communalism"):    ["Communalism", "CommunalViolence", "India"],
    ("SocialIssues", "GenderViolence"): ["GenderViolence", "WomensRights", "India"],
    ("SocialIssues", "FarmerIssues"):   ["FarmerIssues", "Agriculture", "RuralIndia"],
    ("SocialIssues", "TribalRights"):   ["TribalRights", "Adivasi", "India"],
    ("SocialIssues", "GenderRights"):   ["LGBTQ", "GenderRights", "Equality"],
    ("SocialIssues", "General"):        ["SocialIssues", "India"],
    ("Business",     "Startup"):        ["Startup", "IndianStartup", "Entrepreneurship"],
    ("Business",     "StockMarket"):    ["StockMarket", "Sensex", "IndianEconomy"],
    ("Business",     "Economy"):        ["IndianEconomy", "RBI", "GDP"],
    ("Business",     "Conglomerates"):  ["IndianBusiness", "Business", "India"],
    ("Business",     "General"):        ["Business", "IndianEconomy", "India"],
    ("Entertainment","Bollywood"):      ["Bollywood", "IndianCinema", "Hindi"],
    ("Entertainment","RegionalCinema"): ["RegionalCinema", "IndianCinema"],
    ("Entertainment","OTT"):            ["OTT", "Streaming", "Entertainment"],
    ("Entertainment","Celebrity"):      ["Bollywood", "Entertainment", "India"],
    ("Entertainment","General"):        ["Entertainment", "IndianCinema"],
    ("Science",      "Space"):          ["ISRO", "Space", "IndiaInSpace"],
    ("Technology",   "AI"):             ["ArtificialIntelligence", "AI", "Technology"],
    ("Technology",   "General"):        ["Technology", "Digital", "India"],
    ("Science",      "General"):        ["Science", "Research", "India"],
    ("Health",       "COVID"):          ["COVID19", "Health", "Pandemic"],
    ("Health",       "MentalHealth"):   ["MentalHealth", "Health", "India"],
    ("Health",       "General"):        ["Health", "PublicHealth", "India"],
    ("Environment",  "Disaster"):       ["NaturalDisaster", "ClimateChange", "India"],
    ("Environment",  "Climate"):        ["ClimateChange", "Environment", "India"],
    ("Education",    "General"):        ["Education", "India", "Students"],
    ("General",      "General"):        [],
}

INDIA_KEYWORDS = [
    "india", "indian", "delhi", "mumbai", "bangalore", "bengaluru",
    "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur",
    "lucknow", "patna", "bhopal", "surat",
    "cricket", "ipl", "bcci", "sachin", "kohli", "dhoni", "rohit",
    "virat", "bumrah", "jadeja", "icc", "ranji", "shubman",
    "modi", "bjp", "congress", "gandhi", "parliament", "lok sabha",
    "rajya sabha", "aap", "kejriwal", "mamata", "yogi", "rahul",
    "rupee", "rbi", "sensex", "nifty", "infosys", "tata", "wipro",
    "reliance", "adani", "ambani", "flipkart", "zomato", "paytm",
    "isro", "chandrayaan", "gaganyaan",
    "dalit", "caste", "reservation", "neet", "jee", "upsc",
    "kabaddi", "pv sindhu", "saina", "neeraj chopra",
    "bollywood", "srk", "salman", "deepika", "alia",
]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s\.\,\!\?\-\']", " ", text)
    return " ".join(text.split()).strip()

def is_india_relevant(text: str) -> bool:
    tl = text.lower()
    return any(kw in tl for kw in INDIA_KEYWORDS)

def keyword_boost(text: str, base_tags: list) -> list:
    tl = text.lower()
    is_india = is_india_relevant(tl)
    extra = []
    for kw, tags in AG_KEYWORD_TAGS.items():
        if kw in tl:
            for tag in tags:
                if tag.startswith("Indian") and not is_india:
                    continue
                extra.append(tag)
    return list(set(base_tags + extra))

# ── Source processors ──────────────────────────────────────────────────────

def process_scraped(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nProcessing {len(df)} scraped articles...")
    rows = []
    skipped = 0
    for _, row in df.iterrows():
        title  = clean_text(str(row.get("title", "")))
        body   = clean_text(str(row.get("body", "")))
        text   = f"{title} {body}".strip()
        if len(text.split()) < 10:
            skipped += 1
            continue
        domain = str(row.get("domain", "General"))
        sub    = str(row.get("sub_domain", "General"))
        tags   = list(DOMAIN_HASHTAG_MAP.get((domain, sub), []))
        
        # KG Boost
        kg_domain, tags = kg_boost(text, tags)
        if kg_domain and domain == "General":
            domain = kg_domain

        # Relationship label extraction
        tags = extract_relationship_labels(text, tags)

        if not tags:
            skipped += 1
            continue
        rows.append({
            "text":   text[:512],
            "labels": "|".join(sorted(set(tags))),
            "domain": domain,
            "source": str(row.get("source", "scraped")),
        })
    print(f"  ✓ {len(rows)} usable rows from scraped articles")
    print(f"  ✗ {skipped} skipped")
    return pd.DataFrame(rows)

def process_ag_news(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    print(f"\nProcessing {len(df)} AG News rows from {source_name}...")
    rows = []
    skipped = 0
    for _, row in df.iterrows():
        text  = clean_text(str(row.get("text", "")))
        label = int(row.get("label", -1))
        if len(text.split()) < 8 or label not in AG_LABEL_MAP:
            skipped += 1
            continue
        if not is_india_relevant(text):
            skipped += 1
            continue
        mapping = AG_LABEL_MAP[label]
        tags    = keyword_boost(text, list(mapping["tags"]))
        
        # KG Boost
        domain = mapping["domain"]
        kg_domain, tags = kg_boost(text, tags)

        # Relationship label extraction
        tags = extract_relationship_labels(text, tags)

        rows.append({
            "text":   text[:512],
            "labels": "|".join(sorted(set(tags))),
            "domain": domain,
            "source": "ag_news",
        })
    print(f"  ✓ {len(rows)} India-relevant rows")
    return pd.DataFrame(rows)

def process_bbc(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nProcessing {len(df)} BBC articles...")
    rows = []
    skipped = 0
    for _, row in df.iterrows():
        title   = clean_text(str(row.get("title", "")))
        desc    = clean_text(str(row.get("description", "")))
        section = str(row.get("section", "")).lower().strip()
        text    = f"{title} {desc}".strip()
        if len(text.split()) < 8:
            skipped += 1
            continue
        domain, tags = "General", []
        for key, (d, t) in BBC_SECTION_MAP.items():
            if key in section:
                domain, tags = d, list(t)
                break
                
        # KG Boost
        kg_domain, tags = kg_boost(text, tags)
        if kg_domain and domain == "General":
            domain = kg_domain

        # Relationship label extraction
        tags = extract_relationship_labels(text, tags)

        if not tags:
            skipped += 1
            continue
        rows.append({
            "text":   text[:512],
            "labels": "|".join(sorted(set(tags))),
            "domain": domain,
            "source": "bbc",
        })
    print(f"  ✓ {len(rows)} usable BBC rows")
    return pd.DataFrame(rows)

def process_augmented(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process augmented weak-domain data (from data_augmentor.py).
    Labels are already set correctly, just clean text and pass through.
    """
    print(f"\nProcessing {len(df)} augmented rows...")
    rows = []
    for _, row in df.iterrows():
        text   = clean_text(str(row.get("text", "")))
        labels = str(row.get("labels", ""))
        domain = str(row.get("domain", "General"))
        source = str(row.get("source", "augmented"))
        if len(text.split()) < 5 or not labels:
            continue
        rows.append({
            "text":   text[:512],
            "labels": labels,
            "domain": domain,
            "source": source,
        })
    print(f"  Kept {len(rows)} augmented rows")
    return pd.DataFrame(rows)


def process_hf_generic(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    print(f"\nProcessing {len(df)} HF Generic rows from {source_name}...")
    rows = []
    skipped = 0
    for _, row in df.iterrows():
        text = clean_text(str(row.get("text", "")))
        if len(text.split()) < 8:
            skipped += 1
            continue
        
        tags = keyword_boost(text, [])
        domain = str(row.get("domain", "General"))
        
        # KG Boost provides domain
        kg_domain, tags = kg_boost(text, tags)
        if kg_domain and domain == "General":
            domain = kg_domain

        # Relationship label extraction & REAL sensitive hunting
        tags = extract_relationship_labels(text, tags)
        
        # Shift domain based on priority tags
        new_domains = ["LawAndJustice", "Infrastructure", "Defense", "Economy", "Startups", "ArtsAndCulture", "Disasters", "Geopolitics", "Crime", "SocialIssues"]
        for nd in new_domains:
            if nd in tags:
                domain = nd
                break
            
        # If still General and tags empty, we can guess or discard
        if not tags and domain == "General":
            skipped += 1
            continue
        
        # Map yahoo generic domains to specific hashtags
        if not tags and domain != "General":
            tags = DOMAIN_HASHTAG_MAP.get((domain, "General"), [domain])
        
        # Ensure #india is present for context
        if "india" not in [t.lower() for t in tags]:
            tags.append("India")

        rows.append({
            "text":   text[:512],
            "labels": "|".join(sorted(set(tags))),
            "domain": domain,
            "source": f"hf_bulk_{source_name}"
        })
    print(f"  ✓ {len(rows)} usable rows extracted")
    return pd.DataFrame(rows)

# ── Balance ────────────────────────────────────────────────────────────────

def balance_dataset(df: pd.DataFrame, max_per_domain: int = MAX_PER_DOMAIN) -> pd.DataFrame:
    print(f"\nBalancing dataset (max {max_per_domain} per domain)...")
    balanced = []
    # If a domain has less, we keep all of it.
    for domain, group in df.groupby("domain"):
        if domain == "General":
            # Keep general small to avoid skewing
            if len(group) > max_per_domain // 2:
                group = group.sample(max_per_domain // 2, random_state=42)
        else:
            if len(group) > max_per_domain:
                group = group.sample(max_per_domain, random_state=42)
        balanced.append(group)
        bar = "█" * (len(group) // 100)
        print(f"  {domain:<20} {len(group):>5} rows  {bar}")
    return pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

# ── Split and save ─────────────────────────────────────────────────────────

def split_and_save(df: pd.DataFrame):
    print(f"\nSplitting {len(df)} total rows (80/10/10)...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train = df.iloc[:int(n * 0.8)]
    val   = df.iloc[int(n * 0.8):int(n * 0.9)]
    test  = df.iloc[int(n * 0.9):]
    cols  = ["text", "labels", "domain", "source"]
    train[cols].to_csv(os.path.join(PROCESSED, "train.csv"), index=False)
    val[cols].to_csv(os.path.join(PROCESSED,   "val.csv"),   index=False)
    test[cols].to_csv(os.path.join(PROCESSED,  "test.csv"),  index=False)
    print(f"  ✓ train.csv : {len(train)} rows")
    print(f"  ✓ val.csv   : {len(val)} rows")
    print(f"  ✓ test.csv  : {len(test)} rows")

    all_tags = []
    for labels in df["labels"]:
        all_tags.extend(str(labels).split("|"))
    tag_counts = Counter(all_tags).most_common(20)
    print("\n=== TOP 20 LABELS ===")
    for tag, count in tag_counts:
        print(f"  #{tag:<30} {count}")

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("PREPROCESSOR V2 — Building Training Dataset with KG Enrichment")
    print("=" * 55)

    load_kg()

    all_dfs = []

    if os.path.exists(SCRAPED_CSV):
        all_dfs.append(process_scraped(pd.read_csv(SCRAPED_CSV)))

    if os.path.exists(AG_TRAIN_CSV):
        all_dfs.append(process_ag_news(pd.read_csv(AG_TRAIN_CSV), "ag_news_train"))

    if os.path.exists(AG_TEST_CSV):
        all_dfs.append(process_ag_news(pd.read_csv(AG_TEST_CSV), "ag_news_test"))

    if os.path.exists(BBC_CSV):
        all_dfs.append(process_bbc(pd.read_csv(BBC_CSV)))

    # Load any HF bulk CSVs that were successfully downloaded
    for hf_path, name in [
        (HF_CNN,      "hf_cnn_full"),
        (HF_CNN_OLD,  "hf_cnn_old"),
        (HF_CNN_DM,   "hf_cnn_dailymail"),
        (HF_BBC_ALL,  "hf_bbc_alltime"),
        (HF_HUFFPOST, "hf_huffpost"),
        (HF_XSUM,     "hf_xsum"),
        (HF_CCNEWS,   "hf_cc_news"),
        (HF_MULTINEWS,"hf_multinews"),
        (HF_ARGILLA,  "hf_argilla"),
        (HF_FINANCE,  "hf_finance_42k"),
        (HF_SENTIMENT,"hf_sentiment"),
    ]:
        if os.path.exists(hf_path):
            all_dfs.append(process_hf_generic(pd.read_csv(hf_path), name))

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal before filtering: {len(df_all)} rows")

    df_all = df_all[df_all["labels"].str.len() > 0]
    df_all = df_all[df_all["text"].str.split().str.len() >= 8]
    df_all = df_all.drop_duplicates(subset=["text"])
    print(f"Total after dedup + filter: {len(df_all)} rows")

    print("\n=== SOURCE BREAKDOWN ===")
    for src, cnt in df_all["source"].value_counts().items():
        print(f"  {src:<30} {cnt}")

    df_balanced = balance_dataset(df_all, max_per_domain=MAX_PER_DOMAIN)

    print(f"\nFinal dataset: {len(df_balanced)} rows")
    print("\n=== FINAL DOMAIN DISTRIBUTION ===")
    for domain, count in sorted(Counter(df_balanced["domain"]).items(), key=lambda x: -x[1]):
        bar = "█" * (count // 100)
        print(f"  {domain:<20} {count:>5}  {bar}")

    split_and_save(df_balanced)
    print("\n✓ Preprocessing complete.")

if __name__ == "__main__":
    main()
