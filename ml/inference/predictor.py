
"""
predictor.py — Context-aware hashtag generation for Indian news
Architecture (5 layers, no retraining needed):

  Layer 1 — ML Model          : RoBERTa domain classification
  Layer 2 — Suppression       : Remove false positives systematically
  Layer 3 — Sensitive Map     : Crime/Social/Environment keywords → precise tags
  Layer 4 — KG Lookup         : SQLite entity database (5086 Indian entities)
  Layer 5 — Relationship Infer: GT+RR+match → #GTvsRR, India+Aus → #IndVsAustralia
"""

import os
import json
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ──────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CHECKPOINT = os.path.join(BASE, "checkpoints", "best_model")
KG_DB      = os.path.join(BASE, "data", "knowledge_base", "india_kg.db")

# ── Config ─────────────────────────────────────────────────────────────────
MAX_LENGTH = 512
THRESHOLD  = 0.3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 — SUPPRESSION RULES
# If NONE of the required keywords are present, zero out those model tags.
# ══════════════════════════════════════════════════════════════════════════

SUPPRESS_RULES = {
    "cricket_tags": {
        "tags": ["Cricket", "IndianCricket", "IPL", "BCCI"],
        "requires_any": [
            "cricket", "ipl", "bcci", " odi", " t20", "test match",
            "kohli", "dhoni", "sachin", "rohit", "bumrah", "jadeja",
            "shubman", "virat", "ranji", "wicket", "innings",
            "century", "bowler", "batsman", "run chase",
        ],
    },
    "space_tags": {
        "tags": ["ISRO", "Space", "IndiaInSpace", "Chandrayaan"],
        "requires_any": [
            "isro", "space", "chandrayaan", "gaganyaan", "rocket",
            "satellite", "launch", "sriharikota", "mangalyaan",
            "orbit", "spacecraft", "mission", "astronaut",
        ],
    },
    "bollywood_tags": {
        "tags": ["Bollywood", "IndianCinema"],
        "requires_any": [
            "bollywood", "film", "movie", "actor", "actress", "cinema",
            "srk", "shah rukh", "salman", "deepika", "alia", "ranveer",
            "director", "box office", "kollywood", "tollywood",
        ],
    },
    "crime_tags": {
        "tags": ["Murder", "SexualViolence", "MobLynching", "HonourKilling"],
        "requires_any": [
            "murder", "killed", "rape", "assault", "lynching", "mob",
            "honour killing", "honor killing", "kidnap", "robbery",
            "stabbed", "shot dead", "brutal", "victim",
        ],
    },
}

# Domestic crime/social signals suppress WorldNews/International
DOMESTIC_CRIME_SIGNALS = [
    "assault", "rape", "murder", "killed", "arrested",
    "lynching", "dalit", "caste", "riot", "corruption",
    "scam", "fraud", "atrocity", "honour killing",
]


# ══════════════════════════════════════════════════════════════════════════
# LAYER 3 — SENSITIVE KEYWORD MAP
# Handles crime, social issues, environment — not in KG by design
# ══════════════════════════════════════════════════════════════════════════

SENSITIVE_MAP = {
    # ── Crime ──────────────────────────────────────────────────────────────
    "sexual assault":    ["SexualViolence", "GenderViolence", "Crime", "Justice"],
    "rape":              ["SexualViolence", "GenderViolence", "Crime", "Justice"],
    "gang rape":         ["SexualViolence", "GenderViolence", "Crime", "Justice"],
    "acid attack":       ["GenderViolence", "Crime", "Justice"],
    "murder":            ["Murder", "Crime", "LawAndOrder"],
    "killed":            ["Crime", "LawAndOrder"],
    "lynching":          ["MobLynching", "CommunalViolence", "Crime"],
    "mob violence":      ["MobLynching", "CommunalViolence", "Crime"],
    "mob lynching":      ["MobLynching", "CommunalViolence", "Crime"],
    "honour killing":    ["HonourKilling", "GenderViolence", "Crime"],
    "honor killing":     ["HonourKilling", "GenderViolence", "Crime"],
    "dowry":             ["DowryViolence", "GenderViolence", "Crime"],
    "dowry death":       ["DowryViolence", "GenderViolence", "Crime", "Justice"],
    "kidnap":            ["Crime", "LawAndOrder"],
    "kidnapping":        ["Crime", "LawAndOrder"],
    "robbery":           ["Crime", "LawAndOrder"],
    "corruption":        ["Corruption", "Crime", "India"],
    "bribery":           ["Corruption", "Crime", "India"],
    "scam":              ["Corruption", "Crime", "India"],
    "fraud":             ["Corruption", "Crime", "India"],
    "arrested":          ["Crime", "LawAndOrder"],
    "fir filed":         ["Crime", "LawAndOrder"],
    "pocso":             ["ChildAbuse", "Crime", "Justice"],
    "child abuse":       ["ChildAbuse", "Crime", "Justice"],
    "child labour":      ["ChildLabour", "HumanRights", "SocialIssues"],
    "child marriage":    ["ChildMarriage", "GenderViolence", "SocialIssues"],
    "human trafficking": ["HumanTrafficking", "Crime", "HumanRights"],
    "police brutality":  ["PoliceBrutality", "HumanRights", "Crime"],
    "custody death":     ["PoliceBrutality", "HumanRights", "Crime"],
    "encounter killing": ["PoliceEncounter", "Crime", "LawAndOrder"],

    # ── Caste & Social Issues ──────────────────────────────────────────────
    "dalit":             ["DalitRights", "Casteism", "SocialJustice"],
    "caste":             ["Casteism", "SocialJustice"],
    "caste discrimination": ["Casteism", "DalitRights", "SocialJustice"],
    "caste violence":    ["CasteViolence", "Casteism", "Crime"],
    "atrocity":          ["DalitRights", "Casteism", "CasteViolence"],
    "untouchability":    ["Casteism", "DalitRights", "SocialJustice"],
    "manual scavenging": ["Casteism", "DalitRights", "HumanRights"],
    "sc/st":             ["DalitRights", "Casteism", "Reservation"],
    "obc":               ["SocialJustice", "IndianPolitics", "Reservation"],
    "reservation":       ["Reservation", "SocialJustice", "IndianPolitics"],
    "communal riot":     ["CommunalViolence", "Communalism", "Crime"],
    "communal":          ["Communalism", "CommunalViolence"],
    "religious violence":["CommunalViolence", "Communalism", "Crime"],
    "minority":          ["MinorityRights", "SocialJustice"],
    "discrimination":    ["Discrimination", "SocialJustice"],
    "farmer protest":    ["FarmerIssues", "Agriculture", "Protest"],
    "farmer agitation":  ["FarmerIssues", "Agriculture", "Protest"],
    "msp":               ["FarmerIssues", "Agriculture", "IndianEconomy"],
    "tribal":            ["TribalRights", "Adivasi"],
    "adivasi":           ["TribalRights", "Adivasi"],
    "forest rights":     ["TribalRights", "ForestRights", "Adivasi"],
    "gender violence":   ["GenderViolence", "WomensRights", "Crime"],
    "domestic violence": ["DomesticViolence", "GenderViolence", "Crime"],
    "women safety":      ["WomenSafety", "GenderViolence", "WomensRights"],
    "lgbtq":             ["LGBTQ", "GenderRights", "Equality"],
    "transgender":       ["LGBTQ", "GenderRights", "Equality"],

    # ── Space / ISRO ───────────────────────────────────────────────────────
    "isro":              ["ISRO", "Space", "IndiaInSpace"],
    "chandrayaan":       ["ISRO", "Space", "IndiaInSpace", "Chandrayaan"],
    "gaganyaan":         ["ISRO", "Space", "IndiaInSpace", "HumanSpaceFlight"],
    "sriharikota":       ["ISRO", "Space", "IndiaInSpace"],
    "mangalyaan":        ["ISRO", "Space", "IndiaInSpace", "MarsOrbiter"],

    # ── Environment ────────────────────────────────────────────────────────
    "heatwave":          ["Heatwave", "ClimateChange", "Environment"],
    "heat wave":         ["Heatwave", "ClimateChange", "Environment"],
    "flood":             ["Floods", "NaturalDisaster", "Environment"],
    "flooding":          ["Floods", "NaturalDisaster", "Environment"],
    "cyclone":           ["Cyclone", "NaturalDisaster", "Environment"],
    "earthquake":        ["Earthquake", "NaturalDisaster", "Environment"],
    "landslide":         ["Landslide", "NaturalDisaster", "Environment"],
    "drought":           ["Drought", "Environment", "FarmerIssues"],
    "air pollution":     ["AirPollution", "Environment", "Health"],
    "climate change":    ["ClimateChange", "Environment"],
    "global warming":    ["ClimateChange", "GlobalWarming", "Environment"],
    "deforestation":     ["Deforestation", "Environment", "ForestRights"],
    "wildfire":          ["Wildfire", "NaturalDisaster", "Environment"],
    "forest fire":       ["ForestFire", "NaturalDisaster", "Environment"],

    # ── Health ─────────────────────────────────────────────────────────────
    "covid":             ["COVID19", "Health", "Pandemic"],
    "coronavirus":       ["COVID19", "Health", "Pandemic"],
    "mental health":     ["MentalHealth", "Health"],
    "suicide":           ["MentalHealth", "Health"],
    "malnutrition":      ["Malnutrition", "Health", "PublicHealth"],
    "tuberculosis":      ["Tuberculosis", "Health", "PublicHealth"],
    "cancer":            ["Cancer", "Health"],
    "diabetes":          ["Diabetes", "Health"],

    # ── Education ──────────────────────────────────────────────────────────
    "neet":              ["NEET", "Education", "Health"],
    "jee":               ["JEE", "Education", "Engineering"],
    "upsc":              ["UPSC", "Education", "Government"],
    "cbse":              ["CBSE", "Education"],

    # ── Law ────────────────────────────────────────────────────────────────
    "supreme court":     ["SupremeCourt", "Judiciary", "LawAndOrder"],
    "high court":        ["HighCourt", "Judiciary", "LawAndOrder"],
    "fir":               ["Crime", "LawAndOrder"],
    "chargesheet":       ["Crime", "Judiciary"],
    "death penalty":     ["Judiciary", "Crime", "Justice"],
}


# ══════════════════════════════════════════════════════════════════════════
# LAYER 4 — KNOWLEDGE GRAPH LOOKUP
# ══════════════════════════════════════════════════════════════════════════

def _query_kg(cur, name: str) -> list:
    """Query KG for entity by name, return its tags."""
    cur.execute("""
        SELECT e.wikidata_id, e.name, e.domain
        FROM entities e
        WHERE LOWER(e.name) = LOWER(?)
        LIMIT 1
    """, (name,))
    entity = cur.fetchone()
    if not entity:
        return []
        
    # Prevent single-word generic first names (like 'Ashish' or 'Ravi') from 
    # matching random athletes/actors in the KG by blocking unigram matches 
    # for Person-heavy domains.
    if len(name.split()) == 1 and entity["domain"] in ["Sports", "Entertainment", "Politics", "Business", "ArtsAndCulture"]:
        return []

    cur.execute("""
        SELECT tag FROM tags
        WHERE entity_id = ?
        ORDER BY weight DESC
    """, (entity["wikidata_id"],))
    return [row["tag"] for row in cur.fetchall() if row["tag"] and len(row["tag"]) < 50]


def kg_lookup(text: str) -> list:
    """
    Full-text KG lookup.
    Scans unigrams + bigrams + trigrams against entity names.
    Also checks uppercase abbreviations (GT, RR, MI, BJP...).
    Returns list of {tag, entity_name, source}.
    """
    if not os.path.exists(KG_DB):
        return []

    results = []
    words = text.split()
    tl = text.lower()

    ABBREV_WHITELIST = {"BCCI", "ISRO", "DRDO", "RBI", "SBI", "LIC", "TCS", "CSK", "RCB", "MI", "GT", "RR", "KKR", "PBKS", "SRH", "DC", "LSG"}

    UNIGRAM_BLACKLIST = {
        "ashish", "satish", "amit", "rahul", "vijay", "raj", "kumar", "sanjay", "anil", "sunil", 
        "rajesh", "ramesh", "mehta", "radhika", "sandeep", "meena", "srinivas", "vivek", "manish", 
        "seema", "anita", "randhawa", "kitty", "mukesh", "ramdev", "jyoti", "kiran", "badshah", "sunita"
    }

    candidates = set()
    upper_words = set()

    for i, w in enumerate(words):
        clean = w.strip(".,!?;:()")

        # Lowercase unigram/bigram/trigram
        lw = clean.lower()
        if len(lw) >= 3 and lw not in UNIGRAM_BLACKLIST:
            candidates.add(lw)
        if i + 1 < len(words):
            bg = f"{lw} {words[i+1].strip('.,!?;:()').lower()}"
            if len(bg) >= 4:
                candidates.add(bg)
        if i + 2 < len(words):
            tg = f"{lw} {words[i+1].strip('.,!?;:()').lower()} {words[i+2].strip('.,!?;:()').lower()}"
            if len(tg) >= 5:
                candidates.add(tg)

        # Uppercase abbreviations (GT, RR, MI, CSK...)
        uw = clean.upper()
        if 2 <= len(uw) <= 5 and uw.isalpha():
            upper_words.add(uw)

    try:
        conn = sqlite3.connect(KG_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Check lowercase phrases
        for candidate in candidates:
            tags = _query_kg(cur, candidate)
            if tags:
                results.append({
                    "entity_name": candidate,
                    "tags":        tags,
                    "source":      "kg"
                })

        # Check uppercase abbreviations
        for uw in upper_words:
            tags = _query_kg(cur, uw)
            if tags:
                results.append({
                    "entity_name": uw,
                    "tags":        tags,
                    "source":      "kg"
                })

        conn.close()

    except Exception:
        pass  # KG failure must never crash prediction

    return results


# ══════════════════════════════════════════════════════════════════════════
# LAYER 5 — RELATIONSHIP INFERENCE
# Detects entity pairs + context → generates combined hashtags
# ══════════════════════════════════════════════════════════════════════════

# Read dynamic entities from KG instead of hardcoding
def get_entities_by_type(entity_type: str, sub_domain: str = None) -> dict:
    if not os.path.exists(KG_DB):
        return {}
    try:
        conn = sqlite3.connect(KG_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if sub_domain:
            cur.execute("SELECT name, wikidata_id FROM entities WHERE entity_type = ? AND sub_domain = ?", (entity_type, sub_domain))
        else:
            cur.execute("SELECT name, wikidata_id FROM entities WHERE entity_type = ?", (entity_type,))
        rows = cur.fetchall()
        # We can map lowercase name to uppercase abbreviation or name itself
        res = {}
        for r in rows:
            name = r["name"].lower()
            res[name] = r["name"]
            # Generate common abbreviations (e.g., "chennai super kings" -> "CSK")
            abbr = "".join([w[0] for w in name.split() if w.isalpha()]).upper()
            if len(abbr) >= 2:
                res[abbr.lower()] = abbr
        conn.close()
        return res
    except Exception:
        return {}

IPL_TEAMS = get_entities_by_type("ORGANIZATION", "Cricket")
POLITICAL_PARTIES = get_entities_by_type("ORGANIZATION", "Political Party")
CRICKET_OPPONENTS = {
    "australia": "Aus", "australian": "Aus",
    "england": "Eng", "english": "Eng",
    "pakistan": "Pak", "pakistani": "Pak",
    "south africa": "SA", "new zealand": "NZ",
    "west indies": "WI", "sri lanka": "SL", "sri lankan": "SL",
    "bangladesh": "Ban", "bangladeshi": "Ban",
    "afghanistan": "Afg", "afghan": "Afg",
    "zimbabwe": "Zim", "ireland": "Ire",
}

MATCH_SIGNALS = [
    "vs", "versus", "against", "beat", "defeated", "won",
    "lost", "match", "final", "qualifier", "eliminator",
    "semi-final", "semifinal", "clash", "face", "faces",
    "takes on", "battle", "fixture", "encounter",
]

ELECTION_SIGNALS = [
    "election", "poll", "vote", "campaign",
    "constituency", "assembly", "lok sabha",
    "by-election", "bypolls", "byelection",
]

CRICKET_SIGNALS = [
    "wicket", "odi", "t20", "test", "cricket",
    "over", "innings", "run", "century", "ipl",
    "bowl", "bat", "fielding", "powerplay",
]


def infer_relationships(text: str) -> list:
    """
    Detect entity pairs + context → generate combined hashtags.

    Rules:
      IPL team + IPL team + match signal  → #T1vsT2 #IPLMatch #IPL2026
      India + opponent + cricket signal   → #IndVsAus #IndianCricket
      Party + party + election signal     → #BJPvsCongress #Elections
    """
    tl = text.lower()
    generated = []

    has_match    = any(s in tl for s in MATCH_SIGNALS)
    has_election = any(s in tl for s in ELECTION_SIGNALS)
    has_cricket  = any(s in tl for s in CRICKET_SIGNALS)

    # ── IPL team vs IPL team ───────────────────────────────────────────────
    found_ipl = []
    for keyword, abbr in IPL_TEAMS.items():
        if keyword in tl and abbr not in found_ipl:
            found_ipl.append(abbr)

    if len(found_ipl) >= 2 and has_match:
        t1, t2 = found_ipl[0], found_ipl[1]
        generated += [f"{t1}vs{t2}", "IPLMatch", "IPL2026", "Cricket", "IndianCricket"]

    # ── India vs opponent (cricket) ────────────────────────────────────────
    india_present = "india" in tl or "team india" in tl
    if india_present and has_match:
        for keyword, abbr in CRICKET_OPPONENTS.items():
            if keyword in tl:
                generated.append(f"IndVs{abbr}")
                if has_cricket or has_ipl_or_cricket_team(tl):
                    generated += ["Cricket", "IndianCricket", "TeamIndia"]

    # ── Political party vs party (election context) ────────────────────────
    found_parties = []
    for keyword, abbr in POLITICAL_PARTIES.items():
        if keyword in tl and abbr not in found_parties:
            found_parties.append(abbr)

    if len(found_parties) >= 2 and has_election:
        p1, p2 = found_parties[0], found_parties[1]
        generated += [f"{p1}vs{p2}", "Elections", "IndianPolitics"]

    return list(set(generated))


def has_ipl_or_cricket_team(tl: str) -> bool:
    return any(k in tl for k in IPL_TEAMS) or any(k in tl for k in CRICKET_OPPONENTS)


# ══════════════════════════════════════════════════════════════════════════
# MAIN PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

class HashtagPredictor:

    def __init__(self):
        print(f"Loading model from {CHECKPOINT}...")
        with open(os.path.join(CHECKPOINT, "label_classes.json")) as f:
            self.labels = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.model     = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
        self.model.to(DEVICE)
        self.model.eval()
        print(f"  ✓ Model loaded ({len(self.labels)} labels, device={DEVICE})")
        kg_status = "✓ Connected" if os.path.exists(KG_DB) else "✗ Not found"
        print(f"  Knowledge Graph: {kg_status}")

    def predict(self, text: str, threshold: float = THRESHOLD, top_k: int = 10):
        """
        Full 5-layer prediction pipeline.
        Returns list of {hashtag, confidence, source} dicts.
        """
        if not text or not text.strip():
            return []

        # ── Layer 1: ML Model ──────────────────────────────────────────────
        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(
                input_ids=encoding["input_ids"].to(DEVICE),
                attention_mask=encoding["attention_mask"].to(DEVICE),
            ).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # label → {confidence, source}
        lp = {
            self.labels[i]: {"confidence": float(probs[i]), "source": "model"}
            for i in range(len(self.labels))
        }

        tl = text.lower()

        # ── Layer 2: Suppression ───────────────────────────────────────────
        for rule in SUPPRESS_RULES.values():
            if not any(kw in tl for kw in rule["requires_any"]):
                for tag in rule["tags"]:
                    if tag in lp:
                        lp[tag]["confidence"] = 0.0

        if any(s in tl for s in DOMESTIC_CRIME_SIGNALS):
            for tag in ["WorldNews", "International"]:
                if tag in lp:
                    lp[tag]["confidence"] = 0.0

        import re
        # ── Layer 3: Sensitive keyword map ────────────────────────────────
        for keyword, tags in SENSITIVE_MAP.items():
            if re.search(rf'\b{re.escape(keyword)}\b', tl):
                for tag in tags:
                    if tag in lp:
                        lp[tag]["confidence"] = max(lp[tag]["confidence"], 0.88)
                        lp[tag]["source"]     = "sensitive_map"
                    else:
                        lp[tag] = {"confidence": 0.88, "source": "sensitive_map"}

        # ── Layer 4: KG lookup ─────────────────────────────────────────────
        kg_results = kg_lookup(text)
        for result in kg_results:
            for tag in result["tags"]:
                if tag in lp:
                    lp[tag]["confidence"] = max(lp[tag]["confidence"], 0.85)
                    lp[tag]["source"]     = "model+kg" if lp[tag]["source"] == "model" else lp[tag]["source"]
                else:
                    lp[tag] = {"confidence": 0.82, "source": "kg"}

        # ── Layer 5: Relationship inference ───────────────────────────────
        relation_results = list(infer_relationships(text))
        for tag in relation_results:
            if tag in lp:
                lp[tag]["confidence"] = max(lp[tag]["confidence"], 0.85)
                lp[tag]["source"]     = "relationship"
            else:
                lp[tag] = {"confidence": 0.85, "source": "relationship"}

        # ── Domain-specific thresholds ─────────────────────────────────────────────
        # Domains with less training data get lower thresholds to improve recall
        DOMAIN_THRESHOLDS = {
            "Crime": 0.20,
            "Environment": 0.20,
            "SocialIssues": 0.20,
            "Education": 0.25,
            "Health": 0.25,
            "Science": 0.25,
            "Sports": 0.35,
            "Politics": 0.35,
            "Business": 0.35,
            "Entertainment": 0.35,
        }

        source_order = {
            "model+kg":      0,
            "relationship":  1,
            "kg":            2,
            "sensitive_map": 3,
            "model":         4,
        }

        # Determine dominant domain based on ML layer (Layer 1)
        # Using a simple heuristic: take the domain of the most confident label
        # Since labels are hashtags, we approximate.
        # Alternatively, we just apply the threshold per hashtag if we can map it to a domain.
        # Since mapping hashtag->domain is lossy here, we'll just check if any sensitive/kg tags suggest a weak domain.
        # Let's apply a base threshold, and lower it if we detect certain keywords.
        # Actually, since we want per-domain thresholds, we can just use 0.25 as default and 0.20 for weak domains.
        
        results = []
        sources_map = {}
        
        for tag, info in lp.items():
            conf = info["confidence"]
            src = info["source"]
            
            if conf >= threshold or src in ("kg", "model+kg", "relationship", "sensitive_map"):
                # Add explanation reasoning
                reason = src
                if src == "kg" or src == "model+kg":
                    # find which entity matched
                    for r in kg_results:
                        if tag in r["tags"]:
                            reason = f"kg({r['entity_name']} → {tag})"
                            break
                elif src == "sensitive_map":
                    reason = f"sensitive_map(pattern match)"
                
                sources_map[f"#{tag}"] = f"{src}({conf:.2f})" if src == "model" else reason
                
                results.append({
                    "hashtag":    f"#{tag}",
                    "confidence": round(conf, 3),
                    "source":     reason,
                })

        results.sort(key=lambda x: (
            source_order.get(x["source"].split("(")[0].lower(), 9),
            -x["confidence"]
        ))
        
        # Build Pipeline Audit Log
        audit_log = []
        base_predicted = len([k for k, v in lp.items() if v["confidence"] >= threshold])
        audit_log.append(f"[ML Engine] Analyzed {len(text.split())} words using RoBERTa. Predicted {base_predicted} base concepts.")
        if kg_results:
            audit_log.append(f"[Knowledge Graph] Scanned against 5,000+ entities. Confirmed {len(kg_results)} hard matches.")
        else:
            audit_log.append(f"[Knowledge Graph] Scanned against 5,000+ entities. No strict matches found.")
            
        if relation_results:
            audit_log.append(f"[Relationship Engine] Generated {len(relation_results)} compound context tags.")
            
        audit_log.append(f"[Final] Ready. Routing {min(top_k, len(results))} tags to UI.")

        return {"hashtags": results[:top_k], "sources": sources_map, "audit_log": audit_log}


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = HashtagPredictor()

    tests = [
        # Relationship inference
        "GT defeated RR by 7 wickets in IPL 2026 Qualifier 2.",
        "India beat Australia by 6 wickets in the third ODI at Mumbai.",
        "BJP and Congress clash in UP assembly elections 2026.",
        # KG entities
        "Virat Kohli scored a brilliant century as India beat Australia in the third ODI.",
        "Narendra Modi inaugurates new Parliament building in New Delhi.",
        # Sensitive
        "Sexual assault cases against women in UP have risen by 30 percent.",
        "Dalit man beaten by upper caste villagers in Rajasthan.",
        "Honour killing of inter-caste couple shocks Haryana.",
        # ISRO
        "ISRO successfully launched Chandrayaan-3 from Sriharikota.",
        # Business
        "Reliance Jio announces 5G rollout across 50 Indian cities by December.",
        "Zomato reports record quarterly revenue as food delivery surges.",
        # Environment
        "India faces heatwave as temperatures cross 45 degrees in Delhi and UP.",
        # Law
        "The Supreme Court ruled on the reservation policy for OBC communities.",
        # Sports
        "Hyd set to emerge as India’s 2nd largest data centre hub after Mumbai with 1.9GW pipeline",
    ]

    print()
    for text in tests:
        res = predictor.predict(text)
        tags = res["hashtags"]
        print(f"Input : {text}")
        print(f"Output: {' '.join(t['hashtag'] for t in tags)}")
        print(f"Sources: {json.dumps(res['sources'], indent=2)}")
        print()