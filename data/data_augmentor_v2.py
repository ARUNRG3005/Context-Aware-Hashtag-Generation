"""
data_augmentor_v2.py — Combinatorial relationship-aware augmentation.
Generates UNIQUE sentence pairs covering ALL 13 domains including
compound relationship labels: #GTvsRR, #IndVsAus, #BJPvsCongress, etc.
"""

import os
import sqlite3
import random
import itertools
import pandas as pd

BASE    = os.path.dirname(os.path.dirname(__file__))
KG_DB   = os.path.join(BASE, "data", "knowledge_base", "india_kg.db")
RAW_DIR = os.path.join(BASE, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

random.seed(42)

# ══════════════════════════════════════════════════════════════════════════
# KG Data Loading
# ══════════════════════════════════════════════════════════════════════════
def load_entities_by_domain():
    conn = sqlite3.connect(KG_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT e.name, e.domain, e.sub_domain, e.entity_type, GROUP_CONCAT(t.tag, '|') as tags
        FROM entities e
        LEFT JOIN tags t ON e.wikidata_id = t.entity_id
        WHERE e.domain IS NOT NULL AND e.name IS NOT NULL AND LENGTH(e.name) > 3
        GROUP BY e.wikidata_id
    """)
    by_domain = {}
    for row in cur.fetchall():
        domain = row["domain"]
        entity = {
            "name":       row["name"],
            "sub_domain": row["sub_domain"] or "General",
            "tags":       list(set((row["tags"] or "").split("|"))) if row["tags"] else [],
        }
        by_domain.setdefault(domain, []).append(entity)
    conn.close()
    return by_domain

def load_ipl_teams():
    conn = sqlite3.connect(KG_DB)
    cur = conn.cursor()
    cur.execute("SELECT name FROM entities WHERE sub_domain = 'Cricket' AND entity_type = 'ORGANIZATION'")
    teams = [r[0] for r in cur.fetchall()]
    conn.close()
    return teams or [
        "Gujarat Titans", "Rajasthan Royals", "Mumbai Indians", "Chennai Super Kings",
        "Royal Challengers Bengaluru", "Kolkata Knight Riders", "Delhi Capitals",
        "Punjab Kings", "Sunrisers Hyderabad", "Lucknow Super Giants"
    ]

def load_political_parties():
    conn = sqlite3.connect(KG_DB)
    cur = conn.cursor()
    cur.execute("SELECT name FROM entities WHERE sub_domain = 'Political Party'")
    parties = [r[0] for r in cur.fetchall()]
    conn.close()
    return parties or ["BJP", "Congress", "AAP", "TMC", "SP", "BSP", "DMK"]

ABBREVIATIONS = {
    "Gujarat Titans": "GT", "Rajasthan Royals": "RR", "Mumbai Indians": "MI",
    "Chennai Super Kings": "CSK", "Royal Challengers Bengaluru": "RCB",
    "Royal Challengers Bangalore": "RCB", "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC", "Punjab Kings": "PBKS",
    "Sunrisers Hyderabad": "SRH", "Lucknow Super Giants": "LSG",
    "Bharatiya Janata Party": "BJP", "Indian National Congress": "Congress",
    "Aam Aadmi Party": "AAP", "Trinamool Congress": "TMC",
    "Samajwadi Party": "SP", "Bahujan Samaj Party": "BSP",
    "Communist Party of India": "CPI", "DMK": "DMK",
}

def abbr(name: str) -> str:
    if name in ABBREVIATIONS:
        return ABBREVIATIONS[name]
    words = name.split()
    ab = "".join(w[0] for w in words if w.isalpha()).upper()
    return ab if len(ab) >= 2 else name[:3].upper()


# ══════════════════════════════════════════════════════════════════════════
# Locations
# ══════════════════════════════════════════════════════════════════════════
CITIES = [
    "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Patna", "Bhopal",
    "Surat", "Kochi", "Chandigarh", "Bhubaneswar", "Guwahati",
    "Dehradun", "Thiruvananthapuram", "Indore", "Nagpur", "Visakhapatnam",
    "Raipur", "Ranchi", "Shimla", "Agra", "Varanasi", "Amritsar",
]

STATES = [
    "Uttar Pradesh", "Maharashtra", "Rajasthan", "Tamil Nadu", "West Bengal",
    "Kerala", "Gujarat", "Madhya Pradesh", "Bihar", "Jharkhand", "Odisha",
    "Karnataka", "Assam", "Punjab", "Haryana", "Uttarakhand",
    "Chhattisgarh", "Telangana", "Andhra Pradesh", "Himachal Pradesh",
]

YEARS  = ["2024", "2025", "2026"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

CRICKET_OPPONENTS = [
    ("Australia", "Aus"), ("England", "Eng"), ("Pakistan", "Pak"),
    ("South Africa", "SA"), ("New Zealand", "NZ"), ("West Indies", "WI"),
    ("Sri Lanka", "SL"), ("Bangladesh", "Ban"), ("Afghanistan", "Afg"),
    ("Zimbabwe", "Zim"),
]

MATCH_TYPES = ["ODI", "T20", "Test", "T20I", "World Cup match",
               "Champions Trophy", "Asia Cup match", "bilateral series"]


# ══════════════════════════════════════════════════════════════════════════
# RELATIONSHIP AUGMENTATION (most important)
# ══════════════════════════════════════════════════════════════════════════
def gen_ipl_matches(ipl_teams: list, n_per_pair: int = 3) -> list:
    """Generate IPL match sentences with compound #T1vsT2 labels."""
    rows = []
    teams = ipl_teams[:10]
    pairs = list(itertools.combinations(teams, 2))
    for t1, t2 in pairs:
        a1, a2 = abbr(t1), abbr(t2)
        compound = f"{a1}vs{a2}"
        labels   = [compound, "IPLMatch", "IPL", "Cricket", "IndianCricket", "India"]
        venues   = random.sample(CITIES, min(n_per_pair, len(CITIES)))
        for city in venues:
            templates = [
                f"{t1} defeated {t2} by 7 wickets in IPL {random.choice(YEARS)} at {city}.",
                f"{t2} beat {t1} in a nail-biting finish in the IPL {random.choice(YEARS)} qualifier at {city}.",
                f"Thrilling IPL clash: {t1} vs {t2} at {city}. Who will win?",
                f"{t1} and {t2} face off in IPL {random.choice(YEARS)} eliminator at {city}.",
                f"Match preview: {t1} takes on {t2} in IPL {random.choice(YEARS)} at {city}.",
            ]
            for text in templates[:n_per_pair]:
                rows.append({
                    "text":   text,
                    "labels": "|".join(sorted(set(labels))),
                    "domain": "Sports",
                    "source": "aug_ipl_match",
                })
    return rows


def gen_india_cricket(n: int = 500) -> list:
    """Generate India vs <country> cricket sentences."""
    rows = []
    for (country, code) in CRICKET_OPPONENTS:
        compound = f"IndVs{code}"
        labels   = [compound, "Cricket", "IndianCricket", "TeamIndia", "India"]
        for mtype in MATCH_TYPES:
            for city in random.sample(CITIES, 3):
                text = f"India beat {country} by 6 wickets in the {mtype} at {city}."
                rows.append({
                    "text":   text,
                    "labels": "|".join(sorted(set(labels + [mtype.replace(" ", "")]))),
                    "domain": "Sports",
                    "source": "aug_india_cricket",
                })
            # Also add losses and draws for balance
            for city in random.sample(CITIES, 2):
                text = f"{country} defeated India in a closely contested {mtype} at {city}."
                rows.append({
                    "text":   text,
                    "labels": "|".join(sorted(set(labels))),
                    "domain": "Sports",
                    "source": "aug_india_cricket",
                })
    return rows


def gen_political_contests(parties: list) -> list:
    """Generate political contest sentences with #P1vsP2 labels."""
    rows = []
    pairs = list(itertools.combinations(parties[:8], 2))
    for p1, p2 in pairs:
        a1, a2   = abbr(p1), abbr(p2)
        compound = f"{a1}vs{a2}"
        labels   = [compound, "Elections", "IndianPolitics", "Democracy", "India"]
        for state in random.sample(STATES, 5):
            city = random.choice(CITIES)
            templates = [
                f"{p1} and {p2} locked in a bitter contest in {state} assembly elections.",
                f"{p1} vs {p2}: {state} elections see fierce campaign rallies.",
                f"{state} polls: {p1} accuses {p2} of misleading voters.",
                f"Exit polls predict close fight between {p1} and {p2} in {state}.",
                f"Election commission issues notice to {p1} and {p2} over {state} campaign violations.",
            ]
            for text in templates:
                rows.append({
                    "text":   text,
                    "labels": "|".join(sorted(set(labels + [f"{state.replace(' ','')}Elections"]))),
                    "domain": "Politics",
                    "source": "aug_political",
                })
    return rows


def gen_business_deals() -> list:
    """Generate merger/acquisition sentences."""
    companies = [
        ("Tata Group", "Air India", "IndianBusiness"), ("Reliance Jio", "BigBasket", "Business"),
        ("Adani Group", "NDTV", "Business"), ("Infosys", "tech startup", "Technology"),
        ("Wipro", "engineering firm", "Technology"), ("Zomato", "Blinkit", "IndianStartup"),
        ("Paytm", "fintech company", "FinTech"), ("HDFC Bank", "HDFC Ltd", "Finance"),
        ("ONGC", "HPCL", "Energy"), ("Coal India", "mining firm", "Business"),
    ]
    rows = []
    for (buyer, target, domain_tag) in companies:
        labels = ["Acquisition", "Merger", "IndianBusiness", "Business", "India", domain_tag]
        templates = [
            f"{buyer} acquires {target} in a landmark ₹10,000 crore deal.",
            f"In a major corporate move, {buyer} completes takeover of {target}.",
            f"{buyer} and {target} announce merger, creating a new business giant in India.",
            f"{buyer} buys stake in {target}, expanding its portfolio in India.",
            f"Regulatory approval received for {buyer}-{target} merger deal.",
        ]
        for text in templates:
            rows.append({
                "text":   text,
                "labels": "|".join(sorted(set(labels))),
                "domain": "Business",
                "source": "aug_business_deal",
            })
    return rows


def gen_defence_relations() -> list:
    """India defence + bilateral relations."""
    countries = [
        ("USA", "UsaIndia"), ("Russia", "IndiaRussia"), ("China", "IndiaChina"),
        ("France", "IndiaFrance"), ("Israel", "IndiaIsrael"), ("Japan", "IndiaJapan"),
    ]
    rows = []
    for (country, tag) in countries:
        labels = [tag, "IndianDefence", "ForeignPolicy", "India"]
        templates = [
            f"India and {country} sign a major defence deal for advanced military equipment.",
            f"Prime Minister Modi meets {country} President during bilateral summit.",
            f"India-{country} trade relations strengthened with new MoU signed.",
            f"India's military cooperation with {country} deepens with joint exercise.",
            f"India imports cutting-edge technology from {country} for defence upgrade.",
        ]
        for text in templates:
            rows.append({
                "text":   text,
                "labels": "|".join(sorted(set(labels))),
                "domain": "Politics",
                "source": "aug_defence_relations",
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════
# WEAK DOMAIN AUGMENTATION (combinatorial, truly unique)
# ══════════════════════════════════════════════════════════════════════════

CRIME_TEMPLATES = [
    "{actor} was arrested by {city} police on charges of {crime}.",
    "A case of {crime} was reported in {city}, police registered FIR.",
    "{city} court sentences accused in {crime} case to {years} years.",
    "Police in {city} bust {crime} racket, {count} arrested.",
    "Victim of {crime} in {city} demands justice from {actor}.",
    "{actor} faces trial for {crime} in {city} sessions court.",
    "High court in {city} orders probe into {crime} allegation.",
    "National Commission for Women condemns {crime} incident in {city}.",
    "Chargesheet filed in {city} {crime} case after {count} months of investigation.",
    "{city} police receive {count} complaints of {crime} in {month}.",
]

CRIMES = [
    ("corruption", ["Corruption", "Crime", "India"]),
    ("sexual assault", ["SexualViolence", "GenderViolence", "Crime", "Justice", "India"]),
    ("murder", ["Murder", "Crime", "LawAndOrder", "India"]),
    ("human trafficking", ["HumanTrafficking", "Crime", "HumanRights", "India"]),
    ("mob lynching", ["MobLynching", "CommunalViolence", "Crime", "India"]),
    ("honour killing", ["HonourKilling", "GenderViolence", "Crime", "India"]),
    ("child abuse", ["ChildAbuse", "Crime", "Justice", "India"]),
    ("dowry violence", ["DowryViolence", "GenderViolence", "Crime", "India"]),
    ("police brutality", ["PoliceBrutality", "HumanRights", "Crime", "India"]),
    ("financial fraud", ["Corruption", "Crime", "India"]),
    ("caste atrocity", ["Casteism", "DalitRights", "CasteViolence", "Crime", "India"]),
    ("domestic violence", ["DomesticViolence", "GenderViolence", "Crime", "India"]),
]

SOCIAL_TEMPLATES = [
    "Dalit community in {city} protests against {issue}.",
    "Farmer agitation in {state} over {issue} enters {count}th day.",
    "Tribal activists in {state} demand enforcement of {issue}.",
    "LGBTQ community in {city} holds rally for {issue}.",
    "{city} sees protest against {issue} by civil society groups.",
    "OBC students in {city} demand {issue} ahead of academic year.",
    "Human rights body flags {issue} in {state}.",
    "NGO report highlights {issue} in {state} districts.",
    "Women's group in {city} campaign against {issue}.",
    "Communal tension in {city} over {issue}, security tightened.",
]

SOCIAL_ISSUES = [
    ("reservation policy", ["Reservation", "SocialJustice", "IndianPolitics"]),
    ("caste discrimination", ["Casteism", "DalitRights", "SocialJustice"]),
    ("MSP for farmers", ["FarmerIssues", "Agriculture", "RuralIndia", "India"]),
    ("forest rights act violation", ["TribalRights", "ForestRights", "Adivasi"]),
    ("gender equality", ["WomensRights", "GenderRights", "India"]),
    ("OBC reservation quota", ["SocialJustice", "Reservation", "IndianPolitics"]),
    ("domestic violence laws", ["DomesticViolence", "GenderViolence", "WomensRights"]),
    ("LGBTQ rights", ["LGBTQ", "GenderRights", "Equality"]),
    ("communal harmony", ["Communalism", "India", "SocialJustice"]),
    ("child marriage ban", ["ChildMarriage", "SocialIssues", "India"]),
]

ENV_TEMPLATES = [
    "Severe {event} in {city} displaces {count} thousand people.",
    "{city} faces {event} as temperatures breach record levels.",
    "Relief operations launched in {state} after {event}.",
    "NGO report warns of worsening {issue} in {state}.",
    "Government announces ₹{count}00 crore package for {event} relief in {state}.",
    "Scientists warn of increasing {event} frequency due to climate change in {state}.",
    "{city} air quality index hits alarming levels due to {issue}.",
    "National disaster response force deployed in {city} for {event}.",
    "Green activists in {city} protest against {issue}.",
    "India pledges at UN summit to tackle {issue} by {year}.",
]

ENV_EVENTS = [
    ("flooding", ["Floods", "NaturalDisaster", "Environment", "India"]),
    ("cyclone", ["Cyclone", "NaturalDisaster", "Environment", "India"]),
    ("heatwave", ["Heatwave", "ClimateChange", "Environment", "India"]),
    ("drought", ["Drought", "Environment", "FarmerIssues", "India"]),
    ("earthquake", ["Earthquake", "NaturalDisaster", "Environment", "India"]),
    ("landslide", ["Landslide", "NaturalDisaster", "Environment", "India"]),
    ("forest fire", ["ForestFire", "NaturalDisaster", "Environment", "India"]),
    ("air pollution crisis", ["AirPollution", "Environment", "Health", "India"]),
    ("deforestation", ["Deforestation", "Environment", "ForestRights", "India"]),
    ("water crisis", ["WaterCrisis", "Environment", "PublicHealth", "India"]),
]

HEALTH_TEMPLATES = [
    "{event} outbreak reported in {city}, {count} cases confirmed.",
    "{city} hospital launches free {treatment} camp for {count} patients.",
    "Health ministry alerts {state} about rising {event} cases.",
    "Study from {city} medical college reveals alarming {issue} data.",
    "AIIMS {city} achieves breakthrough in {treatment} research.",
    "Government scales up {treatment} programme in {state} amid shortage.",
    "{count} doctors from {city} warn about {issue} epidemic risk.",
    "Mental health crisis among youth in {city} raises alarm.",
    "Counterfeit {treatment} drugs seized in {city}, {count} arrested.",
    "WHO commends India's {treatment} campaign, {state} leads nationally.",
]

HEALTH_EVENTS = [
    ("dengue", ["Dengue", "Health", "PublicHealth", "India"]),
    ("COVID-19", ["COVID19", "Health", "Pandemic", "India"]),
    ("tuberculosis", ["Tuberculosis", "Health", "PublicHealth"]),
    ("malnutrition", ["Malnutrition", "PublicHealth", "Health", "India"]),
    ("malaria", ["Malaria", "Health", "PublicHealth", "India"]),
    ("cancer", ["Cancer", "Health", "MedicalResearch"]),
    ("mental health", ["MentalHealth", "Health", "India"]),
    ("diabetes", ["Diabetes", "Health", "India"]),
    ("swine flu", ["Health", "PublicHealth", "Pandemic"]),
    ("air pollution disease", ["AirPollution", "Health", "Environment", "India"]),
]

TREATMENTS = ["vaccination", "telemedicine", "surgery", "immunisation", "cancer screening",
               "dialysis", "chemotherapy", "mental health counselling", "dental care", "eye care"]

EDU_TEMPLATES = [
    "{exam} results declared for {year}, students from {state} excel.",
    "{city} government school launches digital education initiative for {count} students.",
    "CBSE announces {exam} exam schedule for {year}, {count} students to appear.",
    "Coaching centre scam in {city} exposed, {count} students affected.",
    "IIT {city} opens applications for {course} programme for {year} batch.",
    "Scholarship programme launched for {count} meritorious students in {state}.",
    "Teacher shortage in {state} affects {count} government schools.",
    "Students from {city} win national {exam} olympiad, top the country.",
    "{state} education board revises syllabus for {year}, new topics added.",
    "Mid-day meal scheme expanded in {state}, improves enrolment by {count}%.",
]

EXAMS = ["NEET", "JEE Main", "JEE Advanced", "UPSC", "CBSE Board", "State Board",
         "CLAT", "GATE", "CAT", "SSC", "NDA"]
COURSES = ["Artificial Intelligence", "Cybersecurity", "Renewable Energy", "Data Science",
           "Aerospace Engineering", "Biotechnology", "Quantum Computing"]

SCIENCE_TEMPLATES = [
    "ISRO launches {satellite} from Sriharikota, aims to {goal}.",
    "Indian scientists from {city} publish {discovery} research in {journal}.",
    "DRDO unveils {technology} at defence expo, a first for India.",
    "IIT {city} develops {invention} that could transform {sector}.",
    "Chandrayaan {number} mission reveals new data about {discovery}.",
    "India's {satellite} achieves {milestone} in orbit, ISRO celebrates.",
    "Gaganyaan human spaceflight programme on track for {year} launch.",
    "India joins international {project} research project as key partner.",
    "New {discovery} species found in {state} biodiversity hotspot by Indian researchers.",
    "India's quantum computing lab in {city} achieves {milestone} milestone.",
]

SATELLITES = ["GSAT-30", "RISAT-2BR2", "EOS-06", "NavIC satellite", "INSAT-3DS",
              "OneWeb India satellite", "Aditya-L1 solar probe"]
DISCOVERIES = ["genetic", "astronomical", "climate", "medicinal plant", "marine organism",
               "quantum physics", "carbon capture", "nuclear fusion", "AI algorithm"]
JOURNALS = ["Nature", "Science", "The Lancet", "Cell", "Physical Review Letters"]
MILESTONES = ["100-day operation success", "record data transmission", "first orbit insertion",
              "historic soft landing", "breakthrough quantum state"]


def gen_combinatorial(templates, var_lists, base_labels, domain, source, max_rows=3000):
    """Generate unique rows from all combinations of variables."""
    rows = []
    seen = set()
    
    # Build all combinations from provided variable lists
    for combo in itertools.product(*var_lists):
        for template in templates:
            try:
                text = template
                for i, val in enumerate(combo):
                    text = text.replace(f"{{var{i}}}", str(val))
                # Also fill remaining placeholders with random values
                text = text.replace("{city}", random.choice(CITIES))
                text = text.replace("{state}", random.choice(STATES))
                text = text.replace("{year}", random.choice(YEARS))
                text = text.replace("{month}", random.choice(MONTHS))
                text = text.replace("{count}", str(random.randint(2, 500)))
                if text not in seen:
                    seen.add(text)
                    rows.append({
                        "text":   text,
                        "labels": "|".join(sorted(set(base_labels))),
                        "domain": domain,
                        "source": source,
                    })
                    if len(rows) >= max_rows:
                        return rows
            except Exception:
                continue
    return rows


def gen_crime_rows(max_rows: int = 3000) -> list:
    rows = []
    seen = set()
    for template in CRIME_TEMPLATES:
        for (crime, labels) in CRIMES:
            for city in CITIES:
                for actor in ["police", "CBI", "ED", "court", "accused"]:
                    text = template\
                        .replace("{crime}", crime)\
                        .replace("{city}", city)\
                        .replace("{actor}", actor)\
                        .replace("{years}", str(random.randint(2, 14)))\
                        .replace("{count}", str(random.randint(2, 50)))\
                        .replace("{month}", random.choice(MONTHS))
                    if text not in seen:
                        seen.add(text)
                        rows.append({"text": text, "labels": "|".join(sorted(set(labels))),
                                     "domain": "Crime", "source": "aug_crime"})
                    if len(rows) >= max_rows:
                        return rows
    return rows


def gen_social_rows(max_rows: int = 3000) -> list:
    rows = []
    seen = set()
    for template in SOCIAL_TEMPLATES:
        for (issue, labels) in SOCIAL_ISSUES:
            for city in CITIES:
                for state in STATES:
                    text = template\
                        .replace("{issue}", issue)\
                        .replace("{city}", city)\
                        .replace("{state}", state)\
                        .replace("{count}", str(random.randint(5, 200)))
                    if text not in seen:
                        seen.add(text)
                        rows.append({"text": text, "labels": "|".join(sorted(set(labels))),
                                     "domain": "SocialIssues", "source": "aug_social"})
                    if len(rows) >= max_rows:
                        return rows
    return rows


def gen_env_rows(max_rows: int = 3000) -> list:
    rows = []
    seen = set()
    for template in ENV_TEMPLATES:
        for (event, labels) in ENV_EVENTS:
            for city in CITIES:
                for state in STATES:
                    text = template\
                        .replace("{event}", event)\
                        .replace("{issue}", event)\
                        .replace("{city}", city)\
                        .replace("{state}", state)\
                        .replace("{count}", str(random.randint(5, 200)))\
                        .replace("{year}", random.choice(YEARS))
                    if text not in seen:
                        seen.add(text)
                        rows.append({"text": text, "labels": "|".join(sorted(set(labels))),
                                     "domain": "Environment", "source": "aug_env"})
                    if len(rows) >= max_rows:
                        return rows
    return rows


def gen_health_rows(max_rows: int = 3000) -> list:
    rows = []
    seen = set()
    for template in HEALTH_TEMPLATES:
        for (event, labels) in HEALTH_EVENTS:
            for treatment in TREATMENTS:
                for city in CITIES:
                    text = template\
                        .replace("{event}", event)\
                        .replace("{issue}", event)\
                        .replace("{treatment}", treatment)\
                        .replace("{city}", city)\
                        .replace("{state}", random.choice(STATES))\
                        .replace("{count}", str(random.randint(10, 5000)))
                    if text not in seen:
                        seen.add(text)
                        rows.append({"text": text, "labels": "|".join(sorted(set(labels))),
                                     "domain": "Health", "source": "aug_health"})
                    if len(rows) >= max_rows:
                        return rows
    return rows


def gen_education_rows(max_rows: int = 3000) -> list:
    rows = []
    seen = set()
    for template in EDU_TEMPLATES:
        for exam in EXAMS:
            for city in CITIES:
                for state in STATES:
                    labels = ["Education", "India"]
                    if "NEET" in exam: labels += ["NEET", "Health"]
                    elif "JEE" in exam: labels += ["JEE", "Engineering"]
                    elif "UPSC" in exam: labels += ["UPSC", "Government"]
                    elif "CBSE" in exam: labels += ["CBSE"]
                    
                    text = template\
                        .replace("{exam}", exam)\
                        .replace("{city}", city)\
                        .replace("{state}", state)\
                        .replace("{course}", random.choice(COURSES))\
                        .replace("{year}", random.choice(YEARS))\
                        .replace("{count}", str(random.randint(100, 50000)))
                    if text not in seen:
                        seen.add(text)
                        rows.append({"text": text, "labels": "|".join(sorted(set(labels))),
                                     "domain": "Education", "source": "aug_education"})
                    if len(rows) >= max_rows:
                        return rows
    return rows


def gen_science_rows(max_rows: int = 3000) -> list:
    rows = []
    seen = set()
    for template in SCIENCE_TEMPLATES:
        for satellite in SATELLITES:
            for discovery in DISCOVERIES:
                for city in CITIES:
                    labels = ["Science", "Research", "India"]
                    if "ISRO" in template or "Chandrayaan" in template or "Gaganyaan" in template:
                        labels += ["ISRO", "Space", "IndiaInSpace"]
                    if "DRDO" in template:
                        labels += ["DRDO", "Defence"]
                    if "IIT" in template:
                        labels += ["IIT", "Technology"]
                        
                    text = template\
                        .replace("{satellite}", satellite)\
                        .replace("{discovery}", discovery)\
                        .replace("{city}", city)\
                        .replace("{state}", random.choice(STATES))\
                        .replace("{journal}", random.choice(JOURNALS))\
                        .replace("{milestone}", random.choice(MILESTONES))\
                        .replace("{goal}", "improve weather forecasting across India")\
                        .replace("{number}", str(random.randint(3, 5)))\
                        .replace("{project}", "quantum computing")\
                        .replace("{technology}", "hypersonic missile")\
                        .replace("{invention}", "water purification device")\
                        .replace("{sector}", "healthcare")\
                        .replace("{year}", random.choice(YEARS))
                    if text not in seen:
                        seen.add(text)
                        rows.append({"text": text, "labels": "|".join(sorted(set(labels))),
                                     "domain": "Science", "source": "aug_science"})
                    if len(rows) >= max_rows:
                        return rows
    return rows


def main():
    print("=" * 60)
    print("DATA AUGMENTOR V2 — Combinatorial + Relationship Augmentation")
    print("=" * 60)

    entities_by_domain = load_entities_by_domain()
    ipl_teams    = load_ipl_teams()
    parties      = load_political_parties()

    all_rows = []

    # ── Relationship rows (most valuable) ─────────────────────────────────
    ipl_rows = gen_ipl_matches(ipl_teams, n_per_pair=5)
    print(f"  IPL match pairs:        {len(ipl_rows):>6} rows")
    all_rows.extend(ipl_rows)

    cricket_rows = gen_india_cricket()
    print(f"  India cricket matches:  {len(cricket_rows):>6} rows")
    all_rows.extend(cricket_rows)

    political_rows = gen_political_contests(parties)
    print(f"  Political contests:     {len(political_rows):>6} rows")
    all_rows.extend(political_rows)

    business_rows = gen_business_deals()
    print(f"  Business deals:         {len(business_rows):>6} rows")
    all_rows.extend(business_rows)

    defence_rows = gen_defence_relations()
    print(f"  Defence relations:      {len(defence_rows):>6} rows")
    all_rows.extend(defence_rows)

    # ── Weak domain rows ───────────────────────────────────────────────────
    crime_rows = gen_crime_rows(4000)
    print(f"  Crime:                  {len(crime_rows):>6} rows")
    all_rows.extend(crime_rows)

    social_rows = gen_social_rows(4000)
    print(f"  Social issues:          {len(social_rows):>6} rows")
    all_rows.extend(social_rows)

    env_rows = gen_env_rows(4000)
    print(f"  Environment:            {len(env_rows):>6} rows")
    all_rows.extend(env_rows)

    health_rows = gen_health_rows(4000)
    print(f"  Health:                 {len(health_rows):>6} rows")
    all_rows.extend(health_rows)

    edu_rows = gen_education_rows(4000)
    print(f"  Education:              {len(edu_rows):>6} rows")
    all_rows.extend(edu_rows)

    sci_rows = gen_science_rows(4000)
    print(f"  Science/ISRO:           {len(sci_rows):>6} rows")
    all_rows.extend(sci_rows)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["text"])
    out = os.path.join(RAW_DIR, "augmented_v2.csv")
    df.to_csv(out, index=False)
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"\nTotal unique augmented rows: {len(df):,}  ({size_mb:.1f} MB)")
    print("\nDomain breakdown:")
    for domain, count in df["domain"].value_counts().items():
        bar = "█" * (count // 100)
        print(f"  {domain:<20} {count:>6}  {bar}")

    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
