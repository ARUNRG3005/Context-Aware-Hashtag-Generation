"""
data_augmentor.py — Generate synthetic labeled training samples for weak domains
using entity metadata from the KG and templated sentences.
No external downloads needed. All data comes from india_kg.db.
Target: ~2000-3000 rows per weak domain.
"""

import os
import sqlite3
import random
import pandas as pd

BASE    = os.path.dirname(os.path.dirname(__file__))
KG_DB   = os.path.join(BASE, "data", "knowledge_base", "india_kg.db")
RAW_DIR = os.path.join(BASE, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

random.seed(42)

# ── Load KG entities by domain/sub_domain ─────────────────────────────────
def load_entities_by_domain():
    conn = sqlite3.connect(KG_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT e.name, e.domain, e.sub_domain, e.entity_type, GROUP_CONCAT(t.tag, '|') as tags
        FROM entities e
        LEFT JOIN tags t ON e.wikidata_id = t.entity_id
        WHERE e.domain IS NOT NULL AND e.name IS NOT NULL
        GROUP BY e.wikidata_id
    """)
    rows = cur.fetchall()
    conn.close()
    by_domain = {}
    for row in rows:
        domain = row["domain"]
        entity = {
            "name":       row["name"],
            "sub_domain": row["sub_domain"] or "General",
            "tags":       list(set((row["tags"] or "").split("|"))) if row["tags"] else [],
        }
        by_domain.setdefault(domain, []).append(entity)
    return by_domain


# ── Templates per domain ──────────────────────────────────────────────────
CRIME_TEMPLATES = [
    "{name} was arrested by police on charges of corruption and financial fraud.",
    "A case of murder was reported involving {name} in {city}.",
    "A mob lynching incident occurred in {city}, leaving residents shocked.",
    "Police filed an FIR against {name} following allegations of sexual assault.",
    "A case of honour killing was reported in {city}, prompting calls for justice.",
    "Dalit man was assaulted by upper caste group in {city}, prompting protests.",
    "The Supreme Court heard a case of custodial death in {city} and ordered inquiry.",
    "POCSO case registered against accused in {city} for child abuse.",
    "Human trafficking racket busted by {name} police, several arrested.",
    "A bribery scam was uncovered involving officials in {city}.",
    "Robbery at gunpoint reported in {city} market area, police investigating.",
    "Gang rape victim in {city} demands justice as case goes to high court.",
    "{name} police cracked down on organised crime network operating in {city}.",
    "Dowry death case registered in {city}, husband and in-laws arrested.",
    "Child labour found at factory in {city}, authorities take action.",
]

SOCIAL_TEMPLATES = [
    "Dalits in {city} protest against caste discrimination and demand equality.",
    "Farmers from {city} gathered in Delhi to demand MSP and loan waivers.",
    "Tribal community in {region} fights to protect forest rights under FRA.",
    "LGBTQ activists in {city} held pride march demanding gender equality.",
    "Manual scavenging still practiced in {city} despite legal ban, activists say.",
    "Communal tensions rise in {city} following a property dispute.",
    "Women's rights group in {city} demands stronger laws against domestic violence.",
    "OBC reservation quota dispute sparks protests in {city} colleges.",
    "SC/ST communities in {region} allege atrocities under caste system.",
    "Child marriage reported in {city} district, officials take cognisance.",
    "Reservation policy debate heats up ahead of elections in {city}.",
    "Transgender community in {city} denied basic rights, activists demand change.",
    "Religious violence erupts in {city} during festival, curfew imposed.",
    "Adivasi land rights dispute in {region} escalates, tribals stage sit-in.",
    "Women safety concerns rise in {city} after late night incidents reported.",
]

ENV_TEMPLATES = [
    "Severe flooding in {city} displaces thousands as rivers overflow their banks.",
    "Air pollution in {city} reaches hazardous levels ahead of winter season.",
    "Heatwave grips {city} as temperatures soar past 45 degrees Celsius.",
    "Cyclone warning issued for coastal {region} as storm approaches.",
    "Drought conditions in {region} threaten crop failure, farmers worried.",
    "Earthquake tremors felt in {city} of magnitude 4.2 on the Richter scale.",
    "Forest fire destroys thousands of acres in {region} wildlife sanctuary.",
    "Landslide in {region} blocks national highway, rescue ops underway.",
    "Climate change activists in {city} march demanding India net zero target.",
    "Deforestation in {region} reaches alarming levels, NGOs raise alarm.",
    "River pollution in {city} causes fish kill, residents protest.",
    "Wildfire in {region} threatens wildlife habitat and local villages.",
    "India faces worst water crisis in years, {city} put on alert.",
    "Green energy push: {city} installs solar panels across government buildings.",
    "Plastic pollution on {city} beaches sparks cleanup drive.",
]

HEALTH_TEMPLATES = [
    "COVID-19 cases rise again in {city} as new variant spreads.",
    "Mental health crisis among students in {city} colleges alarming experts.",
    "Tuberculosis cases surge in {region} due to poor healthcare access.",
    "Dengue outbreak reported in {city}, civic bodies on high alert.",
    "Free cancer screening camp held in {city} for underprivileged patients.",
    "Malnutrition among children in {region} remains a serious concern.",
    "Diabetes prevalence increasing rapidly among urban population in {city}.",
    "Suicide rate among farmers in {region} draws national attention.",
    "Government launches mental health helpline for students in {city}.",
    "AIIMS {city} reports breakthrough in cancer treatment research.",
    "Counterfeit drugs seized in {city}, health department orders probe.",
    "Infant mortality rate improves in {region} thanks to immunisation drive.",
    "Vaccination camp held in {city} to combat disease spread.",
    "Hospital in {city} launches telemedicine service for rural patients.",
    "Air quality impact on respiratory health studied in {city} by researchers.",
]

EDUCATION_TEMPLATES = [
    "NEET exam controversy sparks protests in {city} and across the country.",
    "JEE Main results declared, students from {city} top the merit list.",
    "UPSC results announced, candidates from {region} excel in civil services.",
    "CBSE board exam date sheet released for 2025, students in {city} prepare.",
    "IIT {city} launches new AI course for engineering students.",
    "Dropout rate among girls in {region} schools concerns education dept.",
    "Digital education initiative launched in {city} rural government schools.",
    "Students from {city} win national science olympiad competition.",
    "Teacher shortage crisis in {region} government schools flagged to ministry.",
    "Education budget increased for {city} district, new schools to open.",
    "Scholarship scheme launched for meritorious students in {city}.",
    "School dropout drive in {city} reconnects 500 children with classrooms.",
    "University in {city} launches new course in renewable energy management.",
    "Coaching centre scam exposed in {city}, hundreds of students affected.",
    "Mid-day meal scheme expanded in {region}, improves attendance rates.",
]

SCIENCE_TEMPLATES = [
    "ISRO launches new satellite from Sriharikota to improve weather forecasting.",
    "Chandrayaan mission data reveals new insights about lunar surface composition.",
    "Gaganyaan human spaceflight program on track for 2025 launch, says ISRO.",
    "Indian scientists discover new species in {region} biodiversity hotspot.",
    "IIT {city} researchers develop low-cost water purification technology.",
    "DRDO unveils new defence technology in collaboration with Indian industry.",
    "India's first quantum computing research lab inaugurated in {city}.",
    "Scientists from {city} university publish research in Nature journal.",
    "Nuclear fusion research milestone achieved at Indian research facility.",
    "Genome sequencing project maps genetic diversity of Indian population.",
    "AI research from {city} IIT wins international award at global conference.",
    "New drug formulation developed by Indian researchers shows promise in trials.",
    "India joins international space telescope project as key partner.",
    "Mars mission data from Mangalyaan reveals new atmospheric findings.",
    "Renewable energy breakthrough achieved by {city} research institution.",
]

CITIES = [
    "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Patna", "Bhopal",
    "Surat", "Kochi", "Chandigarh", "Bhubaneswar", "Guwahati", "Dehradun"
]

REGIONS = [
    "Uttar Pradesh", "Maharashtra", "Rajasthan", "Tamil Nadu", "West Bengal",
    "Kerala", "Gujarat", "Madhya Pradesh", "Bihar", "Jharkhand", "Odisha",
    "Karnataka", "Assam", "Punjab", "Haryana", "Uttarakhand", "Chhattisgarh"
]

DOMAIN_CONFIGS = {
    "Crime": {
        "templates":  CRIME_TEMPLATES,
        "labels_map": {
            "murder":    ["Murder", "Crime", "LawAndOrder", "India"],
            "arrested":  ["Crime", "LawAndOrder", "India"],
            "assault":   ["Crime", "GenderViolence", "Justice", "India"],
            "rape":      ["SexualViolence", "GenderViolence", "Crime", "Justice", "India"],
            "dalit":     ["DalitRights", "Casteism", "Crime", "SocialJustice"],
            "lynch":     ["MobLynching", "CommunalViolence", "Crime", "India"],
            "honour":    ["HonourKilling", "GenderViolence", "Crime", "India"],
            "pocso":     ["ChildAbuse", "Crime", "Justice", "India"],
            "traffick":  ["HumanTrafficking", "Crime", "HumanRights", "India"],
            "bribery":   ["Corruption", "Crime", "India"],
            "robbery":   ["Crime", "LawAndOrder", "India"],
            "fraud":     ["Corruption", "Crime", "India"],
            "dowry":     ["DowryViolence", "GenderViolence", "Crime", "India"],
            "child":     ["ChildLabour", "Crime", "HumanRights", "India"],
        },
        "default_labels": ["Crime", "LawAndOrder", "India"],
        "count": 2500,
    },
    "SocialIssues": {
        "templates":  SOCIAL_TEMPLATES,
        "labels_map": {
            "dalit":      ["DalitRights", "Casteism", "SocialJustice"],
            "farmer":     ["FarmerIssues", "Agriculture", "Protest", "India"],
            "tribal":     ["TribalRights", "Adivasi", "India"],
            "lgbtq":      ["LGBTQ", "GenderRights", "Equality"],
            "manual":     ["Casteism", "DalitRights", "HumanRights"],
            "communal":   ["Communalism", "CommunalViolence", "India"],
            "women":      ["WomensRights", "GenderViolence", "India"],
            "obc":        ["SocialJustice", "Reservation", "IndianPolitics"],
            "sc/st":      ["DalitRights", "Casteism", "Reservation"],
            "child":      ["ChildMarriage", "GenderViolence", "SocialIssues"],
            "reservation":["Reservation", "SocialJustice", "IndianPolitics"],
            "transgender": ["LGBTQ", "GenderRights", "Equality"],
            "religious":  ["Communalism", "CommunalViolence", "India"],
            "adivasi":    ["TribalRights", "Adivasi", "India"],
            "domestic":   ["DomesticViolence", "GenderViolence", "Crime"],
        },
        "default_labels": ["SocialIssues", "India"],
        "count": 2500,
    },
    "Environment": {
        "templates":  ENV_TEMPLATES,
        "labels_map": {
            "flood":     ["Floods", "NaturalDisaster", "Environment", "India"],
            "pollution": ["AirPollution", "Environment", "Health", "India"],
            "heatwave":  ["Heatwave", "ClimateChange", "Environment", "India"],
            "cyclone":   ["Cyclone", "NaturalDisaster", "Environment", "India"],
            "drought":   ["Drought", "Environment", "FarmerIssues", "India"],
            "earthquake":["Earthquake", "NaturalDisaster", "Environment", "India"],
            "forest":    ["ForestFire", "NaturalDisaster", "Environment", "India"],
            "landslide": ["Landslide", "NaturalDisaster", "Environment", "India"],
            "climate":   ["ClimateChange", "Environment", "India"],
            "deforest":  ["Deforestation", "Environment", "India"],
            "river":     ["Environment", "Pollution", "India"],
            "wildfire":  ["Wildfire", "NaturalDisaster", "Environment", "India"],
            "water":     ["Environment", "WaterCrisis", "India"],
            "solar":     ["RenewableEnergy", "Environment", "India"],
            "plastic":   ["Pollution", "Environment", "India"],
        },
        "default_labels": ["Environment", "NaturalDisaster", "India"],
        "count": 2000,
    },
    "Health": {
        "templates":  HEALTH_TEMPLATES,
        "labels_map": {
            "covid":     ["COVID19", "Health", "Pandemic"],
            "mental":    ["MentalHealth", "Health", "India"],
            "tubercul":  ["Tuberculosis", "Health", "PublicHealth"],
            "dengue":    ["Dengue", "Health", "PublicHealth"],
            "cancer":    ["Cancer", "Health"],
            "malnutri":  ["Malnutrition", "Health", "PublicHealth"],
            "diabetes":  ["Diabetes", "Health"],
            "suicide":   ["MentalHealth", "Health", "India"],
            "aiims":     ["Health", "MedicalResearch", "India"],
            "counterfeit":["Health", "Crime", "India"],
            "infant":    ["PublicHealth", "Health", "India"],
            "vaccin":    ["Health", "PublicHealth", "India"],
            "hospital":  ["Health", "PublicHealth", "India"],
            "telemed":   ["Health", "Technology", "India"],
            "air":       ["AirPollution", "Health", "Environment"],
        },
        "default_labels": ["Health", "PublicHealth", "India"],
        "count": 2000,
    },
    "Education": {
        "templates":  EDUCATION_TEMPLATES,
        "labels_map": {
            "neet":     ["NEET", "Education", "Health"],
            "jee":      ["JEE", "Education", "Engineering"],
            "upsc":     ["UPSC", "Education", "Government"],
            "cbse":     ["CBSE", "Education"],
            "iit":      ["IIT", "Education", "Technology"],
            "dropout":  ["Education", "SocialIssues", "India"],
            "digital":  ["Education", "Technology", "India"],
            "olympiad": ["Education", "Sports", "India"],
            "teacher":  ["Education", "Government", "India"],
            "budget":   ["Education", "Government", "India"],
            "scholar":  ["Education", "India"],
            "school":   ["Education", "India"],
            "univers":  ["Education", "India"],
            "coachi":   ["Education", "Crime", "India"],
            "meal":     ["Education", "SocialIssues", "India"],
        },
        "default_labels": ["Education", "India"],
        "count": 2000,
    },
    "Science": {
        "templates":  SCIENCE_TEMPLATES,
        "labels_map": {
            "isro":     ["ISRO", "Space", "IndiaInSpace"],
            "chandray": ["ISRO", "Space", "Chandrayaan", "IndiaInSpace"],
            "gaganyaan":["ISRO", "Space", "HumanSpaceFlight"],
            "discover": ["Science", "Research", "India"],
            "iit":      ["IIT", "Research", "Technology"],
            "drdo":     ["DRDO", "Defence", "Technology"],
            "quantum":  ["QuantumComputing", "Technology", "Science"],
            "nature":   ["Science", "Research", "India"],
            "nuclear":  ["Science", "Energy", "India"],
            "genome":   ["Science", "MedicalResearch", "India"],
            "ai":       ["ArtificialIntelligence", "AI", "Technology"],
            "drug":     ["MedicalResearch", "Health", "Science"],
            "space":    ["Space", "ISRO", "IndiaInSpace"],
            "mars":     ["ISRO", "Space", "IndiaInSpace", "MarsOrbiter"],
            "renew":    ["RenewableEnergy", "Science", "Environment"],
        },
        "default_labels": ["Science", "Research", "India"],
        "count": 2000,
    },
}


def generate_labels(text: str, labels_map: dict, default_labels: list) -> list:
    tl = text.lower()
    labels = list(default_labels)
    for keyword, tags in labels_map.items():
        if keyword in tl:
            labels.extend(tags)
    return sorted(set(labels))


def augment_domain(domain: str, config: dict, entities: list) -> list:
    templates   = config["templates"]
    labels_map  = config["labels_map"]
    default_lbl = config["default_labels"]
    target      = config["count"]

    rows = []
    names = [e["name"] for e in entities[:200]] if entities else ["officials"]

    attempts = 0
    while len(rows) < target and attempts < target * 5:
        attempts += 1
        template = random.choice(templates)
        name     = random.choice(names) if names else "officials"
        city     = random.choice(CITIES)
        region   = random.choice(REGIONS)

        text = template.replace("{name}", name)\
                       .replace("{city}", city)\
                       .replace("{region}", region)

        labels = generate_labels(text, labels_map, default_lbl)

        rows.append({
            "text":   text,
            "labels": "|".join(labels),
            "domain": domain,
            "source": f"augmented_{domain.lower()}",
        })

    return rows


def main():
    print("Loading KG entities...")
    entities_by_domain = load_entities_by_domain()
    print(f"  Loaded domains: {list(entities_by_domain.keys())}")

    all_rows = []
    for domain, config in DOMAIN_CONFIGS.items():
        entities = entities_by_domain.get(domain, [])
        rows     = augment_domain(domain, config, entities)
        print(f"  Generated {len(rows):>5} rows for domain: {domain}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RAW_DIR, "augmented_weak_domains.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} total augmented rows to {out_path}")
    print("\nDomain breakdown:")
    for domain, count in df["domain"].value_counts().items():
        print(f"  {domain:<20} {count}")


if __name__ == "__main__":
    main()
