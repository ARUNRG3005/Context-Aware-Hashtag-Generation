import pandas as pd
import random
import itertools
import os

BASE = os.path.dirname(os.path.dirname(__file__))
OUT_CSV = os.path.join(BASE, "data", "raw", "augmented_v3_sensitive.csv")

# =========================================================================
# SENSITIVE DOMAIN GENERATORS
# =========================================================================
LOCATIONS = ["Delhi", "UP", "Uttar Pradesh", "Bihar", "Haryana", "Rajasthan", 
             "Mumbai", "Kolkata", "Bengaluru", "Chennai", "MP", "Madhya Pradesh",
             "Punjab", "Gujarat", "Kerala", "Assam"]

# ── Gender Violence & Honour Killings ──
GV_VICTIMS = ["A 21-year-old woman", "A college student", "A minor girl", "A Dalit woman", "A local journalist"]
GV_ACTIONS = [
    ("was brutally raped", "#GenderViolence|#Rape|#Crime|#Justice"),
    ("was subjected to a horrific gangrape", "#GenderViolence|#Rape|#Crime|#SocialIssues"),
    ("was killed in a suspected honour killing case", "#HonourKilling|#GenderViolence|#Crime|#SocialIssues"),
    ("was found dead, sparking protests over honour killing", "#HonourKilling|#GenderViolence|#Protests|#Crime"),
    ("faced severe domestic violence", "#GenderViolence|#DomesticViolence|#Crime|#SocialIssues"),
    ("was attacked with acid", "#AcidAttack|#GenderViolence|#Crime|#Justice"),
    ("was burnt alive over dowry demands", "#DowryDeath|#GenderViolence|#Crime|#SocialIssues"),
    ("was murdered for marrying outside her caste", "#HonourKilling|#CasteViolence|#GenderViolence|#Crime")
]
GV_CONTEXTS = [
    "by a group of men.", "by her own relatives.", "in a secluded area.", "while returning from work.",
    "by her in-laws.", "by her family members.", "leading to massive outrage across the country.",
    "and the police have registered an FIR.", "sparking widespread protests."
]

def generate_gender_violence(count=1500):
    rows = []
    combos = list(itertools.product(GV_VICTIMS, GV_ACTIONS, GV_CONTEXTS, LOCATIONS))
    random.shuffle(combos)
    for v, (action, tags), ctx, loc in combos[:count]:
        text = f"In {loc}, {v.lower()} {action} {ctx}"
        rows.append({"text": text, "labels": tags, "domain": "SocialIssues", "source": "aug_gender_violence"})
    return rows

# ── Riots & Communal Violence ──
RIOT_EVENTS = [
    ("Communal clashes erupted", "#Riots|#CommunalViolence|#SocialIssues|#Crime"),
    ("Violent protests broke out", "#Protests|#Riots|#SocialUnrest|#Politics"),
    ("Curfew was imposed after clashes", "#Curfew|#Riots|#SocialIssues|#LawAndOrder"),
    ("Mob violence resulted in deaths", "#MobViolence|#Riots|#Crime|#LawAndOrder"),
    ("Caste-based violence escalated", "#CasteViolence|#Riots|#SocialIssues|#SocialUnrest"),
    ("Stone-pelting incidents were reported", "#Riots|#LawAndOrder|#Crime|#Protests")
]
RIOT_CONTEXTS = [
    "during a religious procession", "following an inflammatory social media post",
    "after political leaders delivered hate speeches", "over a land dispute",
    "leaving several injured and properties damaged", "and internet services have been suspended"
]

def generate_riots(count=1500):
    rows = []
    combos = list(itertools.product(LOCATIONS, RIOT_EVENTS, RIOT_CONTEXTS))
    random.shuffle(combos)
    for loc, (event, tags), ctx in combos[:count]:
        text = f"{event} in {loc} {ctx}. Police forces have been deployed."
        rows.append({"text": text, "labels": tags, "domain": "SocialIssues", "source": "aug_riots"})
    return rows

# ── Corruption & Scams ──
CORRUPT_ORGS = ["CBI", "ED", "Enforcement Directorate", "Anti-Corruption Bureau", "Income Tax Department"]
CORRUPT_TARGETS = ["a senior politician", "a prominent businessman", "a state minister", "government officials", "a massive shell company network"]
CORRUPT_ACTIONS = [
    ("conducted raids", "#Corruption|#CBI|#ED|#Crime"),
    ("uncovered a massive scam worth thousands of crores", "#Scam|#Corruption|#Crime|#Economy"),
    ("arrested key suspects in the money laundering case", "#MoneyLaundering|#Corruption|#Crime|#ED"),
    ("filed a chargesheet regarding the alleged disproportionate assets", "#Corruption|#Politics|#Crime|#CBI"),
    ("seized uncounted cash and benami properties", "#Corruption|#BlackMoney|#Crime")
]

def generate_corruption(count=1500):
    rows = []
    combos = list(itertools.product(CORRUPT_ORGS, CORRUPT_TARGETS, CORRUPT_ACTIONS, LOCATIONS))
    random.shuffle(combos)
    for org, tgt, (action, tags), loc in combos[:count]:
        text = f"The {org} {action} linked to {tgt} in {loc}."
        rows.append({"text": text, "labels": tags, "domain": "Crime", "source": "aug_corruption"})
    return rows

# ── Generic High-Quality Filler for Balance ──
# Because we need 10,000 per domain, let's aggressively expand Health and Education too

HEALTH_SUBJECTS = ["A new hospital", "A deadly virus outbreak", "Government healthcare schemes", "Malnutrition among children", "Mental health awareness"]
HEALTH_ACTIONS = [
    ("was inaugurated to provide free treatment.", "#PublicHealth|#Hospitals|#Health|#India"),
    ("has claimed dozens of lives this week.", "#PublicHealth|#Disease|#Health|#Crisis"),
    ("are struggling to reach rural populations.", "#PublicHealth|#RuralIndia|#Health|#Policy"),
    ("remains a critical concern for policymakers.", "#Health|#Malnutrition|#PublicHealth|#SocialIssues"),
    ("is gaining momentum among urban youth.", "#MentalHealth|#Health|#Wellness|#Society")
]
def generate_health(count=2000):
    rows = []
    combos = list(itertools.product(HEALTH_SUBJECTS, HEALTH_ACTIONS, LOCATIONS))
    random.shuffle(combos)
    for subj, (action, tags), loc in combos[:count]:
        text = f"In {loc}, {subj.lower()} {action}"
        rows.append({"text": text, "labels": tags, "domain": "Health", "source": "aug_health"})
    return rows

EDU_SUBJECTS = ["The new National Education Policy", "Thousands of students", "Several government schools", "Top universities", "Board exams"]
EDU_ACTIONS = [
    ("is being implemented to modernize the curriculum.", "#Education|#NEP|#Policy|#India"),
    ("are protesting against the recent fee hike.", "#Education|#Protests|#Students|#SocialIssues"),
    ("lack basic infrastructure like clean water and toilets.", "#Education|#Infrastructure|#SocialIssues|#India"),
    ("have seen a surge in foreign student enrollments.", "#Education|#Universities|#HigherEd|#India"),
    ("were postponed due to administrative issues.", "#Education|#Exams|#Students|#India")
]
def generate_education(count=2000):
    rows = []
    combos = list(itertools.product(EDU_SUBJECTS, EDU_ACTIONS, LOCATIONS))
    random.shuffle(combos)
    for subj, (action, tags), loc in combos[:count]:
        text = f"In {loc}, {subj.lower()} {action}"
        rows.append({"text": text, "labels": tags, "domain": "Education", "source": "aug_education"})
    return rows

def main():
    print("============================================================")
    print("DATA AUGMENTOR V3 — High-Quality Sensitive & Weak Domains")
    print("============================================================")
    
    rows = []
    rows.extend(generate_gender_violence(3000))
    rows.extend(generate_riots(3000))
    rows.extend(generate_corruption(3000))
    rows.extend(generate_health(4000))
    rows.extend(generate_education(4000))
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    
    print(f"Total rows generated: {len(df)}")
    print(df["source"].value_counts())
    print(f"\nSaved to: {OUT_CSV}")

if __name__ == "__main__":
    main()
