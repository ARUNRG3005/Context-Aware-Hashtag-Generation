"""
preprocessor.py — Combines all data sources into clean training dataset
Sources:
    1. data/raw/scraped_articles.csv   (India news articles)
    2. data/raw/ag_news_train.csv      (120,000 general news)
    3. data/raw/ag_news_test.csv       (7,600 general news)
    4. data/raw/bbc_news.csv           (10,766 BBC articles with section labels)
 
Output:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
 
Each row: text | labels (pipe-separated hashtags) | domain | source
"""
 
import os
import re
import pandas as pd
from collections import Counter
 
# ── Paths ──────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(__file__))
RAW_DIR      = os.path.join(BASE, "data", "raw")
PROCESSED    = os.path.join(BASE, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)
 
SCRAPED_CSV  = os.path.join(RAW_DIR, "scraped_articles.csv")
AG_TRAIN_CSV = os.path.join(RAW_DIR, "ag_news_train.csv")
AG_TEST_CSV  = os.path.join(RAW_DIR, "ag_news_test.csv")
BBC_CSV      = os.path.join(RAW_DIR, "bbc_news.csv")
 
# ── AG News label → domain + hashtags ─────────────────────────────────────
# FIX: Label 0 is World/International news, NOT IndianPolitics.
# Indian-specific tags are only added by keyword_boost when text is India-relevant.
AG_LABEL_MAP = {
    0: {"domain": "Politics",   "tags": ["WorldNews", "International"]},
    1: {"domain": "Sports",     "tags": ["Sports"]},
    2: {"domain": "Business",   "tags": ["Business", "Economy"]},
    3: {"domain": "Technology", "tags": ["Technology", "Digital"]},
}
 
# AG News text → extra tags added only when keyword is found in article
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
 
# ── BBC section → domain + hashtags ───────────────────────────────────────
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
 
# ── Domain/subdomain → clean hashtag list (scraped articles) ──────────────
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
 
# ── India relevance keywords ───────────────────────────────────────────────
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
 
 
# ── Text utilities ─────────────────────────────────────────────────────────
 
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
    """
    Add extra tags based on keywords found in text.
    FIX: Tags starting with 'Indian' are only added when the article
    is actually India-relevant — prevents Australian/English cricket
    articles from getting #IndianCricket #IndianPolitics etc.
    """
    tl = text.lower()
    is_india = is_india_relevant(tl)
    extra = []
    for kw, tags in AG_KEYWORD_TAGS.items():
        if kw in tl:
            for tag in tags:
                # Indian-specific tags only when article is about India
                if tag.startswith("Indian") and not is_india:
                    continue
                extra.append(tag)
    return list(set(base_tags + extra))
 
 
# ── Source processors ──────────────────────────────────────────────────────
 
def process_scraped(df: pd.DataFrame) -> pd.DataFrame:
    """Scraped India articles — use DOMAIN_HASHTAG_MAP only."""
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
    """AG News — India-relevant rows only, with keyword-boosted tags."""
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
        rows.append({
            "text":   text[:512],
            "labels": "|".join(sorted(set(tags))),
            "domain": mapping["domain"],
            "source": "ag_news",
        })
    print(f"  ✓ {len(rows)} India-relevant rows")
    print(f"  ✗ {skipped} rows skipped")
    return pd.DataFrame(rows)
 
 
def process_bbc(df: pd.DataFrame) -> pd.DataFrame:
    """BBC — uses section column for clean labels."""
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
    print(f"  ✗ {skipped} skipped")
    return pd.DataFrame(rows)
 
 
# ── Balance ────────────────────────────────────────────────────────────────
 
def balance_dataset(df: pd.DataFrame, max_per_domain: int = 3000) -> pd.DataFrame:
    print(f"\nBalancing dataset (max {max_per_domain} per domain)...")
    balanced = []
    for domain, group in df.groupby("domain"):
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
    print("PREPROCESSOR — Building Training Dataset")
    print("=" * 55)
 
    all_dfs = []
 
    if os.path.exists(SCRAPED_CSV):
        all_dfs.append(process_scraped(pd.read_csv(SCRAPED_CSV)))
    else:
        print("WARNING: scraped_articles.csv not found — skipping")
 
    if os.path.exists(AG_TRAIN_CSV):
        all_dfs.append(process_ag_news(pd.read_csv(AG_TRAIN_CSV), "ag_news_train"))
    else:
        print("WARNING: ag_news_train.csv not found — skipping")
 
    if os.path.exists(AG_TEST_CSV):
        all_dfs.append(process_ag_news(pd.read_csv(AG_TEST_CSV), "ag_news_test"))
    else:
        print("WARNING: ag_news_test.csv not found — skipping")
 
    if os.path.exists(BBC_CSV):
        all_dfs.append(process_bbc(pd.read_csv(BBC_CSV)))
    else:
        print("WARNING: bbc_news.csv not found — skipping")
 
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal before filtering: {len(df_all)} rows")
 
    df_all = df_all[df_all["labels"].str.len() > 0]
    df_all = df_all[df_all["text"].str.split().str.len() >= 8]
    df_all = df_all.drop_duplicates(subset=["text"])
    print(f"Total after dedup + filter: {len(df_all)} rows")
 
    print("\n=== SOURCE BREAKDOWN ===")
    for src, cnt in df_all["source"].value_counts().items():
        print(f"  {src:<30} {cnt}")
 
    df_balanced = balance_dataset(df_all, max_per_domain=3000)
 
    print(f"\nFinal dataset: {len(df_balanced)} rows")
    print("\n=== FINAL DOMAIN DISTRIBUTION ===")
    for domain, count in sorted(Counter(df_balanced["domain"]).items(), key=lambda x: -x[1]):
        bar = "█" * (count // 100)
        print(f"  {domain:<20} {count:>5}  {bar}")
 
    split_and_save(df_balanced)
 
    print("\n✓ Preprocessing complete.")
    print(f"  Files saved to: {PROCESSED}")
 
 
if __name__ == "__main__":
    main()
 