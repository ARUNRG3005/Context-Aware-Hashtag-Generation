"""
big_downloader.py — Aggressive multi-dataset downloader for India news.
Target: Up to 50GB on D: drive.
Tries 10+ datasets with graceful fallbacks.
"""

import os
import re
import sys
import pandas as pd
from datasets import load_dataset

# ── Force ALL HuggingFace caches to D: ────────────────────────────────────
os.environ["HF_HOME"]              = r"D:\hashtag-generator\hf_cache"
os.environ["HF_DATASETS_CACHE"]    = r"D:\hashtag-generator\hf_cache\datasets"
os.environ["HF_HUB_CACHE"]         = r"D:\hashtag-generator\hf_cache\hub"
os.environ["TRANSFORMERS_CACHE"]   = r"D:\hashtag-generator\hf_cache\transformers"
os.environ["XDG_CACHE_HOME"]       = r"D:\hashtag-generator\hf_cache"

RAW_DIR = r"D:\hashtag-generator\data\raw"
os.makedirs(RAW_DIR, exist_ok=True)
for d in [r"D:\hashtag-generator\hf_cache\datasets",
          r"D:\hashtag-generator\hf_cache\hub"]:
    os.makedirs(d, exist_ok=True)

# ── India Keyword Filter ────────────────────────────────────────────────────
INDIA_KEYWORDS = [
    "india", "indian", "delhi", "mumbai", "bangalore", "bengaluru",
    "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur",
    "lucknow", "patna", "bhopal", "surat", "kerala", "gujarat",
    "maharashtra", "rajasthan", "tamilnadu", "tamil", "karnataka",
    "cricket", "ipl", "bcci", "sachin", "kohli", "dhoni", "rohit",
    "virat", "bumrah", "jadeja", "ranji", "icc", "shubman",
    "modi", "bjp", "congress", "gandhi", "parliament", "lok sabha",
    "rajya sabha", "aap", "kejriwal", "mamata", "yogi", "rahul gandhi",
    "rupee", "rbi", "sensex", "nifty", "infosys", "tata", "wipro",
    "reliance", "adani", "ambani", "flipkart", "zomato", "paytm",
    "isro", "chandrayaan", "gaganyaan", "sriharikota",
    "dalit", "caste", "reservation", "neet", "jee", "upsc", "cbse",
    "kabaddi", "pv sindhu", "saina", "neeraj chopra", "mary kom",
    "bollywood", "srk", "shah rukh", "salman", "deepika", "alia bhatt",
    "jammu", "kashmir", "ladakh", "northeast india", "assam",
    "election commission", "supreme court india", "high court",
    "army india", "indian army", "navy", "air force india",
    "gst", "income tax", "sebi", "irdai",
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

def save(rows: list, filename: str):
    df = pd.DataFrame(rows)
    out = os.path.join(RAW_DIR, filename)
    df.to_csv(out, index=False)
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"  Saved {len(df):,} rows → {filename} ({size_mb:.1f} MB)")
    return len(df)

# ═══════════════════════════════════════════════════════════════════════════
# DATASET 1: Full CNN DailyMail — all 287K articles
# ═══════════════════════════════════════════════════════════════════════════
def dl_cnn_full():
    print("\n[1/10] Full CNN DailyMail (287K)...")
    try:
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train",
                          cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        for row in ds:
            text = clean_text(row.get("article", ""))
            if is_india_relevant(text):
                rows.append({"text": text[:800], "labels": "", "domain": "General",
                             "source": "cnn_full", "split": "train"})
        # Also val+test
        for split in ["validation", "test"]:
            ds2 = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split,
                               cache_dir=os.environ["HF_DATASETS_CACHE"])
            for row in ds2:
                text = clean_text(row.get("article", ""))
                if is_india_relevant(text):
                    rows.append({"text": text[:800], "labels": "", "domain": "General",
                                 "source": "cnn_full", "split": split})
        save(rows, "hf_cnn_full.csv")
    except Exception as e:
        print(f"  FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DATASET 2: BBC News AllTime
# ═══════════════════════════════════════════════════════════════════════════
def dl_bbc_alltime():
    print("\n[2/10] BBC News AllTime...")
    try:
        ds = load_dataset("RealTimeData/bbc_news_alltime", split="train",
                          cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        for row in ds:
            text = clean_text(str(row.get("content", "")) + " " + str(row.get("title", "")))
            if is_india_relevant(text) or "india" in str(row.get("link", "")).lower():
                rows.append({
                    "text":   text[:800],
                    "labels": "",
                    "domain": "General",
                    "source": "bbc_alltime",
                    "section": str(row.get("section", "")),
                })
        save(rows, "hf_bbc_alltime.csv")
    except Exception as e:
        print(f"  FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DATASET 3: HuffPost News Categories (200K)
# ═══════════════════════════════════════════════════════════════════════════
def dl_huffpost():
    print("\n[3/10] HuffPost News (200K)...")
    KEEP_CATS = {"WORLD NEWS", "POLITICS", "SPORTS", "SCIENCE", "TECH",
                 "BUSINESS", "WELLNESS", "EDUCATION", "ENVIRONMENT", "CRIME"}
    try:
        ds = load_dataset("heegyu/news-category-dataset", split="train",
                          cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        for row in ds:
            cat  = str(row.get("category", "")).upper()
            text = clean_text(str(row.get("headline", "")) + " " + str(row.get("short_description", "")))
            if cat in KEEP_CATS and is_india_relevant(text):
                rows.append({
                    "text":   text[:800],
                    "labels": "",
                    "domain": cat.title().replace(" ", ""),
                    "source": "huffpost",
                })
        save(rows, "hf_huffpost.csv")
    except Exception as e:
        print(f"  FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DATASET 4: Indian News Articles (if exists on HF)
# ═══════════════════════════════════════════════════════════════════════════
def dl_indian_news():
    print("\n[4/10] Indian News Articles...")
    candidates = [
        "d0r1h/Indian_news",
        "GautamDaksh/Indian-News-Articles",
        "Soumitri/Indian_news_dataset",
        "Tejas3/Indian_news",
        "anmolkr2/indian-news",
    ]
    for candidate in candidates:
        try:
            ds = load_dataset(candidate, split="train",
                              cache_dir=os.environ["HF_DATASETS_CACHE"])
            rows = []
            for row in ds:
                # Try common column names
                text = clean_text(
                    str(row.get("text", row.get("article", row.get("content",
                        row.get("headline", row.get("title", ""))))))
                )
                if len(text.split()) >= 5:
                    rows.append({
                        "text":   text[:800],
                        "labels": "",
                        "domain": str(row.get("category", row.get("domain", "General"))),
                        "source": f"indian_news_{candidate.split('/')[-1]}",
                    })
            fname = f"hf_{candidate.split('/')[-1]}.csv"
            save(rows, fname)
            print(f"  SUCCESS with: {candidate}")
            break
        except Exception as e:
            print(f"  {candidate}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DATASET 5: IndicNLP Headline Generation (AI4Bharat)
# ═══════════════════════════════════════════════════════════════════════════
def dl_indicnlp():
    print("\n[5/10] IndicNLP Headline Generation...")
    try:
        ds = load_dataset("ai4bharat/IndicHeadlineGeneration", "en",
                          split="train",
                          cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        for row in ds:
            text = clean_text(str(row.get("article", "")) + " " + str(row.get("headline", "")))
            if len(text.split()) >= 5:
                rows.append({
                    "text":   text[:800],
                    "labels": "",
                    "domain": "General",
                    "source": "indicnlp_headline",
                })
        save(rows, "hf_indicnlp_headline.csv")
    except Exception as e:
        print(f"  FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DATASET 6: MLSUM (multilingual summarization — has India content)
# ═══════════════════════════════════════════════════════════════════════════
def dl_xsum():
    print("\n[6/10] XSum (BBC English summaries)...")
    try:
        ds = load_dataset("EdinburghNLP/xsum", split="train",
                          cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        count = 0
        for row in ds:
            text = clean_text(str(row.get("document", "")))
            if is_india_relevant(text):
                rows.append({
                    "text":   text[:800],
                    "labels": "",
                    "domain": "General",
                    "source": "xsum",
                })
            count += 1
            if len(rows) >= 25000 or count > 200000:
                break
        save(rows, "hf_xsum.csv")
    except Exception as e:
        print(f"  FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DATASET 7: CC News (large, general news)
# ═══════════════════════════════════════════════════════════════════════════
def dl_cc_news():
    print("\n[7/10] CC News (2019)...")
    try:
        ds = load_dataset("cc_news", split="train",
                          cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        count = 0
        for row in ds:
            text = clean_text(str(row.get("text", "")))
            if is_india_relevant(text):
                rows.append({
                    "text":   text[:800],
                    "labels": "",
                    "domain": "General",
                    "source": "cc_news",
                })
            count += 1
            if len(rows) >= 30000 or count > 500000:
                break
        save(rows, "hf_cc_news.csv")
    except Exception as e:
        print(f"  FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DATASET 8: NewsQA / SQUAD India articles
# ═══════════════════════════════════════════════════════════════════════════
def dl_newsqa():
    print("\n[8/10] NewsQA dataset...")
    try:
        ds = load_dataset("newsqa", split="train",
                          cache_dir=os.environ["HF_DATASETS_CACHE"],
                          trust_remote_code=False)
        rows = []
        for row in ds:
            text = clean_text(str(row.get("story_text", "")))
            if is_india_relevant(text):
                rows.append({
                    "text":   text[:800],
                    "labels": "",
                    "domain": "General",
                    "source": "newsqa",
                })
        save(rows, "hf_newsqa.csv")
    except Exception as e:
        print(f"  FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DATASET 9: Multi-News
# ═══════════════════════════════════════════════════════════════════════════
def dl_multinews():
    print("\n[9/10] Multi-News...")
    try:
        ds = load_dataset("multi_news", split="train",
                          cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        count = 0
        for row in ds:
            text = clean_text(str(row.get("document", "")))
            if is_india_relevant(text):
                rows.append({
                    "text":   text[:800],
                    "labels": "",
                    "domain": "General",
                    "source": "multi_news",
                })
            count += 1
            if len(rows) >= 20000 or count > 100000:
                break
        save(rows, "hf_multinews.csv")
    except Exception as e:
        print(f"  FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DATASET 10: Argilla News Summary
# ═══════════════════════════════════════════════════════════════════════════
def dl_argilla():
    print("\n[10/10] Argilla News Summary...")
    try:
        ds = load_dataset("argilla/news-summary", split="train",
                          cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        for row in ds:
            text = clean_text(str(row.get("text", "")))
            if is_india_relevant(text) and len(text.split()) >= 10:
                rows.append({
                    "text":   text[:800],
                    "labels": "",
                    "domain": "General",
                    "source": "argilla_news",
                })
        save(rows, "hf_argilla_news.csv")
    except Exception as e:
        print(f"  FAILED: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("BIG DOWNLOADER — India News Data (up to 50GB on D:)")
    print("=" * 60)
    print(f"Cache: {os.environ['HF_DATASETS_CACHE']}")

    # Run all in order — failures are graceful
    dl_cnn_full()
    dl_bbc_alltime()
    dl_huffpost()
    dl_indian_news()
    dl_indicnlp()
    dl_xsum()
    dl_cc_news()
    dl_newsqa()
    dl_multinews()
    dl_argilla()

    print("\n" + "=" * 60)
    print("Download phase complete. Check D:/hashtag-generator/data/raw/")
    total = sum(
        os.path.getsize(os.path.join(RAW_DIR, f)) for f in os.listdir(RAW_DIR)
        if f.endswith(".csv")
    ) / 1024 / 1024
    print(f"Total raw data: {total:.1f} MB")
