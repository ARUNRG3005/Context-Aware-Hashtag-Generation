"""
bulk_downloader.py — Downloads datasets targeting weak domains and saves them to D: drive.
Ensures HF cache strictly points to D: drive to save C: drive space.
"""

import os
import re
import pandas as pd
from datasets import load_dataset

# Force HuggingFace cache to D: drive
os.environ["HF_HOME"] = r"D:\hashtag-generator\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"D:\hashtag-generator\hf_cache\datasets"
os.environ["HF_HUB_CACHE"] = r"D:\hashtag-generator\hf_cache\hub"

RAW_DIR = r"D:\hashtag-generator\data\raw"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

INDIA_KEYWORDS = [
    "india", "indian", "delhi", "mumbai", "bangalore", "bengaluru",
    "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur",
    "lucknow", "patna", "bhopal", "surat", "kerala", "gujarat", "maharashtra",
    "cricket", "ipl", "bcci", "modi", "bjp", "congress", "gandhi", "parliament", 
    "lok sabha", "rajya sabha", "aap", "kejriwal", "rupee", "rbi", "sensex", "nifty",
    "isro", "chandrayaan", "dalit", "caste", "neet", "jee", "upsc", "bollywood"
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

def download_yahoo():
    print("Downloading yahoo_answers_topics...")
    try:
        dataset = load_dataset("community-datasets/yahoo_answers_topics", split="train", cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        count = 0
        for row in dataset:
            text = clean_text(row.get("question_title", "") + " " + row.get("question_content", ""))
            label = row.get("topic")
            # 0: Society & Culture, 1: Science & Math, 2: Health, 3: Education & Reference, 
            # 4: Computers & Internet, 5: Sports, 6: Business & Finance, 7: Entertainment & Music, 
            # 8: Family & Relationships, 9: Politics & Government
            domain = "General"
            if label == 2: domain = "Health"
            elif label == 3: domain = "Education"
            elif label == 7: domain = "Entertainment"
            elif label == 9: domain = "Politics"
            elif label == 1: domain = "Science"
            elif label == 5: domain = "Sports"
            elif label == 6: domain = "Business"
            elif label == 4: domain = "Technology"
            
            # We skip general to focus on weak domains
            if domain == "General": continue
            
            rows.append({
                "text": text[:1000],
                "labels": "",
                "domain": domain,
                "source": "yahoo_answers"
            })
            count += 1
            if count > 80000: break
            
        df = pd.DataFrame(rows)
        out_path = os.path.join(RAW_DIR, "hf_yahoo.csv")
        df.to_csv(out_path, index=False)
        print(f"  OK Saved {len(df)} rows to {out_path}")
    except Exception as e:
        print(f"Failed yahoo: {e}")

def download_cnn():
    print("Downloading abisee/cnn_dailymail...")
    try:
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        count = 0
        for row in dataset:
            text = clean_text(row.get("article", ""))
            if is_india_relevant(text):
                rows.append({
                    "text": text[:1000],
                    "labels": "",
                    "domain": "General",
                    "source": "cnn_dailymail"
                })
            count += 1
            if len(rows) >= 30000:
                break
        df = pd.DataFrame(rows)
        out_path = os.path.join(RAW_DIR, "hf_cnn_dailymail.csv")
        df.to_csv(out_path, index=False)
        print(f"  OK Saved {len(df)} rows to {out_path}")
    except Exception as e:
        print(f"Failed cnn_dailymail: {e}")

def download_indian_finance():
    print("Downloading pranali96/indian_financial_news_42k...")
    try:
        dataset = load_dataset("pranali96/indian_financial_news_42k", split="train", cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        for row in dataset:
            text = clean_text(str(row.get("headline", "")))
            if len(text.split()) > 5:
                rows.append({
                    "text": text[:1000],
                    "labels": "",
                    "domain": "General", # Preprocessor will shift this
                    "source": "indian_financial_news_42k"
                })
        df = pd.DataFrame(rows)
        out_path = os.path.join(RAW_DIR, "hf_indian_financial_news.csv")
        df.to_csv(out_path, index=False)
        print(f"  OK Saved {len(df)} rows to {out_path}")
    except Exception as e:
        print(f"Failed indian_financial_news_42k: {e}")

def download_indian_news():
    print("Downloading JaiminP20/indian_news_sentiment...")
    try:
        dataset = load_dataset("JaiminP20/indian_news_sentiment", split="train", cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        for row in dataset:
            text = clean_text(row.get("text", ""))
            if text and len(text) > 20:
                rows.append({"text": text, "label": "News"})
        df = pd.DataFrame(rows)
        out_path = os.path.join(RAW_DIR, "hf_indian_sentiment.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved {len(df)} rows to {out_path}")
    except Exception as e:
        print(f"  ✗ Failed to download JaiminP20: {e}")

def download_kdave_finance():
    print("Downloading kdave/Indian_Financial_News...")
    try:
        dataset = load_dataset("kdave/Indian_Financial_News", split="train", cache_dir=os.environ["HF_DATASETS_CACHE"])
        rows = []
        for row in dataset:
            text = clean_text(row.get("Description", "") + " " + row.get("Title", ""))
            if text and len(text) > 20:
                rows.append({"text": text, "label": "Finance"})
        df = pd.DataFrame(rows)
        out_path = os.path.join(RAW_DIR, "hf_kdave_finance.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved {len(df)} rows to {out_path}")
    except Exception as e:
        print(f"  ✗ Failed to download kdave: {e}")

if __name__ == "__main__":
    print("Starting Bulk Download to D: drive...")
    download_indian_news()
    download_kdave_finance()
    download_indian_finance()
    print("Bulk download complete.")
