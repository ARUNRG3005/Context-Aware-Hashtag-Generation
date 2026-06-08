import os
import torch
import feedparser
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer

def get_headlines():
    # 10 Diverse Domains via RSS
    feeds = {
        "Politics (India)": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
        "Sports": "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
        "Technology": "https://techcrunch.com/feed/",
        "Business": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "Entertainment": "https://www.bollywoodhungama.com/rss/news",
        "Health": "https://www.who.int/rss-feeds/news-english.xml",
        "Science": "https://www.sciencedaily.com/rss/top/science.xml",
        "World News": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Automotive": "https://www.autocarindia.com/rss/news",
        "Environment": "https://www.thehindu.com/sci-tech/energy-and-environment/feeder/default.rss"
    }
    
    articles = []
    for domain, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            if feed.entries:
                entry = feed.entries[0] # Get top headline
                articles.append({
                    "domain": domain,
                    "title": entry.title,
                    "summary": getattr(entry, 'summary', '')[:200]
                })
        except Exception as e:
            print(f"Error fetching {domain}: {e}")
            
    return articles

def run_inference():
    print("Scraping Today's Headlines across 10 Domains...")
    articles = get_headlines()
    
    print(f"Successfully scraped {len(articles)} live headlines.\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model_path = os.path.join("d:\\hashtag-generator", "checkpoints", "best_model")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print("Loading AI Model...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    # Load MLB classes to decode predictions
    import json
    with open(os.path.join(model_path, "label_classes.json"), "r") as f:
        classes = json.load(f)
        
    print("\n=======================================================")
    print("LIVE INFERENCE RESULTS")
    print("=======================================================\n")
    
    with torch.no_grad():
        for article in articles:
            text = f"{article['title']} - {article['summary']}"
            
            # Simple KG infusion simulation (assuming user's empty KG)
            inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = torch.sigmoid(logits)
            
            # Get Top 5 predictions
            top5_indices = torch.topk(probs, k=5).indices
            
            print(f"DOMAIN:   {article['domain']}")
            print(f"HEADLINE: {article['title']}")
            print("PREDICTED HASHTAGS:")
            
            for idx in top5_indices:
                tag = classes[idx]
                confidence = probs[idx].item() * 100
                print(f"  #{tag:<20} | Confidence: {confidence:.1f}%")
            print("-" * 55)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    run_inference()
