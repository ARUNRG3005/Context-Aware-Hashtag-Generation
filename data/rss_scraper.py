"""
rss_scraper.py — RSS feed scraper for Indian news sources
No blocking, no rate limits, legal and fast.
Output: appends to data/raw/scraped_articles.csv
"""

import csv
import os
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "raw", "scraped_articles.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# ── RSS feeds by domain ────────────────────────────────────────────────────
RSS_FEEDS = {

    "Sports": [
        "https://www.thehindu.com/sport/cricket/?service=rss",
        "https://www.thehindu.com/sport/?service=rss",
        "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms",  # TOI Sports
        "https://feeds.feedburner.com/ndtvports",
    ],

    "Politics": [
        "https://www.thehindu.com/news/national/?service=rss",
        "https://www.thehindu.com/news/national/tamil-nadu/?service=rss",
        "https://timesofindia.indiatimes.com/rssfeeds/1221656.cms",  # TOI Politics
        "https://feeds.feedburner.com/ndtvindia",
    ],

    "Business": [
        "https://www.thehindu.com/business/?service=rss",
        "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",  # TOI Business
        "https://feeds.feedburner.com/ndtvprofit",
    ],

    "Crime": [
        "https://timesofindia.indiatimes.com/rssfeeds/7098549.cms",  # TOI Crime
        "https://www.thehindu.com/news/national/?service=rss",
    ],

    "SocialIssues": [
        "https://thewire.in/category/rights/feed",
        "https://thewire.in/category/caste/feed",
        "https://thewire.in/category/communalism/feed",
        "https://thewire.in/category/women/feed",
        "https://scroll.in/feed",
    ],

    "Entertainment": [
        "https://www.thehindu.com/entertainment/?service=rss",
        "https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms",  # TOI Entertainment
        "https://feeds.feedburner.com/ndtventertainment",
    ],

    "Science": [
        "https://www.thehindu.com/sci-tech/?service=rss",
        "https://timesofindia.indiatimes.com/rssfeeds/2647163.cms",  # TOI Science
    ],

    "Health": [
        "https://www.thehindu.com/sci-tech/health/?service=rss",
        "https://timesofindia.indiatimes.com/rssfeeds/3908999.cms",  # TOI Health
    ],

    "Environment": [
        "https://www.thehindu.com/sci-tech/energy-and-environment/?service=rss",
        "https://thewire.in/category/environment/feed",
    ],

    "Education": [
        "https://timesofindia.indiatimes.com/rssfeeds/913168846.cms",  # TOI Education
        "https://www.thehindu.com/education/?service=rss",
    ],
}

# ── Domain → label ─────────────────────────────────────────────────────────
DOMAIN_LABEL_MAP = {
    "Sports":        ("Sports",        "General"),
    "Politics":      ("Politics",      "General"),
    "Business":      ("Business",      "General"),
    "Crime":         ("Crime",         "General"),
    "SocialIssues":  ("SocialIssues",  "General"),
    "Entertainment": ("Entertainment", "General"),
    "Science":       ("Science",       "General"),
    "Health":        ("Health",        "General"),
    "Environment":   ("Environment",   "Climate"),
    "Education":     ("Education",     "General"),
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip()


def scrape_rss(feed_url: str, domain: str) -> list:
    articles = []

    try:
        resp = requests.get(feed_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  Skipped {feed_url} — HTTP {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")

        if not items:
            # Try html parser as fallback
            soup = BeautifulSoup(resp.content, "html.parser")
            items = soup.find_all("item")

        d, sub = DOMAIN_LABEL_MAP[domain]
        source = feed_url.split("/")[2].replace("www.", "")

        for item in items:
            try:
                title = clean_text(
                    item.find("title").get_text() if item.find("title") else ""
                )
                description = clean_text(
                    item.find("description").get_text() if item.find("description") else ""
                )
                link = item.find("link")
                url = link.get_text() if link else feed_url

                if not title or len(title) < 10:
                    continue

                # Remove HTML from description
                desc_soup = BeautifulSoup(description, "html.parser")
                body = clean_text(desc_soup.get_text())

                articles.append({
                    "source":     source,
                    "url":        url,
                    "title":      title,
                    "body":       body[:500],
                    "domain":     d,
                    "sub_domain": sub,
                    "scraped_at": datetime.now().isoformat()
                })
                print(f"  + [{domain}] {title[:65]}...")

            except Exception:
                continue

    except Exception as e:
        print(f"  Error on {feed_url}: {e}")

    return articles


def save_append(articles: list):
    if not articles:
        return

    fieldnames = ["source", "url", "title", "body",
                  "domain", "sub_domain", "scraped_at"]
    file_exists = os.path.exists(OUTPUT_PATH)

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(articles)


def run_rss():
    print("=" * 55)
    print("RSS SCRAPER — Indian News Feeds")
    print("=" * 55)

    totals = {}
    all_articles = []

    for domain, feeds in RSS_FEEDS.items():
        print(f"\n── {domain} ──────────────────────────")
        domain_articles = []

        for feed_url in feeds:
            articles = scrape_rss(feed_url, domain)
            domain_articles.extend(articles)
            time.sleep(1)

        save_append(domain_articles)
        totals[domain] = len(domain_articles)
        all_articles.extend(domain_articles)

    print("\n=== RSS SCRAPE SUMMARY ===")
    for domain, count in totals.items():
        bar = "█" * (count // 10)
        print(f"  {domain:<20} {count:>4} articles  {bar}")
    print(f"\n  TOTAL: {sum(totals.values())} articles added")


if __name__ == "__main__":
    run_rss()