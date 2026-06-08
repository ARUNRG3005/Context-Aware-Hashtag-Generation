"""
targeted_scraper.py — Scrapes specifically for underrepresented domains
Targets: Crime, SocialIssues, Environment, Science, Health, Entertainment
Sources: Scroll.in, The Wire, The Hindu (targeted sections)
"""

import csv
import os
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "raw", "scraped_articles.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

# ── Targeted sections per domain ───────────────────────────────────────────
TARGETED_SECTIONS = {

    "Crime": [
        "https://www.thehindu.com/news/national/?topic=crime",
        "https://scroll.in/topic/crime",
        "https://scroll.in/topic/rape",
        "https://scroll.in/topic/murder",
        "https://thewire.in/rights",
        "https://thewire.in/law",
    ],

    "SocialIssues": [
        "https://scroll.in/topic/caste",
        "https://scroll.in/topic/dalits",
        "https://scroll.in/topic/communalism",
        "https://scroll.in/topic/women",
        "https://thewire.in/caste",
        "https://thewire.in/communalism",
        "https://thewire.in/women",
        "https://thewire.in/tribal",
    ],

    "Environment": [
        "https://scroll.in/topic/environment",
        "https://scroll.in/topic/climate-change",
        "https://www.thehindu.com/sci-tech/energy-and-environment/",
        "https://thewire.in/environment",
    ],

    "Science": [
        "https://www.thehindu.com/sci-tech/science/",
        "https://www.thehindu.com/sci-tech/technology/",
        "https://scroll.in/topic/isro",
        "https://scroll.in/topic/science",
    ],

    "Health": [
        "https://www.thehindu.com/sci-tech/health/",
        "https://scroll.in/topic/health",
        "https://thewire.in/health",
    ],

    "Entertainment": [
        "https://scroll.in/topic/bollywood",
        "https://scroll.in/topic/cinema",
        "https://www.thehindu.com/entertainment/movies/",
        "https://www.thehindu.com/entertainment/music/",
    ],
}

# ── Domain → label mapping ─────────────────────────────────────────────────
DOMAIN_LABEL_MAP = {
    "Crime":         ("Crime",         "General"),
    "SocialIssues":  ("SocialIssues",  "General"),
    "Environment":   ("Environment",   "Climate"),
    "Science":       ("Science",       "General"),
    "Health":        ("Health",        "General"),
    "Entertainment": ("Entertainment", "General"),
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip()


def scrape_section(url: str, domain: str, max_articles: int = 40) -> list:
    articles = []
    seen = set()

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  Skipped {url} — HTTP {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        links = []

        # Collect article links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)

            # Filter for real article links
            if len(text) < 20:
                continue
            if any(x in href for x in ["#", "?", "tag/", "author/", "topic/"]):
                continue

            # Source-specific filters
            if "thehindu.com" in url and "/article" not in href:
                continue
            if "scroll.in" in url and "scroll.in/article/" not in href:
                continue
            if "thewire.in" in url and href.count("/") < 3:
                continue

            full_url = href if href.startswith("http") else f"https://{url.split('/')[2]}{href}"
            if full_url not in seen:
                seen.add(full_url)
                links.append(full_url)

        # Scrape each article
        for link in links[:max_articles]:
            try:
                r = requests.get(link, headers=HEADERS, timeout=15)
                if r.status_code != 200:
                    continue

                s = BeautifulSoup(r.text, "html.parser")

                title_tag = s.find("h1")
                title = clean_text(title_tag.get_text()) if title_tag else ""
                if not title or len(title) < 15:
                    continue

                paragraphs = s.find_all("p")
                body = clean_text(" ".join(
                    p.get_text() for p in paragraphs[:5]
                    if len(p.get_text(strip=True)) > 40
                ))

                d, sub = DOMAIN_LABEL_MAP[domain]
                source = url.split("/")[2].replace("www.", "").replace(".com", "").replace(".in", "")

                articles.append({
                    "source":     source,
                    "url":        link,
                    "title":      title,
                    "body":       body[:500],
                    "domain":     d,
                    "sub_domain": sub,
                    "scraped_at": datetime.now().isoformat()
                })
                print(f"  + [{domain}] {title[:65]}...")
                time.sleep(1)

            except Exception:
                continue

        time.sleep(2)

    except Exception as e:
        print(f"  Error on {url}: {e}")

    return articles


def save_append(articles: list):
    """Append to existing CSV."""
    if not articles:
        return

    fieldnames = ["source", "url", "title", "body", "domain", "sub_domain", "scraped_at"]
    file_exists = os.path.exists(OUTPUT_PATH)

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(articles)

    print(f"  ✓ Appended {len(articles)} articles")


def run_targeted():
    print("=" * 55)
    print("TARGETED SCRAPER — Fixing Class Imbalance")
    print("=" * 55)

    totals = {}

    for domain, sections in TARGETED_SECTIONS.items():
        print(f"\n── {domain} ──────────────────────────────")
        domain_articles = []

        for section_url in sections:
            articles = scrape_section(section_url, domain, max_articles=30)
            domain_articles.extend(articles)

            if len(domain_articles) >= 150:
                break

        save_append(domain_articles)
        totals[domain] = len(domain_articles)

    print("\n=== TARGETED SCRAPE SUMMARY ===")
    for domain, count in totals.items():
        print(f"  {domain:<20} {count} articles")
    print(f"\n  TOTAL ADDED: {sum(totals.values())} articles")


if __name__ == "__main__":
    run_targeted()