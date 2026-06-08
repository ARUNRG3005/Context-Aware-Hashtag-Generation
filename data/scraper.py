"""
scraper.py — News article scraper for India Hashtag Generator
Scrapes: The Hindu, NDTV, Indian Express, The Wire, Scroll.in
Output:  data/raw/scraped_articles.csv
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

# ── Domain → Label mapping ─────────────────────────────────────────────────
# URL keyword patterns → domain label
# Used to auto-label articles from their URL path

URL_LABEL_MAP = [
    # Sports
    (["cricket", "ipl", "bcci", "t20", "odi", "test-match"],         "Sports", "Cricket"),
    (["football", "soccer", "fifa", "isl"],                           "Sports", "Football"),
    (["kabaddi", "pkl", "pro-kabaddi"],                               "Sports", "Kabaddi"),
    (["badminton", "tennis", "chess", "wrestling", "boxing"],         "Sports", "Athletics"),
    (["sport", "sports", "game", "tournament", "league"],             "Sports", "General"),

    # Politics
    (["election", "vote", "polling", "lok-sabha", "assembly-poll"],   "Politics", "Elections"),
    (["parliament", "lok-sabha", "rajya-sabha", "budget", "policy"],  "Politics", "Parliament"),
    (["modi", "rahul-gandhi", "bjp", "congress", "aap"],              "Politics", "CentralGovt"),
    (["state-politics", "cm-", "chief-minister"],                     "Politics", "StatePolitics"),
    (["politics", "political", "government", "minister"],             "Politics", "General"),

    # Crime
    (["rape", "sexual-assault", "pocso"],                             "Crime", "SexualViolence"),
    (["murder", "killing", "killed", "dead", "death"],                "Crime", "Murder"),
    (["lynching", "mob-violence", "mob-attack"],                      "Crime", "MobViolence"),
    (["scam", "fraud", "corruption", "bribe"],                        "Crime", "Corruption"),
    (["arrest", "fir", "police", "crime", "criminal"],                "Crime", "General"),

    # Social Issues
    (["dalit", "caste", "atrocity", "sc-st", "reservation"],         "SocialIssues", "Casteism"),
    (["communal", "riot", "hindu-muslim", "temple", "mosque"],        "SocialIssues", "Communalism"),
    (["honour-killing", "honor-killing", "dowry"],                    "SocialIssues", "GenderViolence"),
    (["farmer", "agriculture", "msp", "crop", "rural"],               "SocialIssues", "FarmerIssues"),
    (["tribal", "adivasi", "forest-rights"],                          "SocialIssues", "TribalRights"),
    (["lgbtq", "transgender", "gender"],                               "SocialIssues", "GenderRights"),
    (["social", "protest", "rally", "movement", "rights"],            "SocialIssues", "General"),

    # Business
    (["startup", "funding", "venture", "unicorn"],                    "Business", "Startup"),
    (["market", "sensex", "nifty", "stock", "share"],                 "Business", "StockMarket"),
    (["rbi", "repo-rate", "inflation", "gdp", "economy"],             "Business", "Economy"),
    (["adani", "ambani", "tata", "reliance", "infosys"],              "Business", "Conglomerates"),
    (["business", "company", "corporate", "industry"],                "Business", "General"),

    # Entertainment
    (["bollywood", "hindi-film", "box-office"],                       "Entertainment", "Bollywood"),
    (["kollywood", "tamil-film", "tollywood", "telugu-film"],         "Entertainment", "RegionalCinema"),
    (["ott", "netflix", "amazon-prime", "hotstar"],                   "Entertainment", "OTT"),
    (["celebrity", "award", "filmfare", "iifa"],                      "Entertainment", "Celebrity"),
    (["entertainment", "film", "movie", "cinema", "music"],           "Entertainment", "General"),

    # Science & Tech
    (["isro", "chandrayaan", "gaganyaan", "space", "satellite"],      "Science", "Space"),
    (["ai", "artificial-intelligence", "machine-learning"],           "Technology", "AI"),
    (["tech", "technology", "startup", "digital", "cyber"],           "Technology", "General"),
    (["science", "research", "discovery", "scientist"],               "Science", "General"),

    # Health
    (["covid", "coronavirus", "pandemic", "epidemic"],                "Health", "COVID"),
    (["hospital", "doctor", "patient", "disease", "virus"],           "Health", "General"),
    (["mental-health", "depression", "suicide"],                      "Health", "MentalHealth"),

    # Environment
    (["flood", "cyclone", "earthquake", "disaster"],                  "Environment", "Disaster"),
    (["climate", "pollution", "deforestation", "wildlife"],           "Environment", "Climate"),

    # Education
    (["neet", "jee", "upsc", "exam", "university", "school"],        "Education", "General"),
]


def infer_label(url: str, title: str) -> tuple:
    """Infer domain + sub_domain from URL and title."""
    text = (url + " " + title).lower()
    for keywords, domain, sub in URL_LABEL_MAP:
        if any(kw in text for kw in keywords):
            return domain, sub
    return "General", "General"


def clean_text(text: str) -> str:
    """Remove extra whitespace and newlines."""
    if not text:
        return ""
    return " ".join(text.split()).strip()


# ── Source scrapers ────────────────────────────────────────────────────────

def scrape_the_hindu(max_articles=300):
    """Scrape The Hindu — well structured, reliable labels."""
    print("\n[1/5] Scraping The Hindu...")
    articles = []

    sections = [
        "https://www.thehindu.com/sport/cricket/",
        "https://www.thehindu.com/sport/",
        "https://www.thehindu.com/news/national/",
        "https://www.thehindu.com/news/national/tamil-nadu/",
        "https://www.thehindu.com/news/national/other-states/",
        "https://www.thehindu.com/business/",
        "https://www.thehindu.com/entertainment/",
        "https://www.thehindu.com/sci-tech/",
        "https://www.thehindu.com/news/cities/",
    ]

    seen_urls = set()

    for section_url in sections:
        try:
            resp = requests.get(section_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"  Skipped {section_url} — HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # The Hindu article links are in <a> tags with /article in href
            links = soup.find_all("a", href=True)
            article_links = [
                a["href"] for a in links
                if "/article" in a["href"]
                and a["href"] not in seen_urls
                and len(a.get_text(strip=True)) > 20
            ]

            for link in article_links[:15]:  # max 15 per section
                if len(articles) >= max_articles:
                    break
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                try:
                    full_url = link if link.startswith("http") else "https://www.thehindu.com" + link
                    art_resp = requests.get(full_url, headers=HEADERS, timeout=15)
                    if art_resp.status_code != 200:
                        continue

                    art_soup = BeautifulSoup(art_resp.text, "html.parser")

                    # Extract title
                    title_tag = art_soup.find("h1", class_=lambda x: x and "title" in x.lower()) \
                             or art_soup.find("h1")
                    title = clean_text(title_tag.get_text()) if title_tag else ""
                    if not title or len(title) < 10:
                        continue

                    # Extract first 3 paragraphs as body
                    paragraphs = art_soup.find_all("p")
                    body = clean_text(" ".join(
                        p.get_text() for p in paragraphs[:5]
                        if len(p.get_text(strip=True)) > 50
                    ))

                    domain, sub = infer_label(full_url, title)

                    articles.append({
                        "source":     "TheHindu",
                        "url":        full_url,
                        "title":      title,
                        "body":       body[:500],
                        "domain":     domain,
                        "sub_domain": sub,
                        "scraped_at": datetime.now().isoformat()
                    })
                    print(f"  + {title[:70]}...")
                    time.sleep(1)  # polite delay

                except Exception as e:
                    continue

            time.sleep(2)

        except Exception as e:
            print(f"  Error on {section_url}: {e}")
            continue

    print(f"  ✓ {len(articles)} articles from The Hindu")
    return articles


def scrape_ndtv(max_articles=300):
    """Scrape NDTV — covers all domains well."""
    print("\n[2/5] Scraping NDTV...")
    articles = []

    sections = [
        "https://www.ndtv.com/cricket",
        "https://www.ndtv.com/sports",
        "https://www.ndtv.com/india",
        "https://www.ndtv.com/cities",
        "https://www.ndtv.com/business",
        "https://www.ndtv.com/entertainment",
        "https://www.ndtv.com/science",
        "https://www.ndtv.com/education",
    ]

    seen_urls = set()

    for section_url in sections:
        try:
            resp = requests.get(section_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"  Skipped {section_url} — HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            links = soup.find_all("a", href=True)
            article_links = [
                a["href"] for a in links
                if "ndtv.com" in a["href"]
                and a["href"] not in seen_urls
                and any(x in a["href"] for x in ["/news-", "/article-", "/story-"])
            ]

            for link in article_links[:15]:
                if len(articles) >= max_articles:
                    break
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                try:
                    resp2 = requests.get(link, headers=HEADERS, timeout=15)
                    if resp2.status_code != 200:
                        continue

                    soup2 = BeautifulSoup(resp2.text, "html.parser")

                    title_tag = soup2.find("h1")
                    title = clean_text(title_tag.get_text()) if title_tag else ""
                    if not title or len(title) < 10:
                        continue

                    paragraphs = soup2.find_all("p")
                    body = clean_text(" ".join(
                        p.get_text() for p in paragraphs[:5]
                        if len(p.get_text(strip=True)) > 50
                    ))

                    domain, sub = infer_label(link, title)

                    articles.append({
                        "source":     "NDTV",
                        "url":        link,
                        "title":      title,
                        "body":       body[:500],
                        "domain":     domain,
                        "sub_domain": sub,
                        "scraped_at": datetime.now().isoformat()
                    })
                    print(f"  + {title[:70]}...")
                    time.sleep(1)

                except Exception:
                    continue

            time.sleep(2)

        except Exception as e:
            print(f"  Error on {section_url}: {e}")
            continue

    print(f"  ✓ {len(articles)} articles from NDTV")
    return articles


def scrape_indian_express(max_articles=200):
    """Scrape Indian Express — strong on politics and social issues."""
    print("\n[3/5] Scraping Indian Express...")
    articles = []

    sections = [
        "https://indianexpress.com/section/india/",
        "https://indianexpress.com/section/politics/",
        "https://indianexpress.com/section/cities/",
        "https://indianexpress.com/section/sports/cricket/",
        "https://indianexpress.com/section/entertainment/",
        "https://indianexpress.com/section/business/",
        "https://indianexpress.com/section/technology/",
    ]

    seen_urls = set()

    for section_url in sections:
        try:
            resp = requests.get(section_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if "indianexpress.com" in a["href"]
                and a["href"].count("/") >= 4
                and a["href"] not in seen_urls
            ]

            for link in links[:12]:
                if len(articles) >= max_articles:
                    break
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                try:
                    resp2 = requests.get(link, headers=HEADERS, timeout=15)
                    soup2 = BeautifulSoup(resp2.text, "html.parser")

                    title_tag = soup2.find("h1")
                    title = clean_text(title_tag.get_text()) if title_tag else ""
                    if not title or len(title) < 10:
                        continue

                    paragraphs = soup2.find_all("p")
                    body = clean_text(" ".join(
                        p.get_text() for p in paragraphs[:5]
                        if len(p.get_text(strip=True)) > 50
                    ))

                    domain, sub = infer_label(link, title)

                    articles.append({
                        "source":     "IndianExpress",
                        "url":        link,
                        "title":      title,
                        "body":       body[:500],
                        "domain":     domain,
                        "sub_domain": sub,
                        "scraped_at": datetime.now().isoformat()
                    })
                    print(f"  + {title[:70]}...")
                    time.sleep(1)

                except Exception:
                    continue

            time.sleep(2)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"  ✓ {len(articles)} articles from Indian Express")
    return articles


def scrape_the_wire(max_articles=150):
    """Scrape The Wire — best for social issues, caste, communal topics."""
    print("\n[4/5] Scraping The Wire...")
    articles = []

    sections = [
        "https://thewire.in/politics",
        "https://thewire.in/rights",
        "https://thewire.in/caste",
        "https://thewire.in/communalism",
        "https://thewire.in/women",
        "https://thewire.in/government",
        "https://thewire.in/law",
    ]

    seen_urls = set()

    for section_url in sections:
        try:
            resp = requests.get(section_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if "thewire.in" in a["href"]
                and a["href"].count("/") >= 3
                and a["href"] not in seen_urls
                and not any(x in a["href"] for x in ["#", "?", "tag", "author"])
            ]

            for link in links[:12]:
                if len(articles) >= max_articles:
                    break
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                try:
                    resp2 = requests.get(link, headers=HEADERS, timeout=15)
                    soup2 = BeautifulSoup(resp2.text, "html.parser")

                    title_tag = soup2.find("h1")
                    title = clean_text(title_tag.get_text()) if title_tag else ""
                    if not title or len(title) < 10:
                        continue

                    paragraphs = soup2.find_all("p")
                    body = clean_text(" ".join(
                        p.get_text() for p in paragraphs[:5]
                        if len(p.get_text(strip=True)) > 50
                    ))

                    domain, sub = infer_label(link, title)

                    articles.append({
                        "source":     "TheWire",
                        "url":        link,
                        "title":      title,
                        "body":       body[:500],
                        "domain":     domain,
                        "sub_domain": sub,
                        "scraped_at": datetime.now().isoformat()
                    })
                    print(f"  + {title[:70]}...")
                    time.sleep(1)

                except Exception:
                    continue

            time.sleep(2)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"  ✓ {len(articles)} articles from The Wire")
    return articles


def scrape_scroll(max_articles=150):
    """Scrape Scroll.in — grassroots, rural, minority coverage."""
    print("\n[5/5] Scraping Scroll.in...")
    articles = []

    sections = [
        "https://scroll.in/article",
        "https://scroll.in/latest",
        "https://scroll.in/topic/caste",
        "https://scroll.in/topic/politics",
        "https://scroll.in/topic/cricket",
    ]

    seen_urls = set()

    for section_url in sections:
        try:
            resp = requests.get(section_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if "scroll.in/article/" in a["href"]
                and a["href"] not in seen_urls
            ]

            for link in links[:12]:
                if len(articles) >= max_articles:
                    break
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                try:
                    full_url = link if link.startswith("http") else "https://scroll.in" + link
                    resp2 = requests.get(full_url, headers=HEADERS, timeout=15)
                    soup2 = BeautifulSoup(resp2.text, "html.parser")

                    title_tag = soup2.find("h1")
                    title = clean_text(title_tag.get_text()) if title_tag else ""
                    if not title or len(title) < 10:
                        continue

                    paragraphs = soup2.find_all("p")
                    body = clean_text(" ".join(
                        p.get_text() for p in paragraphs[:5]
                        if len(p.get_text(strip=True)) > 50
                    ))

                    domain, sub = infer_label(full_url, title)

                    articles.append({
                        "source":     "Scroll",
                        "url":        full_url,
                        "title":      title,
                        "body":       body[:500],
                        "domain":     domain,
                        "sub_domain": sub,
                        "scraped_at": datetime.now().isoformat()
                    })
                    print(f"  + {title[:70]}...")
                    time.sleep(1)

                except Exception:
                    continue

            time.sleep(2)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"  ✓ {len(articles)} articles from Scroll.in")
    return articles


# ── Save to CSV ────────────────────────────────────────────────────────────

def save_to_csv(articles: list):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fieldnames = ["source", "url", "title", "body", "domain", "sub_domain", "scraped_at"]

    # Append if file exists, create if not
    file_exists = os.path.exists(OUTPUT_PATH)
    mode = "a" if file_exists else "w"

    with open(OUTPUT_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(articles)

    print(f"\n✓ Saved {len(articles)} articles to {OUTPUT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────

def run_all_scrapers():
    print("=" * 55)
    print("INDIA NEWS SCRAPER — All Sources")
    print("=" * 55)

    all_articles = []
    all_articles += scrape_the_hindu()
    all_articles += scrape_ndtv()
    all_articles += scrape_indian_express()
    all_articles += scrape_the_wire()
    all_articles += scrape_scroll()

    save_to_csv(all_articles)

    # Summary
    print("\n=== SCRAPE SUMMARY ===")
    from collections import Counter
    domain_counts = Counter(a["domain"] for a in all_articles)
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain:<20} {count} articles")
    print(f"\n  TOTAL: {len(all_articles)} articles")
    return all_articles


if __name__ == "__main__":
    run_all_scrapers()