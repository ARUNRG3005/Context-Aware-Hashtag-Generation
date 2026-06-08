# 📊 Grassroots Data & KG Audit Report

This document outlines the exact composition, sourcing, and internal mechanics of the 47,949 row dataset and the current Knowledge Graph (KG) architecture.

---

## 1. Exact Dataset Numbers & Sources (47,949 Total Rows)
We fed 300,000+ global articles into the pipeline. After extreme filtering for relevance to the Indian context and deduping, here is exactly where the final 47,949 rows came from:

| Source Dataset | Usable Rows | Originating Journals / Channels |
| :--- | :--- | :--- |
| **HF CNN/DailyMail** | 49,626* | CNN, The Daily Mail (Global, filtered for India) |
| **HF XSum** | 14,785 | BBC News (Highly compressed summaries) |
| **HF Indian Financial News** | 12,773 | Livemint, The Economic Times, Moneycontrol |
| **AG News** | 10,687 | Reuters, Associated Press (AP), New York Times, Yahoo |
| **HF HuffPost** | 3,582 | The Huffington Post (India & World sections) |
| **BBC Standard** | 2,749 | BBC News Archive |
| **Custom Scraper** | ~1,100 | The Hindu, Times of India, Scroll.in, Indian Express, News18 |
| **HF Argilla News** | 368 | Mixed open-source news aggregation |

*( *Note: CNN/DailyMail provided 49k raw matches, but the final dataset was strictly capped at 10,000 max per domain to prevent AI bias, bringing the total balanced count to 47,949 rows.)*

---

## 2. How the Data is Processed (The Grassroots Pipeline)
Every single raw article undergoes a strict 4-step processing pipeline inside `preprocessor_v2.py`:

1. **Text Cleansing:** HTML tags, URLs, and bizarre formatting characters are stripped out.
2. **Relevance Gatekeeper:** The article is scanned. If it does not contain an Indian keyword (e.g., "India", "Delhi", "BCCI", "Modi") or if the KG does not recognize an entity inside it, the article is permanently deleted. 
3. **KG Entity Extraction:** The text is scanned against the SQLite Knowledge Graph (11,049 entities). If a match is found (e.g., the text mentions "Virat Kohli"), the KG injects the tags (`#Cricket`, `#athlete`) and forces the row into the corresponding domain (`Sports`).
4. **Relationship & Sensitive Hunting:** The text is scanned for complex relationships. If the script sees "Enforcement Directorate" and "raids", it injects `#Corruption` and shifts the domain to `Crime`. If it sees "CSK vs MI", it injects `#CSKvsMI`.
5. **Balancing Filter:** It counts how many articles exist in a domain. If `Sports` hits 10,000, any further sports articles are discarded.

---

## 3. Knowledge Graph (KG) Map Architecture
The KG is a local SQLite database (`india_kg.db`) that currently holds **11,049 entities** and **48,938 hashtag mapping rules**.

### Wikipedia Categories Used by `kg_fetcher_v2.py`
The fetcher automatically scrapes Wikipedia for every page existing under these exact categories:

*   **Sports:** `Category:Indian_cricketers`, `Category:Indian_athletes`, `Category:Football_in_India`
*   **Business:** `Category:Indian_businesspeople`, `Category:Companies_of_India`
*   **Entertainment:** `Category:Hindi-language_films`, `Category:Telugu-language_films`, `Category:Tamil-language_films`, `Category:Indian_film_directors`
*   **Politics:** `Category:Political_parties_in_India`, `Category:Indian_politicians`, `Category:Elections_in_India`
*   **Science:** `Category:Indian_scientists`, `Category:Science_and_technology_in_India`
*   **Places:** `Category:Cities_in_India`, `Category:States_and_union_territories_of_India`
*   **Environment:** `Category:National_parks_of_India`, `Category:Rivers_of_India`, `Category:Fauna_of_India`, `Category:Climate_change_in_India`
*   **Crime:** `Category:Indian_criminal_law`, `Category:Law_enforcement_agencies_of_India`, `Category:Social_issues_in_India`, `Category:Scandals_in_India`

*(Note: The Wikipedia API recently rate-limited us when trying to scrape the Health and Education categories, but the script is hardcoded to retry them during its next scheduled run).*
