import requests

query = """
SELECT DISTINCT ?player ?playerLabel WHERE {
  ?player wdt:P106 wd:Q12299841 .
  ?player wdt:P27 wd:Q668 .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
LIMIT 10
"""

r = requests.get(
    "https://query.wikidata.org/sparql",
    params={"query": query, "format": "json"},
    headers={"User-Agent": "IndiaHashtagKG/1.0"},
    timeout=20
)
results = r.json()["results"]["bindings"]
print(f"Count: {len(results)}")
for row in results:
    print(f"  - {row.get('playerLabel', {}).get('value', '?')}")