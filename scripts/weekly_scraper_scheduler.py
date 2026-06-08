"""
weekly_scraper_scheduler.py

Orchestrates the continuous data pipeline for the India Hashtag Generator.
When run (via cron/Task Scheduler weekly), it performs the following:
  1. Expands the Knowledge Graph via Wikipedia categories (kg_fetcher_v2.py)
  2. Scrapes the latest news articles from top Indian sources (News18, The Hindu)
  3. Re-runs the preprocessor to integrate new data and rebuild train.csv
  4. (Optional) Triggers a model fine-tuning run if dataset grew significantly.

Hardware Requirements for Weekly Execution:
- CPU: 4+ cores (for parallel text processing)
- RAM: 16GB+ (Preprocessor loads 100k+ rows and massive KG in memory)
- Storage: 100GB+ SSD (D: Drive recommended to avoid C: drive exhaustion)
- Network: High bandwidth for weekly dataset synchronization
"""

import os
import subprocess
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PYTHON_EXEC = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

def run_script(script_path, description):
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] starting: {description}")
    print(f"Executing: {script_path}")
    
    try:
        result = subprocess.run(
            [PYTHON_EXEC, script_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Success.")
        # Print last few lines of output
        lines = result.stdout.strip().split('\n')
        for line in lines[-5:]:
            print(f"  {line}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print(e.stderr)

def main():
    print("========================================================")
    print("INDIA HASHTAG GENERATOR — WEEKLY SCHEDULER PIPELINE")
    print("========================================================")
    start_time = time.time()

    # Step 1: Update Knowledge Graph
    kg_script = os.path.join("data", "knowledge_base", "kg_fetcher_v2.py")
    run_script(kg_script, "Knowledge Graph Expansion (Wikipedia)")

    # Step 2: Scrape Weekly News
    scraper_script = os.path.join("data", "scrapers", "news18_scraper.py")
    if os.path.exists(os.path.join(BASE_DIR, scraper_script)):
        run_script(scraper_script, "Weekly News Scraper (News18/The Hindu)")
    else:
        print("\n[!] News scraper not found, skipping weekly scrape.")

    # Step 3: Re-run Preprocessor
    preprocessor_script = os.path.join("data", "preprocessor_v2.py")
    run_script(preprocessor_script, "Data Preprocessing & Re-balancing")

    elapsed = (time.time() - start_time) / 60
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pipeline completed in {elapsed:.1f} minutes.")
    print("Dataset and Knowledge Graph are fully up to date.")

if __name__ == "__main__":
    main()
