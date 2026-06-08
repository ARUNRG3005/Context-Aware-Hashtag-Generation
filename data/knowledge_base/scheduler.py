"""
scheduler.py — Weekly auto-refresh of the India Knowledge Graph.
Run once: python scheduler.py
It will fetch updates every Sunday at 2 AM.
"""

import schedule
import time
from kg_fetcher import run_all


def update_knowledge_base():
    print("=" * 50)
    print("SCHEDULED UPDATE — Weekly KG Refresh")
    print("=" * 50)
    try:
        total = run_all()
        print(f"Scheduled update complete — {total} entities refreshed.")
    except Exception as e:
        print(f"Scheduled update failed: {e}")


def schedule_weekly_update():
    schedule.every().sunday.at("02:00").do(update_knowledge_base)
    print("Scheduler started — weekly update every Sunday at 02:00")
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "now":
        # run immediately (for testing)
        update_knowledge_base()
    else:
        schedule_weekly_update()
