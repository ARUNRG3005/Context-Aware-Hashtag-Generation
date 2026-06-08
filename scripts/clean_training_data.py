import pandas as pd
import re
import sqlite3

def clean_dataset(path):
    print(f"Cleaning {path}...")
    df = pd.read_csv(path)
    
    noisy_patterns = [
        r"memberthe.*loksabha",
        r"memberrajyasabha",
        r"memberthe.*legislativeassembly",
        r"chiefminister.*",
        r"minister.*",
        r"primeminister.*",
        r"^singh$", r"^cameron$", r"^helen$", r"^athlete$", r"^actor$", r"^vincent$",
        r"^hassan$", r"^business$", r"^entrepreneur$", r"^congress$", r"^inc$", r"^bcci$",
        r"^cricket$", r"^indiancricket$", r"^teamindia$", r"^ipl$", r"^bengalpolitics$",
        r"^india$", r"^infrastructure$", r"^transit$"
    ]
    
    replacements = {
        "inc": "INC",
        "congress": "INC",
        "bcci": "BCCI",
        "ipl": "IPL",
        "politics": "IndianPolitics",
        "government": "IndianPolitics",
        "cricket": "Cricket",
        "indiancricket": "IndianCricket",
        "teamindia": "TeamIndia",
        "business": "Business",
        "entrepreneur": "Business"
    }
    
    def process_labels(label_str):
        if not isinstance(label_str, str):
            return ""
        tags = [t.strip() for t in label_str.split('|') if t.strip()]
        new_tags = set()
        for t in tags:
            # Check noise
            is_noise = False
            for p in noisy_patterns:
                if re.match(p, t, re.IGNORECASE):
                    is_noise = True
                    break
            if is_noise:
                continue
                
            # Replacements
            t_lower = t.lower()
            if t_lower in replacements:
                new_tags.add(replacements[t_lower])
            else:
                new_tags.add(t)
        return "|".join(sorted(new_tags))

    df['labels'] = df['labels'].apply(process_labels)
    # Remove rows that have no labels left
    df = df[df['labels'] != ""]
    df.to_csv(path, index=False)
    print(f"Cleaned {path}: {len(df)} rows remaining.")

def clean_kg():
    print("Cleaning Knowledge Graph...")
    conn = sqlite3.connect('data/knowledge_base/india_kg.db')
    c = conn.cursor()
    
    # Delete garbage tags assigned during faulty script generation
    noise_tags = [
        'infrastructure', 'transit', 'cameron', 'athlete', 'helen', 'singh', 
        'actor', 'hassan', 'vincent'
    ]
    for tag in noise_tags:
        c.execute("DELETE FROM tags WHERE tag = ?", (tag,))
        
    conn.commit()
    print(f"Deleted {conn.total_changes} noisy tag mappings from KG.")
    conn.close()

if __name__ == "__main__":
    clean_dataset("data/processed/train.csv")
    clean_dataset("data/processed/val.csv")
    clean_dataset("data/processed/test.csv")
    clean_kg()
    print("All cleaning complete!")
