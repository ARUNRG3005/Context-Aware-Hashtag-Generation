import sqlite3
import os
import pandas as pd
import spacy
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "knowledge_base", "india_kg.db")
TRAIN_CSV = os.path.join(BASE_DIR, "data", "processed", "train.csv")

def load_entities():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT wikidata_id, name FROM entities WHERE length(name) > 4")
    entities = {}
    for row in cur.fetchall():
        entities[row[1].lower()] = row[0]
    conn.close()
    return entities

def extract_spacy_relationships():
    print("Loading SpaCy en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm")
    
    entities = load_entities()
    entity_names = list(entities.keys())
    entity_names.sort(key=len, reverse=True) # match longest first
    
    print("Loading training data...")
    df = pd.read_csv(TRAIN_CSV)
    texts = df['text'].dropna().tolist()
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    relationships_found = 0
    start_time = time.time()
    
    # Process all articles
    # texts = texts[:2000]
    print(f"Scanning {len(texts)} articles using SpaCy Dependency Parsing...")
    
    for i, text in enumerate(texts):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i} articles... Found {relationships_found} relationships.")
            
        # First do a fast pass: don't run heavy SpaCy unless there are at least 2 entities
        text_lower = text.lower()
        found_names = []
        for name in entity_names:
            if name in text_lower:
                found_names.append(name)
                if len(found_names) > 3:
                    break
                    
        if len(found_names) < 2:
            continue
            
        # If there are entities, run SpaCy!
        doc = nlp(text)
        
        # Build mapping of tokens to entities (simplistic multi-word mapping)
        token_to_entity = {}
        for token in doc:
            for name in found_names:
                if token.text.lower() in name and len(token.text) > 3:
                    token_to_entity[token] = name
                    break
        
        # We look for VERBS
        for token in doc:
            if token.pos_ == "VERB":
                # Check its children to find the Subject and the Object
                subjects = []
                objects = []
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subjects.append(child)
                    elif child.dep_ in ["dobj", "pobj", "attr"]:
                        objects.append(child)
                        
                # Recursively check the grammatical sub-tree of the subject and object
                # to see if an Indian Entity is within them.
                for subj in subjects:
                    subj_tokens = list(subj.subtree)
                    subj_entities = [token_to_entity[t] for t in subj_tokens if t in token_to_entity]
                    
                    for obj in objects:
                        obj_tokens = list(obj.subtree)
                        obj_entities = [token_to_entity[t] for t in obj_tokens if t in token_to_entity]
                        
                        if subj_entities and obj_entities:
                            sub_e = subj_entities[0]
                            obj_e = obj_entities[0]
                            predicate = token.lemma_.upper()
                            
                            if sub_e != obj_e:
                                try:
                                    cur.execute("""
                                        INSERT OR IGNORE INTO relationships (subject_id, predicate, object_id)
                                        VALUES (?, ?, ?)
                                    """, (entities[sub_e], predicate, entities[obj_e]))
                                    if cur.rowcount > 0:
                                        relationships_found += 1
                                except Exception:
                                    pass

    conn.commit()
    conn.close()
    elapsed = time.time() - start_time
    print(f"\nSpaCy Extraction Complete in {elapsed:.2f} seconds!")
    print(f"Total High-Accuracy Relationships Extracted: {relationships_found}")

if __name__ == "__main__":
    extract_spacy_relationships()
