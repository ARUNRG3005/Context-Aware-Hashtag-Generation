import os
import sys

# Setup paths to import from data folder
BASE = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE, 'data'))

from preprocessor_v2 import kg_boost, extract_relationship_labels

# Three brand new news snippets completely unseen by the model/train data
FRESH_NEWS = [
    # 1. Sports / Relationship test
    """
    In a thrilling final over, the Gujarat Titans managed to hit a boundary off the last ball,
    securing a dramatic victory over the Rajasthan Royals in front of a packed stadium in Ahmedabad.
    Hardik Pandya was named man of the match for his incredible performance.
    """,
    
    # 2. Politics / Corruption / Sensitive test
    """
    The Enforcement Directorate early Wednesday morning conducted massive raids across multiple
    properties linked to senior Congress leaders in Delhi and Mumbai. The raids are allegedly
    connected to a multi-crore money laundering scam. Several documents and uncounted cash were seized.
    """,
    
    # 3. Crime / Gender Violence / Sensitive test
    """
    Massive protests erupted outside the local police station in Uttar Pradesh after a 19-year-old
    college student was subjected to a horrific sexual assault. Activists are demanding immediate
    action against the perpetrators, stating that instances of gender-based violence have been rising in the district.
    """
]

def main():
    print("=" * 70)
    print("LIVE PIPELINE TEST (Fresh Unseen Data)")
    print("=" * 70)
    
    for i, text in enumerate(FRESH_NEWS, 1):
        print(f"\n--- Fresh News #{i} ---")
        clean_txt = text.strip()
        print(f"TEXT: {clean_txt}")
        
        # 1. KG Boost (Simulates what the preprocessor and the predictor both do)
        # Note: kg_boost might need the KG loaded. preprocessor_v2 does it globally if imported!
        kg_domain, tags = kg_boost(clean_txt, [])
        
        # 2. Relationship / Sensitive Label Extraction
        final_tags = extract_relationship_labels(clean_txt, tags)
        
        print("\n-> EXTRACTED LABELS:")
        for tag in sorted(set(final_tags)):
            print(f"   #{tag}")
        print(f"-> DOMAIN SHIFT: {kg_domain if kg_domain else 'None'}")
        
if __name__ == "__main__":
    main()
