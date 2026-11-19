"""
Comprehensive Deep Exploratory Data Analysis (EDA) for Code-Mixed Mental Health & Toxic Dataset
Analyzes lexical richness, code-mixing characteristics, syntactic templates, entity distributions, and class semantics.
"""
import os
import re
from collections import Counter
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "warangal_mental_health.csv")

WARANGAL_LOCATIONS = [
    "NIT Warangal", "KITS Warangal", "Kakatiya University", "Hanamkonda",
    "Kazipet", "Subedari", "Hunter Road", "Station Road", "Mulugu Road",
    "Bhadrakali", "Thousand Pillar Temple", "Warangal Fort", "Waddepally",
    "Ramappa", "Balasamudram", "Naimnagar", "Pochamma Maidan", "Kishanpura",
    "Narsampet", "MGM Hospital", "Kakaji Colony", "Chowrasta", "Hanmakonda Chowrasta"
]

def tokenize_words(text):
    return re.findall(r"\b[a-zA-Z]+\b", str(text).lower())

def run_deep_eda():
    df = pd.read_csv(DATA_PATH)
    print("=" * 80)
    print("        DEEP CORPUS & LINGUISTIC ANALYSIS: WARANGAL MENTAL HEALTH DATASET")
    print("=" * 80)
    
    total_samples = len(df)
    print(f"\n1. DATASET OVERVIEW:")
    print(f"   - Total Postings / Records: {total_samples}")
    print(f"   - Number of Classes: {df['label_name'].nunique()} ({list(df['label_name'].unique())})")
    print(f"   - Missing Values: {df.isnull().sum().to_dict()}")
    print(f"   - Duplicate Texts: {df['text'].duplicated().sum()}")
    
    # 2. Text Length and Lexical Statistics
    df["word_count"] = df["text"].apply(lambda t: len(tokenize_words(t)))
    df["char_count"] = df["text"].str.len()
    
    print("\n2. CORPUS LENGTH METRICS BY CLASS:")
    length_summary = df.groupby("label_name").agg({
        "word_count": ["mean", "std", "min", "max", "median"],
        "char_count": ["mean", "std", "min", "max", "median"]
    })
    print(length_summary.round(2))
    
    # 3. Vocabulary Richness & Type-Token Ratio (TTR)
    print("\n3. VOCABULARY DIVERSITY & TYPE-TOKEN RATIO (TTR):")
    for cat in ["normal", "depression", "toxic"]:
        sub = df[df["label_name"] == cat]
        all_tokens = [tok for t in sub["text"] for tok in tokenize_words(t)]
        vocab = set(all_tokens)
        ttr = len(vocab) / len(all_tokens) if all_tokens else 0
        print(f"   [{cat.upper():10s}] Tokens: {len(all_tokens):5d} | Unique Vocab: {len(vocab):4d} | TTR: {ttr:.4f}")
        
    # 4. Top Distinctive Keywords & N-Grams
    print("\n4. TOP DISTINCTIVE N-GRAMS PER CLASS:")
    for cat in ["normal", "depression", "toxic"]:
        sub = df[df["label_name"] == cat]
        tokens = [tok for t in sub["text"] for tok in tokenize_words(t)]
        # Filter common generic stopwords for cleaner topical signal
        stop = {"lo", "ga", "tho", "ani", "ki", "ee", "andaru", "undi", "chesthunna", "naa", "naku", "nenu", "nuvvu", "the", "a", "an", "in", "to", "at", "for", "is", "of", "and"}
        filtered = [tok for tok in tokens if tok not in stop and len(tok) > 2]
        top_unigrams = Counter(filtered).most_common(8)
        
        # Bigrams
        bigrams = [
            f"{tok1} {tok2}"
            for t in sub["text"]
            for tok1, tok2 in zip(tokenize_words(t)[:-1], tokenize_words(t)[1:])
        ]
        top_bigrams = Counter(bigrams).most_common(5)
        
        print(f"\n   >>> Class: {cat.upper()} <<<")
        print(f"   Top Keywords: {', '.join([f'{w} ({c})' for w, c in top_unigrams])}")
        print(f"   Top Bigrams:  {', '.join([f'\"{bg}\" ({c})' for bg, c in top_bigrams])}")

    # 5. Location Entity Mention Frequency
    print("\n5. REGIONAL / CAMPUS ENTITY DISTRIBUTION:")
    loc_counts = {loc: 0 for loc in WARANGAL_LOCATIONS}
    loc_by_class = {cat: 0 for cat in ["normal", "depression", "toxic"]}
    
    for _, row in df.iterrows():
        t = row["text"].lower()
        c = row["label_name"]
        for loc in WARANGAL_LOCATIONS:
            if loc.lower() in t:
                loc_counts[loc] += 1
                loc_by_class[c] += 1
                
    total_loc_mentions = sum(loc_counts.values())
    print(f"   - Total Location Mentions: {total_loc_mentions} across {total_samples} samples")
    print(f"   - Location Distribution by Class: {loc_by_class}")
    top_locs = sorted(loc_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    print(f"   - Most Frequent Campus / City Locales: {top_locs}")
    
    # 6. Syntactic Template Repetition Analysis
    print("\n6. SYNTACTIC TEMPLATE FREQUENCY (SHORTCUT LEAKAGE RISK):")
    templates = {
        "anxiety attack exam": r"anxiety attack.*exam ki mundu",
        "mirchi bajji rain": r"rain ayyaka.*mirchi bajji",
        "ragging teach lesson": r"ragging chestham.*lesson tonight",
        "shut filthy mouth": r"shut your filthy mouth",
        "waste of space": r"waste of space",
        "edupu vastundi door close": r"edupu vastundi.*door close",
        "burden to parents": r"burden to my parents",
        "gate study motivated": r"studying for gate.*feeling motivated"
    }
    for name, pattern in templates.items():
        matches = df["text"].str.contains(pattern, case=False, regex=True).sum()
        print(f"   - Template '{name}': {matches} occurrences ({matches/total_samples*100:.1f}% of entire corpus)")

if __name__ == "__main__":
    run_deep_eda()
