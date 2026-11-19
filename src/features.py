"""
Feature Extraction Module using Hybrid Word and Character N-Gram TF-IDF
"""
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import clean_text

def build_hybrid_tfidf():
    """
    Constructs a FeatureUnion combining:
    1. Word-level TF-IDF (1-2 n-grams) to capture semantic phrases
    2. Character-level TF-IDF (2-5 n-grams) to capture subword morphemes,
       transliteration variations, and Telugu suffixes (-thunna, -ledu, -bhaaram).
    """
    word_vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=40000,
        sublinear_tf=True
    )
    
    char_vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=60000,
        sublinear_tf=True
    )
    
    hybrid_feature_extractor = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer)
    ])
    
    return hybrid_feature_extractor
