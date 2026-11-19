"""
Text Preprocessing and Normalization Module for Code-Mixed (Telugu-English / Tenglish) Text
"""
import re
import unicodedata

# Common transliteration variations and contractions in Telugu-English text
PHONETIC_REPLACEMENTS = {
    r"\b(ani|ani\s*pisthundi|anipisthondi|anipistundi|anipistundhi)\b": "anipisthundi",
    r"\b(edusthunna|yedusthunna|edustunna|yedustunna)\b": "edusthunna",
    r"\b(chachipothunna|chachipothanu|chachipovali)\b": "chachipothunna",
    r"\b(bhaarame|bharame|bhaaram)\b": "bhaarame",
    r"\b(nidra|nidhra)\b": "nidra",
    r"\b(teliyatledu|theliyatledu|teliyadamledu)\b": "teliyatledu",
    r"\b(matladalani|matladali)\b": "matladalani",
    r"\b(kastam|kashtam)\b": "kastam",
}

def clean_text(text: str) -> str:
    """
    Cleans and standardizes raw input text while preserving meaningful code-mixed cues,
    emojis, and phonetic characteristics.
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize unicode (decompose accented/composite characters)
    text = unicodedata.normalize("NFKD", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle excessive repeated characters (e.g., 'sooooo tired' -> 'soo tired', 'edupuuu' -> 'edupuu')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Standardize common phonetic spellings in Tenglish
    for pattern, replacement in PHONETIC_REPLACEMENTS.items():
        text = re.sub(pattern, replacement, text)
    
    # Normalize excessive whitespace and punctuation spacing
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_code_mixed(text: str):
    """
    Tokenizes text into word-level tokens, capturing both alphabetic words and emojis/symbols.
    """
    cleaned = clean_text(text)
    tokens = re.findall(r"\w+|[^\w\s]", cleaned, flags=re.UNICODE)
    return tokens
