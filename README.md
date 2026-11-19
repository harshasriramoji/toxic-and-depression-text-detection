# Toxic and Depression-Related Text Detection in Code-Mixed (Telugu-English) Text Using NLP

An end-to-end Natural Language Processing system engineered to identify **Depressive Ideation**, **Cyber-Toxicity / Harassment**, and **Normal Campus Discourse** in code-mixed social media text (Romanized Telugu + English / Tenglish).

---

## 📌 1. Project Overview & Clinical Context

On regional college forums, student confession pages, and campus messaging groups (such as NIT Warangal, KITS, Kakatiya University), students frequently communicate using a hybrid mixture of English and phonetic Latin-script Telugu (Tenglish). 


https://toxic-and-depression-text-detection.onrender.com

This project tackles two critical safety challenges:
1. **Depression Detection**: Detecting covert signs of severe mental distress, hopelessness, isolation, somatic anxiety, and suicidal ideation so that non-intrusive support and emergency helplines can be surfaced.
2. **Toxic Text Detection**: Flagging cyberbullying, ragging threats, abusive slurs, and hostile interpersonal harassment.
3. **Normal Discourse Classification**: Ensuring routine student conversations (campus events, food, festivals, notes sharing, sports) are not falsely flagged.

---

## 📊 2. Dataset Profile (`warangal_mental_health.csv`)

- **Total Samples**: 1,500 balanced rows (500 samples per class).
- **Format**: `text`, `label` (0, 1, 2), `label_name` (`normal`, `depression`, `toxic`).

| Class ID | Class Name | Distinctive Linguistic Markers | Example Post |
| :--- | :--- | :--- | :--- |
| **0** | `normal` | Campus food, festivals (*Bathukamma*, *Medaram*), cricket, GATE/placement prep, notes requests. | *"tonight Subedari daggara new biryani place try chesam, super ga undi"* |
| **1** | `depression` | Feelings of emptiness (*point ledu*, *meaning leni laga undi*), isolation (*okkadine kurchuni edusthunna*), insomnia, panic attacks. | *"Naku em chesina point ledu anipisthundi, life meaning leni laga undi"* |
| **2** | `toxic` | Slurs (*pichi pukku*, *bewarse*, *kukka*), ragging threats, insults (*get lost*, *loser*, *brain dead*). | *"honestly, enti ra pichi pukku, nee mind sarigga undadha, idiot like you should not post here"* |

---

## 🔬 3. NLP Architecture & Technical Methodology

### A. The Challenge of Code-Mixed Transliteration
Standard NLP models (e.g., standard VADER, Spacy, or English-only BERT) fail on Tenglish because:
- **Phonetic Inconsistency**: The same Telugu word can be spelled in multiple ways:
  - *anipisthundi* vs *anipistundi* vs *anipistundhi* (I feel)
  - *edusthunna* vs *yedustunna* (crying)
  - *chachipothunna* vs *chachipothanu* (feel like dying)
- **High Out-of-Vocabulary (OOV) Rate**: Word-level tokenizers fragment transliterated words into unknown tokens.

### B. Hybrid Feature Extraction (Word + Character N-Grams)
To solve this, our pipeline deploys a `FeatureUnion`:
1. **Character N-Grams (2 to 5 characters)**: Captures root stems, morphosyntactic suffixes (`-thunna`, `-ledu`, `-bhaaram`), and phonetic variations.
2. **Word N-Grams (1 to 2 words)**: Captures crucial multi-word phrases (`"anxiety attack"`, `"no motivation"`, `"get fired"`).
3. **Sublinear TF Scaling**: Dampens the effect of repetitive high-frequency terms.

### C. Calibrated Probability Ensemble
- **LinearSVC with `CalibratedClassifierCV`**: Produces well-calibrated posterior probabilities.
- **Regularized Logistic Regression**: Provides balanced log-odds probabilities.
- **Weighted Blending**: Optimized validation-set blending weight for maximal macro F1-score.

---

## ⚠️ 4. Critical Assessment: The "100% Accuracy" Trap

In the Google Colab notebook (`nlp.ipynb`), the classical models and BiLSTM achieved **100% Macro-F1**. 

### Why did this happen?
1. **Template Synthesization**: The dataset was constructed using repeating structural templates:
   - *"idi evarikee cheppaledu, [depression phrase] in [location]"*
   - *"honestly, [insult] in [location] lo nee place ledu"*
   - *"rain ayyaka [location] lo mirchi bajji tinadam oka feeling"*
2. **Shortcut Learning**: The models memorized template prefixes and specific regional place names rather than deep linguistic semantics.

### Production Recommendations:
- **Data Augmentation**: Introduce real-world typos, colloquial abbreviations, and out-of-domain slang.
- **Transliteration Normalization**: Map Latin-script Telugu to native Telugu script (using Soundex or Indic transliteration engines) to test native IndicBERT.
- **Adversarial Validation**: Evaluate on posts that do not mention any of the template locations (Warangal Fort, Kazipet, Hanamkonda).

---

## 🚑 5. Clinical Safety & Crisis Intervention Protocol

When a depression probability exceeds 65%, the inference engine triggers an automated **Crisis Intervention Protocol**, presenting toll-free 24/7 mental health helplines:
- **Tele-MANAS (Govt. of India)**: `14416` or `1800-891-4416` (24/7 Toll-free)
- **KIRAN Helpline**: `1800-599-0019`
- **Vandrevala Foundation**: `+91 9999 666 555`

---

## 🚀 6. Quick Start & Execution

### 1. Train the Models
```bash
python -m pip install -r requirements.txt
python -m src.train
```

### 2. Run Inference Smoke Test
```bash
python -m src.inference
```

### 3. Launch the Interactive Streamlit Web App
```bash
python -m streamlit run streamlit_app.py
```

### Local accounts
The dashboard opens with a login/register screen. Register with either a username
(3–32 letters, numbers, `.`, `_` or `-`) or a valid email address, and use a
password between 8 and 10 characters. Accounts are stored in
`data/accounts.sqlite3`; passwords are salted and hashed.
This local prototype has no password recovery. For a public deployment, use a
dedicated authentication provider and secure persistent storage rather than
treating this local account database as production identity management.
