"""Interactive dashboard for code-mixed mental-health and toxicity text analysis."""
import os
import re
from io import BytesIO
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from xml.sax.saxutils import escape

from src.inference import MentalHealthDetector
from src.auth import initialize_auth_db, register_user, verify_user
from src.preprocessing import clean_text


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "warangal_mental_health.csv")
PAGES = [
    "Home",
    "Text Analyzer",
    "Model Comparison",
    "Batch Analyzer",
    "AI Insights",
    "Dashboard",
    "Model Performance",
    "History",
    "About",
]
LABEL_ICONS = {
    "normal": "🟢",
    "depression": "🟡",
    "toxic": "🔴",
}
SAMPLE_PROMPTS = {
    "Distress and hopelessness": (
        "Naku em chesina point ledu anipisthundi, life meaning leni laga undi"
    ),
    "Anxiety before exams": (
        "anxiety attack vachindi exam ki mundu, hands vanukuthunnayi, "
        "breathe cheyyalekapoyanu"
    ),
    "Hostile comment": (
        "honestly, enti ra pichi pukku, nee mind sarigga undadha, "
        "idiot like you should not post here"
    ),
    "Threatening language": (
        "Bro, nuvvu anta nonsense matladaku, stupid creature, "
        "I will make your life hell in college ra"
    ),
    "Campus food": (
        "tonight Subedari daggara new biryani place try chesam, super ga undi"
    ),
    "Study question": (
        "this morning anyone have notes for machine learning unit 3? please share guys"
    ),
}
LEXICAL_CUES = {
    "Distress / sadness": {
        "sad", "cry", "crying", "edusthunna", "lonely", "alone", "hopeless",
        "empty", "depressed", "meaningless", "burden", "tired",
    },
    "Anxiety / stress": {
        "anxiety", "anxious", "panic", "worry", "worried", "fear", "exam",
        "stress", "tension", "breathe",
    },
    "Anger / hostility": {
        "angry", "hate", "idiot", "stupid", "loser", "useless", "pathetic",
        "pichi", "bewarse", "threat", "hell",
    },
    "Positive / social": {
        "happy", "super", "nice", "good", "thanks", "celebration", "share",
        "biryani", "cricket", "festival",
    },
}

st.set_page_config(
    page_title="Tenglish Insights | Text Safety Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#e5e7eb; --muted:#a1a1aa; --line:#303747; --brand:#818cf8; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp { background:radial-gradient(ellipse at 50% -20%,#202844 0%,#111827 48%,#0b1020 100%); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(11,16,32,.88); }
    [data-testid="stSidebar"] { background:#0f172a; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 { font-family:'Manrope',sans-serif; color:#f8fafc; }
    [data-testid="stSidebar"] [role="radiogroup"] label { color:#cbd5e1; }
    h1,h2,h3,h4,p,li,label { color:var(--ink); }
    h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.03em; }
    [data-testid="stMarkdownContainer"] a { color:#a5b4fc; }
    [data-testid="stCaptionContainer"], .subtle { color:var(--muted); }
    .hero { padding:2rem 2.2rem; border-radius:24px; color:white;
        background:linear-gradient(120deg,#1e1b4b 0%,#3730a3 52%,#6d28d9 100%);
        border:1px solid rgba(165,180,252,.2);
        box-shadow:0 18px 45px rgba(0,0,0,.32); margin-bottom:1.2rem; }
    .hero h1,.hero p,.hero .eyebrow { color:#f8fafc!important; }
    .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.75rem;
        font-weight:700; opacity:.78; }
    .section-card, [data-testid="stMetric"], [data-testid="stExpander"] {
        background:#151d2e; border:1px solid var(--line); border-radius:16px; color:var(--ink); }
    .section-card { padding:1.1rem 1.25rem; }
    [data-testid="stMetric"] { padding:1rem 1.1rem; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:#e5e7eb; }
    [data-testid="stWidgetLabel"] p { color:#cbd5e1; }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stFileUploader"] section {
        background:#111827; color:#f1f5f9; border-color:#3a465b; }
    [data-testid="stTextArea"] textarea::placeholder, [data-testid="stTextInput"] input::placeholder {
        color:#94a3b8; }
    [data-testid="stSelectbox"] [data-baseweb="select"] * { color:#e5e7eb; }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background:#151d2e; color:#e5e7eb; }
    [role="option"] { color:#e5e7eb; }
    [data-testid="stAlert"] { background:#172033; color:#e5e7eb; border:1px solid #3a465b; }
    [data-testid="stCode"], pre, code { background:#0b1220; color:#c7d2fe; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    [data-testid="stDataFrame"] iframe { color-scheme:dark; }
    [data-testid="stTable"] { color:#e5e7eb; }
    [data-testid="stProgress"] > div { background:#263248; }
    .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
        border-radius:12px; border:1px solid #3a465b; background:#1b2538; color:#e5e7eb; }
    .stButton>button:hover, .stDownloadButton>button:hover {
        border-color:#818cf8; color:#c7d2fe; background:#252f46; }
    .stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] {
        background:linear-gradient(100deg,#4f46e5,#7c3aed); border-color:#818cf8; color:white; }
    hr { border-color:var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

AUTH_DB_PATH = os.environ.get(
    "TENGLISH_AUTH_DB", os.path.join(BASE_DIR, "data", "accounts.sqlite3")
)

@st.cache_resource
def get_detector():
    return MentalHealthDetector()


@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


def render_auth_page(database_path):
    st.markdown(
        """
        <div class="hero">
            <h1>Sign in to explore the dashboard</h1>
            <p>Create a local account or sign in to continue to the text analysis tools.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        mode = st.radio("Account", ["Log in", "Register"], horizontal=True)
        with st.form("account_form"):
            username = st.text_input(
                "Username or email", max_chars=254, autocomplete="username"
            )
            password = st.text_input(
                "Password",
                type="password",
                max_chars=10,
                autocomplete="current-password" if mode == "Log in" else "new-password",
            )
            confirm_password = ""
            if mode == "Register":
                confirm_password = st.text_input(
                    "Confirm password", type="password", max_chars=10
                )
            submitted = st.form_submit_button(
                mode, type="primary", use_container_width=True
            )

        if submitted:
            if mode == "Register" and password != confirm_password:
                st.error("The passwords do not match.")
            else:
                try:
                    if mode == "Register":
                        created = register_user(database_path, username, password)
                        if not created:
                            st.error("That username is already registered.")
                        else:
                            st.session_state["auth_user"] = username.strip()
                            st.rerun()
                    elif verify_user(database_path, username, password):
                        st.session_state["auth_user"] = username.strip()
                        st.rerun()
                    else:
                        st.error("Username or password is incorrect.")
                except ValueError as error:
                    st.error(str(error))

        st.caption(
            "Accounts are stored on this machine. This prototype does not provide "
            "password recovery or production-grade account management."
        )


def add_history(result, source="Text Analyzer"):
    records = st.session_state.setdefault("analysis_history", [])
    records.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "text": result["input_text"],
            "category": result["predicted_label"],
            "confidence": result["confidence"],
        },
    )
    del records[100:]


def go_to(page):
    st.session_state["navigation_page"] = page


def log_out():
    st.session_state["auth_user"] = None
    st.session_state["navigation_page"] = "Home"


def load_selected_sample():
    selected = st.session_state.get("selected_sample")
    if selected in SAMPLE_PROMPTS:
        st.session_state["text_input"] = SAMPLE_PROMPTS[selected]


def load_comparison_sample():
    selected = st.session_state.get("comparison_sample")
    if selected in SAMPLE_PROMPTS:
        st.session_state["comparison_text"] = SAMPLE_PROMPTS[selected]


def show_hero(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _register_report_font():
    if "TenglishReport" in pdfmetrics.getRegisteredFontNames():
        return "TenglishReport"
    font_candidates = [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "Nirmala.ttc"),
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf"),
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("TenglishReport", font_path, subfontIndex=0))
            return "TenglishReport"
    return "Helvetica"


def _pdf_paragraph(value, style):
    safe_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value))
    safe_text = safe_text.replace("\r\n", "\n").replace("\r", "\n")
    return Paragraph(escape(safe_text).replace("\n", "<br/>"), style)


def build_pdf_report(title, records, summary, model_info):
    from xml.sax.saxutils import escape

    output = BytesIO()
    font_name = _register_report_font()
    page_size = landscape(A4) if len(records) > 1 else A4
    document = SimpleDocTemplate(
        output,
        pagesize=page_size,
        rightMargin=36,
        leftMargin=36,
        topMargin=42,
        bottomMargin=42,
        title=title,
        author="Tenglish Insights",
    )
    base_styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base_styles["Title"],
        fontName=font_name,
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#312e81"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=base_styles["Heading2"],
        fontName=font_name,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#3730a3"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=base_styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=5,
    )
    cell_style = ParagraphStyle(
        "ReportCell",
        parent=body_style,
        fontSize=8,
        leading=10,
        wordWrap="CJK",
    )
    story = [
        Paragraph(escape(title), title_style),
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            "Code-mixed Telugu-English text analysis",
            body_style,
        ),
        Paragraph("Analysis summary", heading_style),
    ]
    summary_rows = [
        [_pdf_paragraph(key, cell_style), _pdf_paragraph(value, cell_style)]
        for key, value in summary.items()
    ]
    summary_table = Table(summary_rows, colWidths=[150, document.width - 150])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2ff")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1f2937")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([summary_table, Paragraph("Model information", heading_style)])
    for item in model_info:
        story.append(_pdf_paragraph(f"• {item}", body_style))

    story.append(Paragraph("Predictions", heading_style))
    include_scores = any(record.get("probabilities") for record in records)
    if include_scores:
        headers = ["#", "Prediction", "Confidence", "Class scores", "Input text"]
        widths = [
            26,
            72,
            60,
            118,
            document.width - 276,
        ]
    else:
        headers = ["#", "Prediction", "Confidence", "Input text"]
        widths = [30, 95, 78, document.width - 203]
    table_data = [[_pdf_paragraph(header, cell_style) for header in headers]]
    for index, record in enumerate(records, start=1):
        prediction = str(record.get("predicted_label", "unknown")).title()
        confidence = record.get("confidence", 0.0)
        input_text = record.get("input_text", record.get("text", ""))
        row = [
            _pdf_paragraph(index, cell_style),
            _pdf_paragraph(prediction, cell_style),
            _pdf_paragraph(f"{float(confidence):.1%}", cell_style),
        ]
        if include_scores:
            probabilities = record.get("probabilities", {})
            score_text = " · ".join(
                f"{label.title()}: {float(score):.1%}"
                for label, score in probabilities.items()
            )
            row.append(_pdf_paragraph(score_text, cell_style))
        row.append(_pdf_paragraph(input_text, cell_style))
        table_data.append(row)

    prediction_table = LongTable(
        table_data,
        colWidths=widths,
        repeatRows=1,
        splitByRow=1,
    )
    prediction_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312e81")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1f2937")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(prediction_table)
    story.extend(
        [
            Spacer(1, 12),
            Paragraph("Interpretation and responsible use", heading_style),
            _pdf_paragraph(
                "Predictions and confidence values are outputs of a text classifier, not a "
                "clinical diagnosis or proof of a person's intent. This project uses a small, "
                "template-based dataset; scores may not generalize to real-world messages. "
                "Review results in context and do not use them alone for consequential decisions.",
                body_style,
            ),
        ]
    )

    def draw_page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(36, 30, page_size[0] - 36, 30)
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(36, 18, "Tenglish Insights · Automated analysis report")
        canvas.drawRightString(page_size[0] - 36, 18, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return output.getvalue()


def show_result(result):
    label = result["predicted_label"]
    probabilities = result["probabilities"]
    second_label, second_probability = sorted(
        probabilities.items(), key=lambda item: item[1], reverse=True
    )[1]
    st.markdown("### Analysis result")
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Detected category", f"{LABEL_ICONS.get(label, '🔎')} {label.title()}"
    )
    metric_columns[1].metric("Top model score", f"{result['confidence']:.1%}")
    metric_columns[2].metric("Words analyzed", len(result["cleaned_text"].split()))

    descriptions = {
        "normal": "The strongest match is with routine or neutral conversation patterns.",
        "depression": (
            "The text resembles distress-related patterns in the training data. "
            "This is not a clinical assessment."
        ),
        "toxic": (
            "The text resembles hostile or abusive language patterns in the training data. "
            "Context still matters."
        ),
    }
    st.info(descriptions.get(label, "The model selected the class with the highest score."))

    chart, details = st.columns([1.2, 1])
    with chart:
        st.markdown("**Class scores**")
        chart_df = pd.DataFrame(
            {
                "Category": [name.title() for name in probabilities],
                "Score": list(probabilities.values()),
            }
        ).set_index("Category")
        st.bar_chart(chart_df, color="#6366f1")
        st.caption(
            f"Next closest: {second_label.title()} ({second_probability:.1%}). "
            "Scores are model outputs—not the probability someone has a condition "
            "or intended harm."
        )
    with details:
        st.markdown("**Text preparation**")
        st.code(result["cleaned_text"] or "(No text remained after normalization)")
        st.caption(
            "Text is lowercased, whitespace is normalized, and selected Tenglish spelling "
            "variants are standardized before feature extraction."
        )

    if result.get("crisis_intervention_required"):
        st.warning(
            "The model crossed its distress-support threshold. This flag is not a diagnosis. "
            "If someone may be in immediate danger, contact local emergency services or a "
            "trusted support professional."
        )
        st.markdown(
            "**India support resources:** Tele-MANAS **14416** / **1800-891-4416** · "
            "KIRAN **1800-599-0019** · Vandrevala Foundation **+91 9999 666 555**"
        )
    elif result.get("toxic_threat_flag"):
        st.warning(
            "High toxicity score: consider reviewing the message in context and following "
            "your community's reporting or moderation process."
        )
    report_bytes = build_pdf_report(
        "Text Analysis Report",
        [result],
        {
            "Messages analyzed": "1",
            "Predicted category": label.title(),
            "Top model score": f"{result['confidence']:.1%}",
            "Words after normalization": str(len(result["cleaned_text"].split())),
            "Characters in input": str(len(result["input_text"])),
            "Distress support flag": "Yes" if result.get("crisis_intervention_required") else "No",
            "Toxicity review flag": "Yes" if result.get("toxic_threat_flag") else "No",
        },
        [
            "Ensemble: calibrated LinearSVC and Logistic Regression.",
            f"Saved SVM blend weight: {detector.svm_weight:.0%}; Logistic Regression weight: "
            f"{1 - detector.svm_weight:.0%}.",
            "Features: word TF-IDF n-grams (1–2) plus character TF-IDF n-grams (2–5).",
            "Preprocessing: lowercase, whitespace normalization, and selected Tenglish spelling normalization.",
        ],
    )
    st.download_button(
        "Download PDF report",
        data=report_bytes,
        file_name="tenglish_text_analysis_report.pdf",
        mime="application/pdf",
        key="single_analysis_pdf",
    )


@st.cache_data(show_spinner="Reconstructing held-out test evaluation...")
def evaluate_saved_models(dataset_path, svm_path, lr_path):
    dataset = pd.read_csv(dataset_path)
    dataset["text"] = dataset["text"].fillna("").astype(str).str.strip()
    dataset = dataset[dataset["text"] != ""].copy()
    label_names = sorted(dataset["label_name"].unique())
    label_to_id = {name: index for index, name in enumerate(label_names)}
    labels = dataset["label_name"].map(label_to_id)
    _, temp_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )
    _, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=42,
        stratify=labels.iloc[temp_indices],
    )
    test_texts = dataset.iloc[test_indices]["text"]
    y_test = labels.iloc[test_indices]
    svm_model = joblib.load(svm_path)
    lr_model = joblib.load(lr_path)
    svm_scores = svm_model.predict_proba(test_texts)
    lr_scores = lr_model.predict_proba(test_texts)
    predictions = {
        "Calibrated LinearSVC": svm_scores.argmax(axis=1),
        "Logistic Regression": lr_scores.argmax(axis=1),
    }
    return {
        "labels": label_names,
        "y_test": y_test.to_numpy(),
        "models": {
            name: {
                "accuracy": accuracy_score(y_test, predicted),
                "macro_f1": f1_score(y_test, predicted, average="macro"),
                "report": classification_report(
                    y_test,
                    predicted,
                    labels=list(range(len(label_names))),
                    target_names=label_names,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion": confusion_matrix(
                    y_test, predicted, labels=list(range(len(label_names)))
                ),
            }
            for name, predicted in predictions.items()
        },
    }


def explain_features(detector, text, predicted_label):
    pipeline = detector.lr_pipeline
    feature_union = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["lr"]
    transformed = feature_union.transform([text])
    feature_names = feature_union.get_feature_names_out()
    class_index = list(classifier.classes_).index(
        detector.label2id[predicted_label]
    )
    contributions = transformed.multiply(classifier.coef_[class_index]).toarray()[0]
    top_indices = np.argsort(contributions)[::-1]
    return [
        (str(feature_names[index]), float(contributions[index]))
        for index in top_indices
        if contributions[index] > 0
    ][:8]


def compare_model_predictions(detector, text):
    cleaned = clean_text(text)
    if not cleaned:
        return cleaned, {}

    models = {
        "Calibrated LinearSVC": detector.svm_pipeline,
        "Logistic Regression": detector.lr_pipeline,
    }
    results = {}
    for model_name, pipeline in models.items():
        class_ids = pipeline.classes_
        class_scores = pipeline.predict_proba([cleaned])[0]
        probabilities = {
            detector.id2label[int(class_id)]: float(score)
            for class_id, score in zip(class_ids, class_scores)
        }
        predicted_label = max(probabilities, key=probabilities.get)
        results[model_name] = {
            "predicted_label": predicted_label,
            "confidence": probabilities[predicted_label],
            "probabilities": probabilities,
        }
    return cleaned, results


initialize_auth_db(AUTH_DB_PATH)
st.session_state.setdefault("auth_user", None)
if not st.session_state["auth_user"]:
    render_auth_page(AUTH_DB_PATH)
    st.stop()


try:
    detector = get_detector()
    dataset = load_dataset(DATA_PATH)
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Could not load the application data or models: {error}")
    st.stop()

model_dir = os.path.join(BASE_DIR, "models")
st.session_state.setdefault("navigation_page", "Home")
st.session_state.setdefault("analysis_history", [])
st.session_state.setdefault(
    "text_input",
    "Naku em chesina point ledu anipisthundi, life meaning leni laga undi",
)
st.session_state.setdefault(
    "comparison_text",
    "Naku em chesina point ledu anipisthundi, life meaning leni laga undi",
)
st.session_state.setdefault("analysis_result", None)
st.session_state.setdefault("model_comparison", None)
st.session_state.setdefault("batch_results", None)

with st.sidebar:
    st.markdown("# 🛡️ Text Safety Insights")
    st.caption("Code-mixed text safety explorer")
    st.caption(f"Signed in as **{st.session_state['auth_user']}**")
    st.radio("Navigate", PAGES, key="navigation_page", label_visibility="collapsed")
    st.divider()
    st.markdown("**Project snapshot**")
    st.caption("1,500 labeled posts · 3 text categories · Word + character TF-IDF")
    st.caption("Prototype for educational exploration—not a clinical tool.")
    st.button("Log out", use_container_width=True, on_click=log_out)

page = st.session_state["navigation_page"]
history = st.session_state["analysis_history"]

if page == "Home":
    show_hero(
        "CODE-MIXED NLP · PROJECT EXPLORER",
        "Understand the signal behind the text.",
        "Explore a Telugu-English text classifier built to study distress-related language, "
        "online hostility, and everyday campus conversation.",
    )
    sample_count = len(dataset)
    category_count = dataset["label_name"].nunique()
    normal_share = float((dataset["label_name"] == "normal").mean())
    metrics = st.columns(4)
    metrics[0].metric("Labeled examples", f"{sample_count:,}")
    metrics[1].metric("Text categories", category_count)
    metrics[2].metric("Recent analyses", len(history))
    metrics[3].metric("Normal class share", f"{normal_share:.0%}")

    st.markdown("### Pick a place to start")
    feature_cols = st.columns(4)
    feature_cards = [
        ("🔎", "Text Analyzer", "Try a post and inspect its class scores."),
        ("📁", "Batch Analyzer", "Classify rows from a CSV or lines from a TXT file."),
        ("🧠", "AI Insights", "Explore model feature contributions and language cues."),
        ("📊", "Dashboard", "See dataset composition and your session's results."),
    ]
    for column, (icon, target, description) in zip(feature_cols, feature_cards):
        with column:
            st.markdown(f"#### {icon} {target}")
            st.write(description)
            st.button(
                f"Explore {target}",
                key=f"home_{target}",
                use_container_width=True,
                on_click=go_to,
                args=(target,),
            )

    st.markdown("### How this project works")
    flow = st.columns(4)
    steps = [
        ("01 · Normalize", "Standardize case, whitespace, and selected Tenglish variants."),
        ("02 · Represent", "Combine word n-grams with character fragments using TF-IDF."),
        ("03 · Classify", "Score posts with calibrated LinearSVC and Logistic Regression."),
        ("04 · Review", "Inspect class scores and apply human judgment to safety flags."),
    ]
    for column, (title, description) in zip(flow, steps):
        with column:
            st.markdown(f"**{title}**")
            st.caption(description)
elif page == "Text Analyzer":
    show_hero(
        "INTERACTIVE CLASSIFIER",
        "Text Analyzer",
        "Enter a short post in Telugu-English or English, or load a sample to explore the model.",
    )
    left, right = st.columns([1, 1.5])
    with left:
        st.markdown("#### Try a sample")
        st.selectbox(
            "Example posts",
            ["Choose an example..."] + list(SAMPLE_PROMPTS),
            key="selected_sample",
            on_change=load_selected_sample,
        )
        st.caption("Samples demonstrate patterns represented in the project dataset.")
    with right:
        with st.form("text_analysis_form"):
            text = st.text_area(
                "Post or comment",
                key="text_input",
                height=150,
                placeholder="Type a short code-mixed or English message...",
            )
            submitted = st.form_submit_button(
                "Analyze text →", type="primary", use_container_width=True
            )
    if submitted:
        if not text.strip():
            st.warning("Add some text before running the analysis.")
            st.session_state["analysis_result"] = None
        else:
            with st.spinner("Analyzing text..."):
                result = detector.predict(text.strip())
            st.session_state["analysis_result"] = result
            add_history(result)
    result = st.session_state.get("analysis_result")
    if result and result["input_text"] == st.session_state["text_input"].strip():
        show_result(result)

elif page == "Model Comparison":
    show_hero(
        "MODEL COMPARISON LABORATORY",
        "Compare how models read the same text",
        "Run one message through both trained classifiers and compare their predictions, "
        "confidence scores, and class-level probabilities.",
    )
    st.markdown(
        "Both models use the same normalized text and hybrid word/character TF-IDF features. "
        "Only the classifier differs: calibrated LinearSVC versus Logistic Regression."
    )
    sample_col, input_col = st.columns([1, 1.6])
    with sample_col:
        st.selectbox(
            "Load a project example",
            ["Choose an example..."] + list(SAMPLE_PROMPTS),
            key="comparison_sample",
            on_change=load_comparison_sample,
        )
    with input_col:
        with st.form("model_comparison_form"):
            comparison_text = st.text_area(
                "Text to compare",
                key="comparison_text",
                height=120,
                placeholder="Enter one message for both models...",
            )
            compare_submitted = st.form_submit_button(
                "Compare models →", type="primary", use_container_width=True
            )

    if compare_submitted:
        cleaned_text, comparison = compare_model_predictions(
            detector, comparison_text.strip()
        )
        if not cleaned_text:
            st.session_state["model_comparison"] = None
            st.warning("Enter or select some text before comparing models.")
        else:
            st.session_state["model_comparison"] = {
                "input_text": comparison_text.strip(),
                "cleaned_text": cleaned_text,
                "models": comparison,
            }

    comparison = st.session_state.get("model_comparison")
    if comparison and comparison["input_text"] == st.session_state["comparison_text"].strip():
        model_results = comparison["models"]
        model_columns = st.columns(len(model_results))
        for column, (model_name, result) in zip(model_columns, model_results.items()):
            with column:
                st.markdown(f"#### {model_name}")
                st.metric(
                    "Predicted category",
                    f"{LABEL_ICONS.get(result['predicted_label'], '🔎')} "
                    f"{result['predicted_label'].title()}",
                )
                st.metric("Top model score", f"{result['confidence']:.1%}")

        model_labels = [result["predicted_label"] for result in model_results.values()]
        if len(set(model_labels)) == 1:
            st.success(f"Both models agree on **{model_labels[0].title()}** for this input.")
        else:
            st.warning(
                "The models disagree. Review the message and class scores in context; "
                "neither prediction should be treated as a definitive judgment."
            )

        comparison_chart = pd.DataFrame(
            {
                model_name: result["probabilities"]
                for model_name, result in model_results.items()
            }
        ).rename(index=str.title)
        st.markdown("#### Probability comparison by category")
        st.bar_chart(comparison_chart)
        st.dataframe(
            comparison_chart.style.format("{:.1%}"),
            use_container_width=True,
        )

        st.markdown("#### What the comparison tells you")
        st.markdown(
            f"""
            - **Calibrated LinearSVC** separates examples using learned class boundaries;
              calibration maps its decision scores into class scores.
            - **Logistic Regression** scores each class from weighted text features and
              produces its own probability distribution.
            - The score difference for a class shows how differently these two classifiers
              rank this particular text. Similar scores do not prove the models are correct.
            - The Text Analyzer uses the saved blend: **{detector.svm_weight:.0%} LinearSVC**
              and **{1 - detector.svm_weight:.0%} Logistic Regression**.
            """
        )
        st.markdown("**Normalized text seen by both models**")
        st.code(comparison["cleaned_text"])
        st.caption(
            "This is a research comparison on a small, template-based dataset. Model scores "
            "are not diagnoses or proof of a person's intent."
        )
    else:
        st.info("Enter a message and select **Compare models** to see a side-by-side result.")

elif page == "Batch Analyzer":
    show_hero(
        "BULK CLASSIFICATION",
        "Batch Analyzer",
        "Analyze a CSV with a `text` column or a TXT file with one message per line.",
    )
    st.markdown(
        '<div class="section-card">Files are processed in this session. '
        'Use de-identified examples and avoid uploading private or sensitive personal data.</div>',
        unsafe_allow_html=True,
    )
    with st.form("batch_analysis_form"):
        uploaded = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"])
        analyze_batch = st.form_submit_button("Analyze file →", type="primary")
    if analyze_batch:
        if uploaded is None:
            st.warning("Choose a CSV or TXT file first.")
        else:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    batch_df = pd.read_csv(uploaded)
                    if "text" not in batch_df.columns:
                        st.error("CSV needs a column named `text`.")
                        batch_df = None
                else:
                    lines = uploaded.getvalue().decode("utf-8-sig").splitlines()
                    batch_df = pd.DataFrame({"text": [line.strip() for line in lines if line.strip()]})
                if batch_df is not None:
                    if batch_df.empty:
                        st.warning("The uploaded file has no text rows to analyze.")
                    else:
                        with st.spinner(f"Analyzing {len(batch_df):,} messages..."):
                            output_rows = [
                                detector.predict(str(value) if pd.notna(value) else "")
                                for value in batch_df["text"]
                            ]
                        batch_df["predicted_label"] = [
                            result["predicted_label"] for result in output_rows
                        ]
                        batch_df["confidence"] = [
                            result["confidence"] for result in output_rows
                        ]
                        st.session_state["batch_results"] = batch_df
                        st.session_state["batch_report_records"] = [
                            {
                                **result,
                                "input_text": str(batch_df.iloc[index]["text"]),
                            }
                            for index, result in enumerate(output_rows)
                        ]
                        for item in output_rows[-100:]:
                            add_history(item, source="Batch Analyzer")
            except (UnicodeDecodeError, pd.errors.ParserError, OSError) as error:
                st.error(f"Could not read the uploaded file: {error}")
    batch_results = st.session_state.get("batch_results")
    if batch_results is not None:
        st.success(f"Processed {len(batch_results):,} rows.")
        st.dataframe(batch_results.head(100), hide_index=True, use_container_width=True)
        st.download_button(
            "Download results as CSV",
            batch_results.to_csv(index=False).encode("utf-8"),
            file_name="tenglish_analysis_results.csv",
            mime="text/csv",
        )
        report_records = st.session_state.get("batch_report_records", [])
        if report_records:
            category_counts = batch_results["predicted_label"].value_counts()
            batch_summary = {
                "Messages analyzed": f"{len(batch_results):,}",
                "Average top model score": f"{batch_results['confidence'].mean():.1%}",
                "Median top model score": f"{batch_results['confidence'].median():.1%}",
                "Highest top model score": f"{batch_results['confidence'].max():.1%}",
                "Lowest top model score": f"{batch_results['confidence'].min():.1%}",
                "Category distribution": ", ".join(
                    f"{label.title()}: {int(count)}"
                    for label, count in category_counts.items()
                ),
            }
            batch_report = build_pdf_report(
                "Batch Text Analysis Report",
                report_records,
                batch_summary,
                [
                    "Ensemble: calibrated LinearSVC and Logistic Regression.",
                    f"Saved SVM blend weight: {detector.svm_weight:.0%}; Logistic Regression weight: "
                    f"{1 - detector.svm_weight:.0%}.",
                    "Features: word TF-IDF n-grams (1–2) plus character TF-IDF n-grams (2–5).",
                    "Each row includes its predicted category, top score, and class scores.",
                ],
            )
            st.download_button(
                "Download batch PDF report",
                data=batch_report,
                file_name="tenglish_batch_analysis_report.pdf",
                mime="application/pdf",
                key="batch_analysis_pdf",
            )

elif page == "AI Insights":
    show_hero(
        "TRANSPARENT MODEL SIGNALS",
        "AI Insights",
        "See which learned text features supported a prediction, alongside a simple cue scan.",
    )
    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Run an analysis in Text Analyzer first to unlock per-post insights.")
        st.button(
            "Open Text Analyzer",
            on_click=go_to,
            args=("Text Analyzer",),
        )
    else:
        st.markdown(f"#### Current result: {LABEL_ICONS[result['predicted_label']]} {result['predicted_label'].title()}")
        try:
            contributions = explain_features(
                detector, result["cleaned_text"], result["predicted_label"]
            )
            if contributions:
                st.markdown("**Learned features supporting the Logistic Regression class score**")
                contribution_df = pd.DataFrame(
                    contributions, columns=["Feature", "Contribution"]
                ).set_index("Feature")
                st.bar_chart(contribution_df, color="#8b5cf6")
                st.caption(
                    "These are positive TF-IDF feature contributions from Logistic Regression, "
                    "not causal explanations. SVM ensemble weight in the saved config is "
                    f"{detector.svm_weight:.0%}."
                )
            else:
                st.info("No positive feature contributions were available for this input.")
        except (AttributeError, KeyError, ValueError, IndexError) as error:
            st.warning(f"Feature contribution view is unavailable: {error}")

        normalized_tokens = set(result["cleaned_text"].split())
        cue_rows = [
            {"Cue group": group, "Matched terms": ", ".join(sorted(cues & normalized_tokens))}
            for group, terms in LEXICAL_CUES.items()
            if (cues := terms & normalized_tokens)
        ]
        st.markdown("**Simple language cue scan**")
        if cue_rows:
            st.dataframe(pd.DataFrame(cue_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No terms from the small illustrative cue list matched this text.")
        st.caption(
            "The cue scan is a transparent keyword lookup, not a trained emotion classifier. "
            "This project does not currently contain a separate emotion model."
        )

elif page == "Dashboard":
    show_hero(
        "DATA & SESSION OVERVIEW",
        "Dashboard",
        "Explore the labeled dataset and the analyses performed during this browser session.",
    )
    dataset_counts = dataset["label_name"].value_counts().rename_axis("Category").reset_index(name="Posts")
    history_df = pd.DataFrame(history)
    total_cols = st.columns(4)
    total_cols[0].metric("Dataset examples", f"{len(dataset):,}")
    total_cols[1].metric("Categories", dataset["label_name"].nunique())
    total_cols[2].metric("This session", len(history))
    total_cols[3].metric(
        "Mean model score",
        f"{history_df['confidence'].mean():.1%}" if not history_df.empty else "—",
    )
    chart_col, latest_col = st.columns([1, 1])
    with chart_col:
        st.markdown("#### Training dataset composition")
        st.bar_chart(dataset_counts.set_index("Category"), color="#6366f1")
    with latest_col:
        st.markdown("#### Session predictions")
        if history_df.empty:
            st.caption("Run text or batch analysis to populate this chart.")
        else:
            session_counts = history_df["category"].value_counts().rename_axis("Category").to_frame("Posts")
            st.bar_chart(session_counts, color="#0ea5e9")
    st.markdown("#### Dataset profile")
    st.dataframe(
        dataset_counts.assign(Share=dataset_counts["Posts"] / len(dataset)),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Session charts and history are held in Streamlit session state, not a permanent database.")

elif page == "Model Performance":
    show_hero(
        "EVALUATION & LIMITATIONS",
        "Model Performance",
        "Compare both classifiers, inspect per-class results, and understand what the scores mean.",
    )
    st.markdown(
        "The evaluation below uses a stratified holdout: **70% for training, 15% for validation, "
        "and 15% for testing**. Stratification preserves the class proportions in each split. "
        "The test data is held aside while the classifiers are trained; validation data is used "
        "to select the ensemble blend."
    )
    try:
        evaluation = evaluate_saved_models(
            DATA_PATH,
            os.path.join(model_dir, "svm_model.joblib"),
            os.path.join(model_dir, "lr_model.joblib"),
        )
        comparison_rows = [
            {
                "Model": name,
                "Accuracy": scores["accuracy"],
                "Macro F1": scores["macro_f1"],
            }
            for name, scores in evaluation["models"].items()
        ]
        comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
        st.markdown("#### At a glance")
        metric_cols = st.columns(len(comparison_rows))
        for column, row in zip(metric_cols, comparison_rows):
            column.metric(f"{row['Model']} · accuracy", f"{row['Accuracy']:.1%}")
            column.metric("Macro F1", f"{row['Macro F1']:.1%}")

        test_size = len(evaluation["y_test"])
        st.caption(
            f"Test set: **{test_size:,} posts** "
            f"({test_size // len(evaluation['labels']):,} per category in this balanced split)."
        )
        with st.expander("What happened during training and evaluation"):
            train_size = len(dataset) - int(len(dataset) * 0.30)
            validation_size = int(len(dataset) * 0.30) - test_size
            st.markdown(
                f"""
                1. **Training ({train_size:,} posts):** The vectorizers learn which word and
                   character patterns are useful, and the classifiers learn how those patterns
                   relate to the three labels.
                2. **Validation ({validation_size:,} posts):** The SVM/Logistic Regression blend
                   is selected using macro F1. The saved SVM blend weight is
                   **{detector.svm_weight:.0%}**; the remaining weight is assigned to Logistic
                   Regression.
                3. **Testing ({test_size:,} posts):** Both saved models are scored on the same
                   held-out posts. This page compares each model individually; the interactive
                   analyzer uses the saved blended probabilities.
                """
            )
            st.markdown(
                """
                **Classifier overview:** Calibrated LinearSVC learns separating boundaries in
                the sparse TF-IDF feature space; its calibration layer converts decision scores
                into class probabilities. Logistic Regression estimates class probabilities
                directly from weighted feature combinations. Both models use class-balanced
                training and the same word-plus-character feature representation.
                """
            )

        st.markdown("#### Model comparison")
        st.bar_chart(comparison_df, y=["Accuracy", "Macro F1"])
        with st.expander("How to read these metrics"):
            st.markdown(
                """
                - **Accuracy** = correct predictions / all test posts. It answers: “How many
                  posts were classified correctly overall?”
                - **Precision for a category** = correct predictions of that category / all
                  predictions made for that category. Higher precision means fewer false alarms
                  for that label.
                - **Recall for a category** = correct predictions of that category / all actual
                  test examples in that category. Higher recall means fewer missed examples.
                - **F1** is the harmonic mean of precision and recall; it falls when either one
                  is low.
                - **Macro F1** averages F1 across categories equally. Here the dataset is
                  balanced, but macro F1 still makes category-level performance visible.

                For example, if 70 of 75 actual depression examples were found, recall would be
                70/75 = 93.3%. If the model also incorrectly labeled normal posts as depression,
                those false positives would reduce depression precision.
                """
            )

        st.markdown("#### Per-category performance")
        st.write(
            "Select a model to see **precision** (how often predictions for a category were "
            "correct), **recall** (how many examples of that category were found), and **F1** "
            "(the balance of precision and recall). Support is the number of test examples "
            "belonging to that category."
        )

        selected_model = st.selectbox("Per-class report", list(evaluation["models"]))
        scores = evaluation["models"][selected_model]
        report_df = pd.DataFrame(scores["report"]).T
        st.dataframe(report_df.round(3), use_container_width=True)
        st.caption(
            "A high precision with lower recall means the model is cautious about assigning that "
            "label but misses some relevant examples. A high recall with lower precision means it "
            "finds more examples but also raises more false positives."
        )

        st.markdown("#### Confusion matrix")
        st.write(
            "Rows are the **actual** dataset labels and columns are the **predicted** labels. "
            "The diagonal contains correct predictions; off-diagonal cells show which categories "
            "were confused with one another."
        )
        matrix_df = pd.DataFrame(
            scores["confusion"],
            index=[f"Actual {name}" for name in evaluation["labels"]],
            columns=[f"Predicted {name}" for name in evaluation["labels"]],
        )
        st.dataframe(matrix_df, use_container_width=True)
        total_correct = int(np.trace(scores["confusion"]))
        st.caption(
            f"For **{selected_model}**, the diagonal contains {total_correct:,} correct "
            f"classifications out of {test_size:,}. Any non-zero off-diagonal cell identifies "
            "a specific pair of categories the model confused."
        )

        st.markdown("#### What does a perfect score mean here?")
        st.markdown(
            f"""
            On this split, **{selected_model}** correctly classified all
            **{total_correct:,} of {test_size:,}** test posts. This describes performance on
            this test sample only. The dataset has 1,500 balanced, template-based examples;
            related writing patterns can appear in both the training and test partitions even
            when exact rows differ. A model can therefore learn recurring templates or wording
            shortcuts and still score perfectly on a random split.

            This score does **not** establish that the model understands mental health, detects
            intent, or will achieve the same results on real messages, other regions, dialects,
            sarcasm, or new spelling styles. In real use, missed distress (false negatives) and
            incorrectly flagged ordinary posts (false positives) both matter.
            """
        )
        with st.expander("How to make a stronger evaluation"):
            st.markdown(
                """
                - Collect a larger, consented, de-identified dataset with clear labeling guidance.
                - Keep posts from the same source, author, or generated template in only one split.
                - Test separately on a later time period and on communities or dialects not used
                  during training.
                - Review errors with trained human annotators; report per-class precision,
                  recall, F1, and uncertainty—not just one accuracy number.
                - Treat the model as decision support for research, not a diagnosis or automatic
                  action against a person.
                """
            )
        st.caption(
            "Evaluation reconstructs the training script's fixed random_state=42 split. "
            "Use independently collected, consented, de-identified examples and a separate "
            "out-of-domain test set before considering any real-world application."
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        st.error(f"Could not evaluate saved models: {error}")

elif page == "History":
    show_hero(
        "YOUR CURRENT SESSION",
        "Analysis History",
        "Review recent analyses from this browser session. Entries are not saved permanently.",
    )
    if not history:
        st.info("Your history is empty. Analyze a message to add the first entry.")
    else:
        history_df = pd.DataFrame(history)
        st.dataframe(
            history_df.assign(
                category=history_df["category"].map(
                    lambda label: f"{LABEL_ICONS.get(label, '')} {label.title()}"
                ),
                confidence=history_df["confidence"].map(lambda score: f"{score:.1%}"),
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download session history",
            history_df.to_csv(index=False).encode("utf-8"),
            file_name="tenglish_session_history.csv",
            mime="text/csv",
        )
        if st.button("Clear session history"):
            st.session_state["analysis_history"] = []
            st.rerun()

elif page == "About":
    show_hero(
        "PROJECT NOTES",
        "About this project",
        "A classroom-scale NLP prototype for exploring text classification in Romanized Telugu-English.",
    )
    st.markdown("### Project goals")
    st.write(
        "Study whether hybrid word and character features can separate everyday campus "
        "conversation, distress-related language patterns, and toxic or harassing language "
        "in short code-mixed posts."
    )
    about_cols = st.columns(3)
    about_cols[0].metric("Labeled posts", f"{len(dataset):,}")
    about_cols[1].metric("Language style", "Telugu + English")
    about_cols[2].metric("Classes", dataset["label_name"].nunique())
    st.markdown("### Technology")
    st.markdown(
        "- **Python** for application and model logic\n"
        "- **Streamlit** for the interactive dashboard\n"
        "- **pandas / NumPy** for dataset handling\n"
        "- **scikit-learn** for TF-IDF, calibration, classification, and evaluation\n"
        "- **joblib** for saved model artifacts"
    )
    st.markdown("### Team")
    st.info(
        "Team member names and affiliations were not included in the project files. "
        "Add verified contributor details here before presenting the project publicly."
    )
    st.markdown("### Responsible-use notes")
    st.write(
        "This prototype is not a diagnostic, emergency-response, or moderation authority. "
        "Text classifiers can miss context, dialect, sarcasm, and spelling variation. "
        "Do not upload identifying or sensitive messages; a human should review any consequential use."
    )
