"""
Model Training and Calibration Pipeline
Trains Calibrated LinearSVC, Logistic Regression, and Ensemble for Code-Mixed Mental Health & Toxic Detection.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from src.features import build_hybrid_tfidf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "warangal_mental_health.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_models():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Clean text column and map labels
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    
    label_names = sorted(df["label_name"].unique())
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    df["label"] = df["label_name"].map(label2id)
    
    print(f"Dataset summary: Total samples = {len(df)}")
    print("Class distribution:\n", df["label_name"].value_counts())
    
    # 70% Train, 15% Validation, 15% Test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val = val_df["text"], val_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]
    
    # 1. Calibrated Linear Support Vector Machine
    print("\n--- Training Calibrated LinearSVC ---")
    svm_pipeline = Pipeline([
        ("tfidf", build_hybrid_tfidf()),
        ("svm", CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=42, C=1.0),
            cv=3
        ))
    ])
    svm_pipeline.fit(X_train, y_train)
    val_preds_svm = svm_pipeline.predict(X_val)
    print("SVM Val Accuracy:", accuracy_score(y_val, val_preds_svm))
    print("SVM Val Macro F1:", f1_score(y_val, val_preds_svm, average="macro"))
    
    # 2. Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr_pipeline = Pipeline([
        ("tfidf", build_hybrid_tfidf()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=5.0,
            random_state=42
        ))
    ])
    lr_pipeline.fit(X_train, y_train)
    val_preds_lr = lr_pipeline.predict(X_val)
    print("LR Val Accuracy:", accuracy_score(y_val, val_preds_lr))
    print("LR Val Macro F1:", f1_score(y_val, val_preds_lr, average="macro"))
    
    # 3. Optimize Ensemble Weight on Validation Set
    svm_val_probs = svm_pipeline.predict_proba(X_val)
    lr_val_probs = lr_pipeline.predict_proba(X_val)
    
    best_weight = 0.5
    best_f1 = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * svm_val_probs + (1 - w) * lr_val_probs
        preds = blend.argmax(axis=1)
        score = f1_score(y_val, preds, average="macro")
        if score > best_f1:
            best_f1 = score
            best_weight = w
            
    print(f"\nOptimal SVM Ensemble Weight: {best_weight:.2f} (Val Macro F1 = {best_f1:.4f})")
    
    # 4. Final Evaluation on Independent Test Set
    print("\n================== TEST SET EVALUATION ==================")
    svm_test_probs = svm_pipeline.predict_proba(X_test)
    lr_test_probs = lr_pipeline.predict_proba(X_test)
    final_probs = best_weight * svm_test_probs + (1 - best_weight) * lr_test_probs
    final_preds = final_probs.argmax(axis=1)
    
    test_acc = accuracy_score(y_test, final_preds)
    test_macro_f1 = f1_score(y_test, final_preds, average="macro")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f}\n")
    print(classification_report(y_test, final_preds, target_names=[id2label[i] for i in range(len(id2label))]))
    
    # 5. Save Artifacts
    joblib.dump(svm_pipeline, os.path.join(MODEL_DIR, "svm_model.joblib"))
    joblib.dump(lr_pipeline, os.path.join(MODEL_DIR, "lr_model.joblib"))
    
    config = {
        "svm_weight": float(best_weight),
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "metrics": {
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_macro_f1)
        }
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    print(f"Models and configuration successfully exported to: {MODEL_DIR}")

if __name__ == "__main__":
    train_models()
