"""
Inference and Clinical/Safety Decision Engine for Mental Health and Toxic Text Detection
"""
import os
import json
import joblib
import numpy as np
from src.preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Emergency helpline information for crisis intervention
CRISIS_HELPLINES = {
    "tele_manas": {
        "name": "Tele-MANAS (National Mental Health Helpline India)",
        "contact": "14416 / 1800-891-4416",
        "availability": "24/7 Toll-free"
    },
    "kiran": {
        "name": "KIRAN Mental Health Rehabilitation",
        "contact": "1800-599-0019",
        "availability": "24/7 Toll-free"
    },
    "vandrevala": {
        "name": "Vandrevala Foundation for Mental Health",
        "contact": "+91 9999 666 555",
        "availability": "24/7 Free Helpline"
    }
}

class MentalHealthDetector:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.config_path = os.path.join(model_dir, "config.json")
        self.svm_path = os.path.join(model_dir, "svm_model.joblib")
        self.lr_path = os.path.join(model_dir, "lr_model.joblib")
        
        self._load_models()
        
    def _load_models(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found at {self.config_path}. Run training first.")
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.svm_weight = self.config.get("svm_weight", 0.5)
        self.label2id = self.config.get("label2id", {"depression": 0, "normal": 1, "toxic": 2})
        self.id2label = {int(k): v for k, v in self.config.get("id2label", {}).items()}
        
        self.svm_pipeline = joblib.load(self.svm_path)
        self.lr_pipeline = joblib.load(self.lr_path)
        
    def predict(self, text: str) -> dict:
        """
        Analyzes code-mixed text and outputs predicted class, calibrated probabilities,
        confidence level, and safety/crisis escalation triggers.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "input_text": text,
                "cleaned_text": "",
                "predicted_label": "normal",
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in self.label2id.keys()},
                "crisis_intervention_required": False
            }
            
        # Get calibrated probabilities from both models
        svm_probs = self.svm_pipeline.predict_proba([cleaned])[0]
        lr_probs = self.lr_pipeline.predict_proba([cleaned])[0]
        
        # Weighted ensemble blend
        blended_probs = self.svm_weight * svm_probs + (1.0 - self.svm_weight) * lr_probs
        pred_idx = int(np.argmax(blended_probs))
        pred_label = self.id2label[pred_idx]
        confidence = float(blended_probs[pred_idx])
        
        prob_dict = {
            self.id2label[i]: float(blended_probs[i]) for i in range(len(self.id2label))
        }
        
        # Check crisis intervention criteria for depression
        depression_prob = prob_dict.get("depression", 0.0)
        crisis_needed = (pred_label == "depression" and depression_prob >= 0.65) or (depression_prob >= 0.75)
        
        # Check toxicity escalation
        toxic_prob = prob_dict.get("toxic", 0.0)
        toxic_threat = (pred_label == "toxic" and toxic_prob >= 0.70)
        
        result = {
            "input_text": text,
            "cleaned_text": cleaned,
            "predicted_label": pred_label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "crisis_intervention_required": crisis_needed,
            "toxic_threat_flag": toxic_threat,
            "support_resources": CRISIS_HELPLINES if crisis_needed else None
        }
        
        return result

if __name__ == "__main__":
    detector = MentalHealthDetector()
    
    test_cases = [
        "Naku em chesina point ledu anipisthundi, life meaning leni laga undi",
        "honestly, enti ra pichi pukku, nee mind sarigga undadha, idiot like you should not post here",
        "tonight Subedari daggara new biryani place try chesam, super ga undi",
        "feeling very anxious before exam in KITS, hands vanukuthunnayi, cannot breathe"
    ]
    
    print("\n=== Model Inference Smoke Test ===")
    for text in test_cases:
        res = detector.predict(text)
        print(f"\nText: {res['input_text']}")
        print(f"Prediction: {res['predicted_label'].upper()} (Confidence: {res['confidence']:.2%})")
        print(f"Probabilities: {res['probabilities']}")
        if res['crisis_intervention_required']:
            print(">>> CRISIS INTERVENTION TRIGGERED <<<")
