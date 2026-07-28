# src/hf_text_risk.py
"""
Hugging Face Text Risk Engine — AegisBank
Uses Hugging Face Transformers zero-shot classification & sentiment pipeline
to analyze borrower descriptions, loan application notes, and financial explanations.
"""

import logging
from typing import Dict, Any, List, Optional

log = logging.getLogger(__name__)

# Candidate risk labels for zero-shot text classification
DEFAULT_CANDIDATE_LABELS = [
    "financial stability and repayment capacity",
    "financial distress or high debt burden",
    "business expansion or capital investment",
    "gambling, speculation, or emergency medical debt",
]


class HFTextRiskAnalyzer:
    """
    NLP Text Risk Analyzer leveraging Hugging Face Transformers.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.classifier = None
        self._initialized = False

    def _initialize_pipeline(self):
        """Lazy load Hugging Face pipeline when needed."""
        if self._initialized:
            return
        try:
            from transformers import pipeline
            log.info(f"Loading Hugging Face Zero-Shot Classification model ({self.model_name})...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
            )
            self._initialized = True
        except Exception as e:
            log.warning(f"Could not initialize Hugging Face Transformers pipeline: {e}")
            self.classifier = None
            self._initialized = True

    def analyze_text(self, text: str, candidate_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze qualitative borrower text for default risk indicators.
        Returns risk score (0.0 - 1.0), top intent/concern, and label confidence probabilities.
        """
        if not text or not text.strip():
            return {
                "text_provided": False,
                "text_risk_score": 0.0,
                "sentiment_label": "NEUTRAL",
                "risk_signal": "No qualitative text provided",
                "confidence_scores": {},
            }

        labels = candidate_labels or DEFAULT_CANDIDATE_LABELS

        # Attempt Hugging Face Transformers execution
        self._initialize_pipeline()

        if self.classifier is not None:
            try:
                res = self.classifier(text, candidate_labels=labels)
                label_scores = dict(zip(res["labels"], res["scores"]))

                distress_score = label_scores.get("financial distress or high debt burden", 0.0)
                speculation_score = label_scores.get("gambling, speculation, or emergency medical debt", 0.0)
                stability_score = label_scores.get("financial stability and repayment capacity", 0.0)

                # Calculate text risk score (0.0 to 1.0)
                risk_score = round(float(distress_score * 0.6 + speculation_score * 0.8 + (1.0 - stability_score) * 0.2), 3)
                risk_score = min(max(risk_score, 0.0), 1.0)

                top_label = res["labels"][0]
                top_score = round(float(res["scores"][0]), 3)

                return {
                    "text_provided": True,
                    "text_risk_score": risk_score,
                    "top_intent": top_label,
                    "top_confidence": top_score,
                    "risk_signal": f"Primary theme: '{top_label}' ({int(top_score * 100)}% confidence)",
                    "confidence_scores": {k: round(float(v), 3) for k, v in label_scores.items()},
                    "hf_engine": "transformers-zero-shot",
                }
            except Exception as e:
                log.warning(f"Error during HF Transformer inference: {e}")

        # Rule-based fallback if Hugging Face pipeline is not loaded / fails
        return self._heuristic_fallback(text)

    def _heuristic_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback heuristic risk analysis for qualitative text."""
        text_lower = text.lower()
        high_risk_keywords = ["gambling", "crypto", "bankruptcy", "debt relief", "emergency", "eviction", "payday"]
        low_risk_keywords = ["home improvement", "refinance", "business expansion", "salary", "investment", "education"]

        high_hits = [w for w in high_risk_keywords if w in text_lower]
        low_hits = [w for w in low_risk_keywords if w in text_lower]

        if high_hits:
            score = 0.75
            signal = f"Identified high risk text terms: {', '.join(high_hits)}"
        elif low_hits:
            score = 0.15
            signal = f"Identified positive text terms: {', '.join(low_hits)}"
        else:
            score = 0.35
            signal = "Neutral loan description"

        return {
            "text_provided": True,
            "text_risk_score": score,
            "top_intent": "keyword-heuristic-fallback",
            "top_confidence": 0.70,
            "risk_signal": signal,
            "confidence_scores": {},
            "hf_engine": "heuristic-fallback",
        }


# Global singleton instance
_analyzer_instance: Optional[HFTextRiskAnalyzer] = None

def get_text_risk_analyzer() -> HFTextRiskAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = HFTextRiskAnalyzer()
    return _analyzer_instance


def evaluate_borrower_text(text: str) -> Dict[str, Any]:
    """Helper function to evaluate borrower text easily."""
    analyzer = get_text_risk_analyzer()
    return analyzer.analyze_text(text)
