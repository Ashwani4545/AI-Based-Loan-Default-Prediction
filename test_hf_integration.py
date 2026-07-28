# test_hf_integration.py
"""
Integration test suite for Hugging Face features in AegisBank system.
Tests:
  1. Hugging Face Hub availability and mock/offline fallback
  2. Hugging Face Text Risk Engine
  3. Flask Webapp Hugging Face endpoints & text prediction integration
"""

import os
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.config import HF_MODEL_REPO, HF_DATASET_REPO
from utils.hf_hub import check_hf_hub_available, upload_model_to_hf, download_model_from_hf
from src.hf_text_risk import get_text_risk_analyzer, evaluate_borrower_text


class TestHuggingFaceIntegration(unittest.TestCase):

    def test_hf_hub_availability(self):
        available = check_hf_hub_available()
        self.assertIsInstance(available, bool)

    def test_text_risk_engine_empty(self):
        res = evaluate_borrower_text("")
        self.assertFalse(res["text_provided"])
        self.assertEqual(res["text_risk_score"], 0.0)

    def test_text_risk_engine_positive_text(self):
        res = evaluate_borrower_text("Borrower requires funding for business expansion and capital investment into new equipment.")
        self.assertTrue(res["text_provided"])
        self.assertIn("text_risk_score", res)
        self.assertGreaterEqual(res["text_risk_score"], 0.0)
        self.assertLessEqual(res["text_risk_score"], 1.0)

    def test_text_risk_engine_distress_text(self):
        res = evaluate_borrower_text("Borrower facing severe financial distress, urgent payday loan needed to avoid eviction and gambling debts.")
        self.assertTrue(res["text_provided"])
        self.assertGreater(res["text_risk_score"], 0.5)

    def test_flask_app_hf_endpoints(self):
        from webapp.app import app
        client = app.test_client()

        # Test GET /api/v1/hf/status
        resp = client.get("/api/v1/hf/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("hf_hub_installed", data)
        self.assertIn("model_repo", data)

        # Test POST /api/v1/hf/text-risk
        resp = client.post("/api/v1/hf/text-risk", json={"text": "Loan for home improvement and kitchen renovation."})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("analysis", data)

    def test_hf_upload_without_token(self):
        # Should gracefully fail with error message if token is missing
        res = upload_model_to_hf(token=None)
        if not os.environ.get("HF_TOKEN"):
            self.assertEqual(res["status"], "error")
            self.assertIn("token", res["message"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
