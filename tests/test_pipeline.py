"""
Pipeline Verification and Integration Test Suite
Validates configuration integrity, disease database structure,
image preprocessing tensors, prediction logic, and Flask web routes.
"""

import json
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from app import app
from predict import get_predictor


class TestPlantDiseasePipeline(unittest.TestCase):
    """Integration and Unit Test Cases."""

    def setUp(self):
        self.client = app.test_client()
        self.predictor = get_predictor()

    def test_01_configuration_paths(self):
        """Verify essential directories and paths are created and valid."""
        self.assertTrue(config.DATASET_DIR.exists())
        self.assertTrue(config.MODELS_DIR.exists())
        self.assertTrue(config.RESULTS_DIR.exists())
        self.assertTrue(config.UPLOADS_DIR.exists())
        self.assertTrue(config.DISEASE_INFO_DIR.exists())
        self.assertEqual(config.IMAGE_SIZE, (224, 224))

    def test_02_disease_database_coverage(self):
        """Verify disease_details.json structure and required attributes."""
        self.assertTrue(config.DISEASE_INFO_PATH.exists())
        with open(config.DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
            disease_db = json.load(f)

        self.assertGreaterEqual(len(disease_db), 35, "Disease database should cover PlantVillage classes")
        sample_entry = disease_db.get("Tomato___Early_blight")
        self.assertIsNotNone(sample_entry)
        self.assertEqual(sample_entry["crop_name"], "Tomato")
        self.assertIn("recommendations", sample_entry)
        self.assertIn("cultural", sample_entry["recommendations"])
        self.assertIn("organic", sample_entry["recommendations"])

    def test_03_preprocessing_tensor_shape(self):
        """Verify image preprocessing produces expected MobileNetV2 dimensions (1, 224, 224, 3)."""
        sample_img = config.SAMPLES_DIR / "Tomato_Healthy.jpg"
        if sample_img.exists():
            batch = self.predictor.preprocess_image(sample_img)
            self.assertEqual(batch.shape, (1, 224, 224, 3))
            self.assertEqual(batch.dtype, "float32")

    def test_04_flask_home_route(self):
        """Verify Home page returns HTTP 200 and loads title."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"FloraGuard", response.data)
        self.assertIn(b"Plant Disease Detection", response.data)

    def test_05_flask_detect_route(self):
        """Verify Detection upload page returns HTTP 200."""
        response = self.client.get("/detect")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload Leaf Image For Analysis", response.data)

    def test_06_flask_about_route(self):
        """Verify About page returns HTTP 200 and details MobileNetV2."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"MobileNetV2", response.data)
        self.assertIn(b"PlantVillage", response.data)

    def test_07_api_predict_validation(self):
        """Verify API returns 400 when no file is submitted."""
        response = self.client.post("/api/predict")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])


if __name__ == "__main__":
    unittest.main()
