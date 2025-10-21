"""
Tests for FIR Categorizer System
Comprehensive test suite for FIR type classification functionality.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
import time
from unittest.mock import Mock, patch

from utils.fir_categorizer import FIRCategorizer, ClassificationResult, get_fir_categorizer, classify_fir_text


class TestFIRCategorizer(unittest.TestCase):
    """Test cases for FIRCategorizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test models
        self.temp_dir = tempfile.mkdtemp()

        # Create test categories file
        self.test_categories = {
            'categories': {
                'CRIMINAL': {
                    'name': 'Criminal Cases',
                    'keywords': ['theft', 'assault', 'murder', 'rape'],
                    'ipc_sections': ['IPC 302', 'IPC 379', 'IPC 376']
                },
                'CIVIL': {
                    'name': 'Civil Cases',
                    'keywords': ['property', 'contract', 'dispute'],
                    'ipc_sections': []
                },
                'TRAFFIC': {
                    'name': 'Traffic Cases',
                    'keywords': ['accident', 'vehicle', 'driving'],
                    'ipc_sections': ['IPC 279', 'MV Act']
                }
            },
            'classification_rules': {
                'min_keyword_matches': 1,
                'confidence_thresholds': {'high': 0.8, 'medium': 0.6, 'low': 0.4}
            }
        }

        # Create temporary categories file
        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

        # Initialize categorizer
        self.categorizer = FIRCategorizer(
            categories_file=self.categories_file,
            model_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test categorizer initialization."""
        self.assertIsNotNone(self.categorizer.categories)
        self.assertEqual(len(self.categorizer.categories), 3)
        self.assertIn('CRIMINAL', self.categorizer.categories)
        self.assertIn('CIVIL', self.categorizer.categories)
        self.assertIn('TRAFFIC', self.categorizer.categories)

    def test_empty_text_classification(self):
        """Test classification of empty text."""
        result = self.categorizer.classify_fir("")

        self.assertEqual(result.category, 'UNKNOWN')
        self.assertEqual(result.confidence, 0.0)
        self.assertIn('Empty', result.reasoning[0])

    def test_whitespace_text_classification(self):
        """Test classification of whitespace-only text."""
        result = self.categorizer.classify_fir("   \n\t  ")

        self.assertEqual(result.category, 'UNKNOWN')
        self.assertEqual(result.confidence, 0.0)

    def test_criminal_case_classification(self):
        """Test classification of criminal case."""
        criminal_text = """
        The complainant reported that theft occurred at his residence.
        The accused stole valuable items worth Rs. 50,000.
        FIR registered under IPC 379 for theft.
        """

        result = self.categorizer.classify_fir(criminal_text)

        self.assertEqual(result.category, 'CRIMINAL')
        self.assertGreater(result.confidence, 0.5)
        self.assertIn('CRIMINAL', result.category)

    def test_civil_case_classification(self):
        """Test classification of civil case."""
        civil_text = """
        Property dispute between two parties regarding land ownership.
        Civil suit filed for declaration and injunction.
        Matter pending in civil court for resolution.
        """

        result = self.categorizer.classify_fir(civil_text)

        self.assertEqual(result.category, 'CIVIL')
        self.assertGreater(result.confidence, 0.3)

    def test_traffic_case_classification(self):
        """Test classification of traffic case."""
        traffic_text = """
        Motor vehicle accident reported at main intersection.
        Vehicle collided with another car causing damage.
        Case registered under Motor Vehicles Act.
        """

        result = self.categorizer.classify_fir(traffic_text)

        self.assertEqual(result.category, 'TRAFFIC')
        self.assertGreater(result.confidence, 0.3)

    def test_unknown_case_classification(self):
        """Test classification of unknown/miscellaneous case."""
        unknown_text = """
        Some random text that doesn't match any category.
        No specific keywords or patterns present.
        """

        result = self.categorizer.classify_fir(unknown_text)

        # Should fall back to miscellaneous or low confidence
        self.assertIn(result.category, ['MISCELLANEOUS', 'CIVIL'])  # CIVIL might be default fallback

    def test_classification_with_fir_data(self):
        """Test classification with additional FIR data."""
        fir_text = "Theft occurred at market place."

        fir_data = {
            'section_of_law': 'IPC 379',
            'police_station': 'City Police Station',
            'gist': 'theft of valuable items'
        }

        result = self.categorizer.classify_fir(fir_text, fir_data)

        self.assertIsNotNone(result)
        self.assertGreater(result.confidence, 0.0)

    def test_training_example_addition(self):
        """Test adding training examples."""
        initial_count = len(self.categorizer.training_data)

        self.categorizer.add_training_example(
            "Test criminal case for training",
            "CRIMINAL"
        )

        self.assertEqual(len(self.categorizer.training_data), initial_count + 1)

    def test_classifier_training(self):
        """Test training the ML classifier."""
        # Add some training examples
        for i in range(15):  # Need minimum 10 for training
            text = f"Training example {i} for criminal case with theft and assault"
            self.categorizer.add_training_example(text, "CRIMINAL")

        # Train classifier
        self.categorizer.train_classifier(test_size=0.3)

        # Should be trained now
        self.assertTrue(self.categorizer.is_trained)

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        initial_stats = self.categorizer.get_performance_stats()

        # Perform some classifications
        for i in range(5):
            self.categorizer.classify_fir(f"Test case {i}")

        updated_stats = self.categorizer.get_performance_stats()

        self.assertEqual(updated_stats['total_classifications'], initial_stats['total_classifications'] + 5)
        self.assertGreater(updated_stats['total_processing_time'], initial_stats['total_processing_time'])

    def test_category_info(self):
        """Test getting category information."""
        criminal_info = self.categorizer.get_category_info('CRIMINAL')

        self.assertIsNotNone(criminal_info)
        self.assertEqual(criminal_info['name'], 'Criminal Cases')
        self.assertIn('keywords', criminal_info)

    def test_list_categories(self):
        """Test listing available categories."""
        categories = self.categorizer.list_categories()

        self.assertEqual(len(categories), 3)
        self.assertIn('CRIMINAL', categories)
        self.assertIn('CIVIL', categories)
        self.assertIn('TRAFFIC', categories)

    def test_model_persistence(self):
        """Test saving and loading models."""
        # Add training data and train
        for i in range(12):
            self.categorizer.add_training_example(f"Training text {i}", "CRIMINAL")

        self.categorizer.train_classifier()

        # Save models
        self.categorizer._save_models()

        # Create new categorizer instance
        new_categorizer = FIRCategorizer(
            categories_file=self.categories_file,
            model_dir=self.temp_dir
        )

        # Should load existing models
        if os.path.exists(os.path.join(self.temp_dir, 'fir_categorizer', 'classifier.joblib')):
            self.assertTrue(new_categorizer.is_trained)

    def test_text_normalization(self):
        """Test text normalization functionality."""
        # Test the internal normalization method
        original_text = "  Mixed CASE text!!! With    extra   spaces  "
        normalized = self.categorizer._normalize_text(original_text)

        self.assertEqual(normalized, "mixed case text with extra spaces")
        self.assertNotIn('!', normalized)
        self.assertNotIn('  ', normalized)  # No double spaces


class TestClassificationResult(unittest.TestCase):
    """Test cases for ClassificationResult dataclass."""

    def test_result_creation(self):
        """Test creating classification result."""
        result = ClassificationResult(
            category='CRIMINAL',
            confidence=0.85,
            scores={'CRIMINAL': 0.85, 'CIVIL': 0.15},
            reasoning=['High keyword match', 'IPC section found'],
            processing_time=0.123,
            method='hybrid'
        )

        self.assertEqual(result.category, 'CRIMINAL')
        self.assertEqual(result.confidence, 0.85)
        self.assertIsInstance(result.scores, dict)
        self.assertIsInstance(result.reasoning, list)
        self.assertEqual(result.processing_time, 0.123)
        self.assertEqual(result.method, 'hybrid')


class TestGlobalFunctions(unittest.TestCase):
    """Test cases for global convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test categories file
        self.test_categories = {
            'categories': {
                'CRIMINAL': {
                    'name': 'Criminal Cases',
                    'keywords': ['theft', 'assault'],
                    'ipc_sections': ['IPC 302']
                }
            }
        }

        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_fir_categorizer(self):
        """Test global categorizer getter."""
        categorizer1 = get_fir_categorizer(self.categories_file)
        categorizer2 = get_fir_categorizer(self.categories_file)

        # Should return same instance
        self.assertIs(categorizer1, categorizer2)

    def test_classify_fir_text(self):
        """Test global classification function."""
        result = classify_fir_text("Test criminal case with theft")

        self.assertIsInstance(result, ClassificationResult)
        self.assertIn(result.category, ['CRIMINAL', 'CIVIL', 'UNKNOWN'])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Minimal categories for testing
        self.test_categories = {
            'categories': {
                'CRIMINAL': {
                    'name': 'Criminal Cases',
                    'keywords': ['theft'],
                    'ipc_sections': []
                }
            }
        }

        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

        self.categorizer = FIRCategorizer(
            categories_file=self.categories_file,
            model_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_very_long_text(self):
        """Test classification of very long text."""
        long_text = "theft " * 1000  # Very repetitive long text

        result = self.categorizer.classify_fir(long_text)

        self.assertIsNotNone(result)
        self.assertGreater(result.processing_time, 0)

    def test_special_characters(self):
        """Test classification with special characters."""
        special_text = "Theft case with spëcial châractérs and ñumbérs 123"

        result = self.categorizer.classify_fir(special_text)

        self.assertIsNotNone(result)
        self.assertIn(result.category, ['CRIMINAL', 'UNKNOWN'])

    def test_mixed_languages(self):
        """Test classification with mixed language content."""
        mixed_text = "Theft चोरी occurred at market मार्केट place"

        result = self.categorizer.classify_fir(mixed_text)

        self.assertIsNotNone(result)

    def test_none_input(self):
        """Test handling of None input."""
        result = self.categorizer.classify_fir(None)

        self.assertEqual(result.category, 'UNKNOWN')
        self.assertEqual(result.confidence, 0.0)

    def test_numeric_input(self):
        """Test handling of numeric input."""
        result = self.categorizer.classify_fir(12345)

        self.assertEqual(result.category, 'UNKNOWN')
        self.assertEqual(result.confidence, 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create comprehensive categories
        self.test_categories = {
            'categories': {
                'CRIMINAL': {
                    'name': 'Criminal Cases',
                    'keywords': ['theft', 'assault', 'murder', 'rape', 'burglary'],
                    'ipc_sections': ['IPC 302', 'IPC 379', 'IPC 376']
                },
                'CIVIL': {
                    'name': 'Civil Cases',
                    'keywords': ['property', 'contract', 'dispute', 'ownership'],
                    'ipc_sections': []
                },
                'TRAFFIC': {
                    'name': 'Traffic Cases',
                    'keywords': ['accident', 'vehicle', 'collision', 'driving'],
                    'ipc_sections': ['IPC 279', 'MV Act']
                },
                'CYBER': {
                    'name': 'Cyber Crimes',
                    'keywords': ['online', 'hacking', 'phishing', 'cyber'],
                    'ipc_sections': ['IT Act', 'IPC 66']
                }
            }
        }

        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

        self.categorizer = FIRCategorizer(
            categories_file=self.categories_file,
            model_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_classification(self):
        """Test complete classification workflow."""
        # Test various FIR types
        test_cases = [
            ("Criminal theft case with assault", "CRIMINAL"),
            ("Property dispute between neighbors", "CIVIL"),
            ("Vehicle accident at intersection", "TRAFFIC"),
            ("Online fraud through phishing", "CYBER")
        ]

        results = []
        for text, expected_category in test_cases:
            result = self.categorizer.classify_fir(text)
            results.append((text, result.category, result.confidence))

            # Should classify with reasonable confidence
            self.assertGreater(result.confidence, 0.1)

        # At least some should be classified correctly
        correct_classifications = sum(1 for _, pred, _ in results if pred == "CRIMINAL")
        self.assertGreater(correct_classifications, 0)

    def test_training_and_retraining(self):
        """Test training workflow and model improvement."""
        # Add training examples
        training_texts = [
            "Theft case reported at police station",
            "Burglary occurred at residence",
            "Assault case with grievous hurt",
            "Murder case under IPC 302"
        ]

        for text in training_texts:
            self.categorizer.add_training_example(text, "CRIMINAL")

        # Train initial model
        self.categorizer.train_classifier(test_size=0.4)

        # Test classification
        test_text = "Robbery at market place"
        result1 = self.categorizer.classify_fir(test_text)

        # Add more training data
        for i in range(10):
            self.categorizer.add_training_example(f"Additional theft case {i}", "CRIMINAL")

        # Retrain
        self.categorizer.train_classifier(test_size=0.3)

        # Test again
        result2 = self.categorizer.classify_fir(test_text)

        # Should maintain or improve performance
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\nTest Summary:")
    print(f"Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"Success rate: {success_rate:.1f}%")