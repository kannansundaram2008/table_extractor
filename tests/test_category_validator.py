"""
Tests for Category Validator System
Comprehensive test suite for FIR category validation and quality assessment.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
import time
from unittest.mock import Mock, patch

from utils.category_validator import CategoryValidator, ValidationResult, QualityMetrics, PredictionAnalysis


class TestCategoryValidator(unittest.TestCase):
    """Test cases for CategoryValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test models
        self.temp_dir = tempfile.mkdtemp()

        # Create test categories file
        self.test_categories = {
            'categories': {
                'CRIMINAL': {'name': 'Criminal Cases'},
                'CIVIL': {'name': 'Civil Cases'},
                'TRAFFIC': {'name': 'Traffic Cases'}
            }
        }

        # Create temporary categories file
        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

        # Initialize validator
        self.validator = CategoryValidator(
            categories_file=self.categories_file,
            models_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test validator initialization."""
        self.assertIsNotNone(self.validator.categories)
        self.assertEqual(len(self.validator.categories), 3)

    def test_no_models_available(self):
        """Test behavior when no models are available."""
        summary = self.validator.get_validation_summary()

        self.assertEqual(len(summary['available_models']), 0)
        self.assertEqual(summary['validation_history_count'], 0)

    def test_empty_quality_report(self):
        """Test quality report with no data."""
        quality = self.validator.generate_quality_report()

        self.assertEqual(quality.overall_accuracy, 0.0)
        self.assertEqual(len(quality.category_accuracies), 0)

    def test_prediction_analysis_creation(self):
        """Test creating prediction analysis."""
        analysis = PredictionAnalysis(
            prediction_id="test_1",
            true_category="CRIMINAL",
            predicted_category="CRIMINAL",
            confidence=0.85,
            text_length=150,
            category_difficulty=0.2,
            is_correct=True
        )

        self.assertEqual(analysis.prediction_id, "test_1")
        self.assertEqual(analysis.true_category, "CRIMINAL")
        self.assertEqual(analysis.predicted_category, "CRIMINAL")
        self.assertTrue(analysis.is_correct)
        self.assertIsNone(analysis.error_type)

    def test_incorrect_prediction_analysis(self):
        """Test analysis of incorrect prediction."""
        analysis = PredictionAnalysis(
            prediction_id="test_2",
            true_category="CRIMINAL",
            predicted_category="CIVIL",
            confidence=0.3,
            text_length=50,
            category_difficulty=0.1,
            is_correct=False
        )

        self.assertFalse(analysis.is_correct)
        self.assertIsNotNone(analysis.error_type)  # Should classify error type

    def test_analyze_predictions(self):
        """Test analyzing prediction results."""
        # Mock prediction data
        texts = ["Criminal case text", "Civil case text", "Traffic case text"]
        true_categories = ["CRIMINAL", "CIVIL", "TRAFFIC"]
        predictions = ["CRIMINAL", "CIVIL", "CIVIL"]  # One incorrect
        confidences = [0.8, 0.9, 0.4]

        analyses = self.validator.analyze_predictions(
            "test_model", texts, true_categories, predictions, confidences
        )

        self.assertEqual(len(analyses), 3)
        self.assertEqual(analyses[0].is_correct, True)  # Correct prediction
        self.assertEqual(analyses[1].is_correct, True)  # Correct prediction
        self.assertEqual(analyses[2].is_correct, False)  # Incorrect prediction

    def test_category_difficulty_calculation(self):
        """Test category difficulty calculation."""
        # Add some prediction analyses
        analyses = [
            PredictionAnalysis("1", "CRIMINAL", "CRIMINAL", 0.8, 100, 0.0, True),
            PredictionAnalysis("2", "CRIMINAL", "CIVIL", 0.3, 100, 0.0, False),
            PredictionAnalysis("3", "CIVIL", "CIVIL", 0.9, 100, 0.0, True)
        ]

        self.validator.prediction_analyses = analyses

        # Criminal should be more difficult (had an error)
        criminal_difficulty = self.validator._calculate_category_difficulty("CRIMINAL")
        civil_difficulty = self.validator._calculate_category_difficulty("CIVIL")

        self.assertGreater(criminal_difficulty, civil_difficulty)

    def test_similar_categories_detection(self):
        """Test detection of similar categories."""
        # Test similar categories
        self.assertTrue(self.validator._are_similar_categories("CRIMINAL", "CYBER"))
        self.assertFalse(self.validator._are_similar_categories("CRIMINAL", "CIVIL"))

    def test_rare_category_detection(self):
        """Test detection of rare categories."""
        # Add analyses with imbalanced categories
        analyses = []
        for i in range(20):
            category = "CRIMINAL" if i < 18 else "TRAFFIC"  # TRAFFIC is rare
            analysis = PredictionAnalysis(f"{i}", category, category, 0.8, 100, 0.0, True)
            analyses.append(analysis)

        self.validator.prediction_analyses = analyses

        self.assertTrue(self.validator._is_rare_category("TRAFFIC"))
        self.assertFalse(self.validator._is_rare_category("CRIMINAL"))

    def test_data_quality_score_calculation(self):
        """Test data quality score calculation."""
        analyses = [
            PredictionAnalysis("1", "CRIMINAL", "CRIMINAL", 0.8, 200, 0.0, True),
            PredictionAnalysis("2", "CIVIL", "CIVIL", 0.9, 300, 0.0, True),
            PredictionAnalysis("3", "TRAFFIC", "TRAFFIC", 0.7, 250, 0.0, True)
        ]

        score = self.validator._calculate_data_quality_score(analyses)

        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_model_stability_score_calculation(self):
        """Test model stability score calculation."""
        analyses = [
            PredictionAnalysis("1", "CRIMINAL", "CRIMINAL", 0.9, 100, 0.0, True),
            PredictionAnalysis("2", "CIVIL", "CIVIL", 0.8, 100, 0.0, True),
            PredictionAnalysis("3", "TRAFFIC", "CIVIL", 0.3, 100, 0.0, False)  # Low confidence error
        ]

        score = self.validator._calculate_model_stability_score(analyses)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_error_recommendations_generation(self):
        """Test generation of error recommendations."""
        from collections import Counter

        error_types = Counter({'low_confidence': 5, 'similar_category': 3})
        category_error_rates = {'CRIMINAL': 0.4, 'CIVIL': 0.2}

        recommendations = self.validator._generate_error_recommendations(error_types, category_error_rates)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    def test_validation_summary(self):
        """Test getting validation summary."""
        # Add some mock data
        self.validator.validation_history = [{'test': 'data'}]
        self.validator.prediction_analyses = [{'test': 'analysis'}]

        summary = self.validator.get_validation_summary()

        self.assertIn('validation_history_count', summary)
        self.assertIn('prediction_analyses_count', summary)
        self.assertEqual(summary['validation_history_count'], 1)
        self.assertEqual(summary['prediction_analyses_count'], 1)


class TestValidationResult(unittest.TestCase):
    """Test cases for ValidationResult dataclass."""

    def test_result_creation(self):
        """Test creating validation result."""
        result = ValidationResult(
            model_name="test_model",
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            kappa_score=0.80,
            confusion_matrix=[[10, 2], [1, 12]],
            classification_report={'CRIMINAL': {'precision': 0.8}},
            validation_time=5.2,
            test_set_size=25
        )

        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.accuracy, 0.85)
        self.assertEqual(result.test_set_size, 25)
        self.assertIsInstance(result.confusion_matrix, list)


class TestQualityMetrics(unittest.TestCase):
    """Test cases for QualityMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(
            overall_accuracy=0.85,
            category_accuracies={'CRIMINAL': 0.9, 'CIVIL': 0.8},
            confidence_distribution={'0.0-0.3': 5, '0.8-1.0': 15},
            processing_time_stats={'mean': 0.5, 'std': 0.1},
            error_analysis={'total_errors': 3, 'error_rate': 0.15},
            data_quality_score=0.8,
            model_stability_score=0.9
        )

        self.assertEqual(metrics.overall_accuracy, 0.85)
        self.assertIsInstance(metrics.category_accuracies, dict)
        self.assertIsInstance(metrics.confidence_distribution, dict)


class TestErrorAnalysis(unittest.TestCase):
    """Test error analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        self.test_categories = {
            'categories': {
                'CRIMINAL': {'name': 'Criminal Cases'},
                'CIVIL': {'name': 'Civil Cases'},
                'TRAFFIC': {'name': 'Traffic Cases'},
                'CYBER': {'name': 'Cyber Crimes'}
            }
        }

        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

        self.validator = CategoryValidator(
            categories_file=self.categories_file,
            models_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_error_analysis_report_no_errors(self):
        """Test error analysis when no errors exist."""
        # Add only correct predictions
        analyses = [
            PredictionAnalysis("1", "CRIMINAL", "CRIMINAL", 0.8, 100, 0.0, True),
            PredictionAnalysis("2", "CIVIL", "CIVIL", 0.9, 100, 0.0, True)
        ]

        self.validator.prediction_analyses = analyses

        report = self.validator.generate_error_analysis_report()

        self.assertEqual(report['total_errors'], 0)
        self.assertEqual(report['error_rate'], 0.0)

    def test_error_analysis_report_with_errors(self):
        """Test error analysis with prediction errors."""
        # Add mix of correct and incorrect predictions
        analyses = [
            PredictionAnalysis("1", "CRIMINAL", "CRIMINAL", 0.8, 100, 0.0, True),
            PredictionAnalysis("2", "CIVIL", "TRAFFIC", 0.3, 50, 0.0, False),
            PredictionAnalysis("3", "TRAFFIC", "CIVIL", 0.4, 75, 0.0, False),
            PredictionAnalysis("4", "CYBER", "CYBER", 0.9, 200, 0.0, True)
        ]

        self.validator.prediction_analyses = analyses

        report = self.validator.generate_error_analysis_report()

        self.assertEqual(report['total_errors'], 2)
        self.assertEqual(report['error_rate'], 0.5)  # 2 errors out of 4
        self.assertIn('error_type_distribution', report)
        self.assertIn('category_error_rates', report)

    def test_model_recommendations(self):
        """Test generation of model recommendations."""
        # Set up quality metrics with poor performance
        self.validator.quality_metrics = {
            'overall_accuracy': 0.5,
            'data_quality_score': 0.4,
            'model_stability_score': 0.3,
            'error_analysis': {'error_rate': 0.5}
        }

        recommendations = self.validator.get_model_recommendations()

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the validation system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create comprehensive categories
        self.test_categories = {
            'categories': {
                'CRIMINAL': {'name': 'Criminal Cases'},
                'CIVIL': {'name': 'Civil Cases'},
                'TRAFFIC': {'name': 'Traffic Cases'}
            }
        }

        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

        self.validator = CategoryValidator(
            categories_file=self.categories_file,
            models_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_validation_workflow(self):
        """Test complete validation workflow."""
        # Simulate prediction results
        texts = [
            "Theft case with assault",
            "Property dispute matter",
            "Vehicle accident case",
            "Burglary at residence"
        ]
        true_categories = ["CRIMINAL", "CIVIL", "TRAFFIC", "CRIMINAL"]
        predictions = ["CRIMINAL", "CIVIL", "TRAFFIC", "CIVIL"]  # Last one incorrect
        confidences = [0.8, 0.9, 0.7, 0.4]

        # Analyze predictions
        analyses = self.validator.analyze_predictions(
            "test_model", texts, true_categories, predictions, confidences
        )

        self.assertEqual(len(analyses), 4)

        # Generate quality report
        quality = self.validator.generate_quality_report()

        self.assertEqual(quality.overall_accuracy, 0.75)  # 3 out of 4 correct
        self.assertIn('CRIMINAL', quality.category_accuracies)
        self.assertIn('CIVIL', quality.category_accuracies)

        # Generate error analysis
        error_report = self.validator.generate_error_analysis_report()

        self.assertEqual(error_report['total_errors'], 1)
        self.assertEqual(error_report['error_rate'], 0.25)

    def test_reset_validation_data(self):
        """Test resetting validation data."""
        # Add some data
        self.validator.validation_history = [{'test': 'data'}]
        self.validator.prediction_analyses = [{'test': 'analysis'}]
        self.validator.quality_metrics = {'test': 'metrics'}

        # Reset
        self.validator.reset_validation_data()

        # Should be empty
        self.assertEqual(len(self.validator.validation_history), 0)
        self.assertEqual(len(self.validator.prediction_analyses), 0)
        self.assertEqual(len(self.validator.quality_metrics), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        self.test_categories = {
            'categories': {
                'CRIMINAL': {'name': 'Criminal Cases'}
            }
        }

        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

        self.validator = CategoryValidator(
            categories_file=self.categories_file,
            models_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_single_category_difficulty(self):
        """Test difficulty calculation with single category."""
        analyses = [
            PredictionAnalysis("1", "CRIMINAL", "CRIMINAL", 0.8, 100, 0.0, True)
        ]

        self.validator.prediction_analyses = analyses

        difficulty = self.validator._calculate_category_difficulty("CRIMINAL")
        self.assertEqual(difficulty, 0.0)  # No errors = no difficulty

    def test_empty_analyses_stability(self):
        """Test stability calculation with empty analyses."""
        score = self.validator._calculate_model_stability_score([])
        self.assertEqual(score, 0.0)

    def test_extreme_confidence_values(self):
        """Test handling of extreme confidence values."""
        analyses = [
            PredictionAnalysis("1", "CRIMINAL", "CRIMINAL", 0.0, 100, 0.0, True),
            PredictionAnalysis("2", "CIVIL", "CIVIL", 1.0, 100, 0.0, True)
        ]

        score = self.validator._calculate_model_stability_score(analyses)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_very_short_text_analysis(self):
        """Test analysis of very short text predictions."""
        analysis = PredictionAnalysis(
            "1", "CRIMINAL", "CIVIL", 0.3, 5, 0.0, False  # Very short text
        )

        error_type = self.validator._classify_error_type(
            analysis.true_category, analysis.predicted_category,
            "short", analysis.confidence
        )

        self.assertEqual(error_type, 'short_text')


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