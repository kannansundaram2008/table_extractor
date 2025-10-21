"""
Tests for Category Trainer System
Comprehensive test suite for FIR category training functionality.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
import time
from unittest.mock import Mock, patch

from utils.category_trainer import CategoryTrainer, TrainingExample, TrainingResult, create_synthetic_training_data


class TestCategoryTrainer(unittest.TestCase):
    """Test cases for CategoryTrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()

        # Create test categories file
        self.test_categories = {
            'categories': {
                'CRIMINAL': {
                    'name': 'Criminal Cases',
                    'keywords': ['theft', 'assault', 'murder']
                },
                'CIVIL': {
                    'name': 'Civil Cases',
                    'keywords': ['property', 'contract', 'dispute']
                },
                'TRAFFIC': {
                    'name': 'Traffic Cases',
                    'keywords': ['accident', 'vehicle', 'driving']
                }
            },
            'training_config': {
                'test_size': 0.2,
                'validation_size': 0.1,
                'random_state': 42,
                'max_features': 1000,
                'ngram_range': [1, 2]
            }
        }

        # Create temporary categories file
        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

        # Initialize trainer
        self.trainer = CategoryTrainer(
            categories_file=self.categories_file,
            training_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer.categories)
        self.assertEqual(len(self.trainer.categories), 3)
        self.assertIn('CRIMINAL', self.trainer.categories)
        self.assertIn('CIVIL', self.trainer.categories)
        self.assertIn('TRAFFIC', self.trainer.categories)

    def test_add_training_example(self):
        """Test adding training examples."""
        initial_count = len(self.trainer.training_examples)

        self.trainer.add_training_example(
            "Test criminal case for training",
            "CRIMINAL",
            "test_file_1"
        )

        self.assertEqual(len(self.trainer.training_examples), initial_count + 1)
        self.assertEqual(self.trainer.training_examples[-1].category, "CRIMINAL")
        self.assertEqual(self.trainer.training_examples[-1].source_file, "test_file_1")

    def test_add_validation_example(self):
        """Test adding validation examples."""
        initial_count = len(self.trainer.validation_examples)

        self.trainer.add_validation_example(
            "Test civil case for validation",
            "CIVIL",
            "validation_file_1"
        )

        self.assertEqual(len(self.trainer.validation_examples), initial_count + 1)
        self.assertEqual(self.trainer.validation_examples[-1].category, "CIVIL")

    def test_add_example_invalid_category(self):
        """Test adding example with invalid category."""
        initial_count = len(self.trainer.training_examples)

        self.trainer.add_training_example(
            "Test case with invalid category",
            "INVALID_CATEGORY"
        )

        # Should not add example with invalid category
        self.assertEqual(len(self.trainer.training_examples), initial_count)

    def test_insufficient_training_data(self):
        """Test training with insufficient data."""
        # Add only 5 examples (less than minimum 10)
        for i in range(5):
            self.trainer.add_training_example(f"Test case {i}", "CRIMINAL")

        # Should raise error due to insufficient data
        with self.assertRaises(ValueError):
            self.trainer.prepare_training_data()

    def test_prepare_training_data(self):
        """Test training data preparation."""
        # Add sufficient training examples
        for i in range(15):
            self.trainer.add_training_example(f"Test case {i}", "CRIMINAL")

        X, y, vectorizer = self.trainer.prepare_training_data()

        self.assertEqual(X.shape[0], 15)  # 15 examples
        self.assertEqual(len(y), 15)  # 15 labels
        self.assertGreater(X.shape[1], 0)  # Some features

    def test_model_training(self):
        """Test training a specific model."""
        # Add training data
        for i in range(20):
            category = "CRIMINAL" if i % 2 == 0 else "CIVIL"
            self.trainer.add_training_example(f"Test case {i}", category)

        # Train model
        result = self.trainer.train_model('random_forest', test_size=0.3)

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.model_name, 'random_forest')
        self.assertGreater(result.accuracy, 0.0)
        self.assertEqual(result.test_samples, 6)  # 30% of 20

    def test_cross_validation(self):
        """Test cross-validation functionality."""
        # Add training data
        for i in range(20):
            category = "CRIMINAL" if i % 2 == 0 else "CIVIL"
            self.trainer.add_training_example(f"Test case {i}", category)

        # Perform cross-validation
        cv_results = self.trainer.cross_validate_model('random_forest', cv_folds=3)

        self.assertIn('mean_accuracy', cv_results)
        self.assertIn('std_accuracy', cv_results)
        self.assertIn('cv_scores', cv_results)
        self.assertEqual(len(cv_results['cv_scores']), 3)

    def test_analyze_training_data(self):
        """Test training data analysis."""
        # Add examples with different categories and sources
        for i in range(10):
            category = "CRIMINAL" if i % 3 == 0 else "CIVIL" if i % 3 == 1 else "TRAFFIC"
            self.trainer.add_training_example(f"Test case {i}", category, f"source_{i%3}")

        analysis = self.trainer.analyze_training_data()

        self.assertIn('total_examples', analysis)
        self.assertIn('category_distribution', analysis)
        self.assertIn('balance_score', analysis)
        self.assertEqual(analysis['total_examples'], 10)
        self.assertEqual(len(analysis['category_distribution']), 3)

    def test_data_augmentation(self):
        """Test data augmentation functionality."""
        # Add some training examples
        for i in range(5):
            self.trainer.add_training_example(f"Original case {i}", "CRIMINAL")

        initial_count = len(self.trainer.training_examples)

        # Augment data
        self.trainer.augment_training_data(target_per_category=10)

        # Should have more examples now
        self.assertGreater(len(self.trainer.training_examples), initial_count)

    def test_save_and_load_training_data(self):
        """Test saving and loading training data."""
        # Add some examples
        for i in range(10):
            self.trainer.add_training_example(f"Test case {i}", "CRIMINAL", f"file_{i}")

        # Save data
        self.trainer.save_training_data()

        # Create new trainer instance
        new_trainer = CategoryTrainer(
            categories_file=self.categories_file,
            training_dir=self.temp_dir
        )

        # Should load existing data
        self.assertGreater(len(new_trainer.training_examples), 0)

    def test_load_from_csv(self):
        """Test loading training data from CSV file."""
        # Create test CSV file
        csv_content = "text,category,source\n"
        csv_content += "Criminal case 1,CRIMINAL,file1\n"
        csv_content += "Civil case 1,CIVIL,file2\n"
        csv_content += "Traffic case 1,TRAFFIC,file3\n"

        csv_file = os.path.join(self.temp_dir, 'test_data.csv')
        with open(csv_file, 'w') as f:
            f.write(csv_content)

        # Load data
        self.trainer.load_training_data_from_file(csv_file)

        self.assertEqual(len(self.trainer.training_examples), 3)

    def test_load_from_json(self):
        """Test loading training data from JSON file."""
        # Create test JSON file
        json_data = [
            {"text": "Criminal case 1", "category": "CRIMINAL", "source": "file1"},
            {"text": "Civil case 1", "category": "CIVIL", "source": "file2"}
        ]

        json_file = os.path.join(self.temp_dir, 'test_data.json')
        with open(json_file, 'w') as f:
            json.dump(json_data, f)

        # Load data
        self.trainer.load_training_data_from_file(json_file)

        self.assertEqual(len(self.trainer.training_examples), 2)

    def test_training_summary(self):
        """Test getting training summary."""
        # Add some examples
        for i in range(5):
            self.trainer.add_training_example(f"Test case {i}", "CRIMINAL")

        summary = self.trainer.get_training_summary()

        self.assertIn('total_training_examples', summary)
        self.assertIn('categories', summary)
        self.assertIn('available_models', summary)
        self.assertEqual(summary['total_training_examples'], 5)
        self.assertIn('CRIMINAL', summary['categories'])


class TestTrainingExample(unittest.TestCase):
    """Test cases for TrainingExample dataclass."""

    def test_example_creation(self):
        """Test creating training example."""
        example = TrainingExample(
            text="Test FIR text",
            category="CRIMINAL",
            source_file="test_file.txt",
            fir_data={"section": "IPC 379"},
            metadata={"priority": "high"}
        )

        self.assertEqual(example.text, "Test FIR text")
        self.assertEqual(example.category, "CRIMINAL")
        self.assertEqual(example.source_file, "test_file.txt")
        self.assertIsNotNone(example.timestamp)

    def test_example_creation_minimal(self):
        """Test creating training example with minimal data."""
        example = TrainingExample(
            text="Test text",
            category="CIVIL"
        )

        self.assertEqual(example.text, "Test text")
        self.assertEqual(example.category, "CIVIL")
        self.assertIsNone(example.fir_data)
        self.assertIsNone(example.metadata)
        self.assertIsNotNone(example.timestamp)


class TestTrainingResult(unittest.TestCase):
    """Test cases for TrainingResult dataclass."""

    def test_result_creation(self):
        """Test creating training result."""
        result = TrainingResult(
            model_name="random_forest",
            accuracy=0.85,
            precision=0.83,
            recall=0.85,
            f1_score=0.84,
            training_time=12.5,
            test_samples=50,
            feature_count=1000,
            model_size=50000,
            confusion_matrix=[[40, 5], [3, 47]],
            classification_report={'CRIMINAL': {'precision': 0.8}}
        )

        self.assertEqual(result.model_name, "random_forest")
        self.assertEqual(result.accuracy, 0.85)
        self.assertEqual(result.test_samples, 50)
        self.assertIsInstance(result.confusion_matrix, list)


class TestSyntheticData(unittest.TestCase):
    """Test cases for synthetic data generation."""

    def test_create_synthetic_data(self):
        """Test creating synthetic training data."""
        test_categories = {
            'CRIMINAL': {'name': 'Criminal Cases'},
            'CIVIL': {'name': 'Civil Cases'}
        }

        examples = create_synthetic_training_data(test_categories, examples_per_category=5)

        self.assertEqual(len(examples), 10)  # 2 categories * 5 examples
        self.assertEqual(len([e for e in examples if e.category == 'CRIMINAL']), 5)
        self.assertEqual(len([e for e in examples if e.category == 'CIVIL']), 5)

        # Check that all examples have required fields
        for example in examples:
            self.assertIsInstance(example.text, str)
            self.assertGreater(len(example.text), 0)
            self.assertIn(example.category, ['CRIMINAL', 'CIVIL'])
            self.assertIsNotNone(example.timestamp)


class TestErrorConditions(unittest.TestCase):
    """Test error conditions and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        self.test_categories = {
            'categories': {
                'CRIMINAL': {'name': 'Criminal Cases', 'keywords': []}
            }
        }

        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

        self.trainer = CategoryTrainer(
            categories_file=self.categories_file,
            training_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unknown_model_training(self):
        """Test training unknown model."""
        # Add some training data
        for i in range(15):
            self.trainer.add_training_example(f"Test case {i}", "CRIMINAL")

        # Try to train unknown model
        with self.assertRaises(ValueError):
            self.trainer.train_model("unknown_model")

    def test_empty_training_data_analysis(self):
        """Test analyzing empty training data."""
        analysis = self.trainer.analyze_training_data()

        self.assertIn('error', analysis)
        self.assertEqual(analysis['error'], 'No training data available')

    def test_unsupported_file_format(self):
        """Test loading unsupported file format."""
        # Create file with unsupported extension
        unsupported_file = os.path.join(self.temp_dir, 'test.txt')
        with open(unsupported_file, 'w') as f:
            f.write("test data")

        # Should not raise error but also not load anything
        self.trainer.load_training_data_from_file(unsupported_file)

        # Should have no training examples
        self.assertEqual(len(self.trainer.training_examples), 0)

    def test_malformed_csv(self):
        """Test loading malformed CSV file."""
        # Create malformed CSV
        csv_file = os.path.join(self.temp_dir, 'malformed.csv')
        with open(csv_file, 'w') as f:
            f.write("invalid,csv,content\nwithout,proper,structure")

        # Should not raise error but handle gracefully
        self.trainer.load_training_data_from_file(csv_file)

        # Should have no valid examples loaded
        self.assertEqual(len(self.trainer.training_examples), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the training system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create comprehensive categories
        self.test_categories = {
            'categories': {
                'CRIMINAL': {'name': 'Criminal Cases', 'keywords': ['theft', 'assault']},
                'CIVIL': {'name': 'Civil Cases', 'keywords': ['property', 'contract']},
                'TRAFFIC': {'name': 'Traffic Cases', 'keywords': ['accident', 'vehicle']}
            }
        }

        self.categories_file = os.path.join(self.temp_dir, 'test_categories.json')
        with open(self.categories_file, 'w') as f:
            json.dump(self.test_categories, f)

        self.trainer = CategoryTrainer(
            categories_file=self.categories_file,
            training_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_training_workflow(self):
        """Test complete training workflow."""
        # Add diverse training examples
        training_texts = [
            ("Theft occurred at residence", "CRIMINAL"),
            ("Property dispute case", "CIVIL"),
            ("Vehicle accident reported", "TRAFFIC"),
            ("Assault with grievous hurt", "CRIMINAL"),
            ("Contract breach dispute", "CIVIL"),
            ("Hit and run accident", "TRAFFIC")
        ]

        for text, category in training_texts:
            self.trainer.add_training_example(text, category, "workflow_test")

        # Analyze data
        analysis = self.trainer.analyze_training_data()
        self.assertEqual(analysis['total_examples'], 6)

        # Train model
        result = self.trainer.train_model('random_forest', test_size=0.3)

        self.assertIsInstance(result, TrainingResult)
        self.assertGreater(result.accuracy, 0.0)

        # Test cross-validation
        cv_results = self.trainer.cross_validate_model('random_forest', cv_folds=3)
        self.assertIn('mean_accuracy', cv_results)

    def test_model_comparison(self):
        """Test comparing multiple models."""
        # Add training data
        for i in range(30):
            category = "CRIMINAL" if i % 2 == 0 else "CIVIL"
            self.trainer.add_training_example(f"Test case {i}", category)

        # Train all models
        results = self.trainer.train_all_models()

        # Should have results for all models
        self.assertIn('random_forest', results)
        self.assertIn('logistic_regression', results)

        # All results should be TrainingResult instances
        for model_name, result in results.items():
            self.assertIsInstance(result, TrainingResult)
            self.assertGreater(result.accuracy, 0.0)


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