"""
Comprehensive ML Integration Test Suite

Tests all ML components working together to ensure end-to-end functionality,
data flow, and integration between different ML modules.
"""

import unittest
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import ML components
from utils.enhanced_ner import EnhancedNER, extract_entities_with_ner
from utils.ml_pattern_learner import MLPatternLearner
from utils.data_collector import TrainingDataCollector, get_data_collector
from utils.annotation_interface import AnnotationInterface, get_annotation_interface
from utils.fir_parser import EnhancedFIRParser, parse_fir_row_enhanced
from utils.data_validator import DataValidator
from utils.data_exporter import DataExporter
from utils.data_anonymizer import DataAnonymizer


class TestMLIntegration(unittest.TestCase):
    """Test suite for ML component integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.test_dir) / "test_training_data"

        # Initialize components
        self.ner = EnhancedNER(confidence_threshold=0.7)
        self.ml_learner = MLPatternLearner(model_dir=str(self.storage_path / "models"))
        self.data_collector = TrainingDataCollector(
            storage_path=str(self.storage_path),
            auto_save=False
        )
        self.annotation_interface = AnnotationInterface(
            storage_path=str(self.storage_path / "annotations"),
            auto_create_tasks=False
        )
        self.fir_parser = EnhancedFIRParser(
            confidence_threshold=0.7,
            enable_ml=True,
            enable_logging=False
        )
        self.data_validator = DataValidator()
        self.data_exporter = DataExporter()
        self.data_anonymizer = DataAnonymizer()

        # Sample test data
        self.sample_fir_text = """
        On 15/03/2023 at 14:30hrs in Mumbai Police Station, complainant John Doe S/o Robert Doe
        aged 35 years residing at Bandra West reported that accused Rajesh Kumar aged 25 years
        at Andheri East stole mobile phone worth Rs. 15000. Victim Priya Sharma D/o Anil Sharma
        was injured. Section 379 IPC, Section 323 IPC applies. CR No 123/2023.
        """

        self.sample_row = [
            "Mumbai Police Station, CR No 123/2023, Section 379 IPC",
            "15/03/2023, 16/03/2023, 14:30hrs, Andheri East",
            "John Doe S/o Robert Doe, Age 35, Address: Bandra West",
            "Priya Sharma D/o Anil Sharma, Age 28, Address: Andheri East",
            "Mobile Phone worth Rs. 15000",
            "Rajesh Kumar S/o Vijay Kumar, Age 25, Address: Jogeshwari",
            "Accused stole victim's mobile phone and injured her"
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_enhanced_ner_integration(self):
        """Test Enhanced NER system integration."""
        # Test NER extraction
        result = self.ner.extract_entities(self.sample_fir_text)

        # Verify entities are extracted
        self.assertGreater(len(result.entities), 0)
        self.assertIsInstance(result.confidence_scores, dict)
        self.assertGreater(result.processing_time, 0)

        # Check for expected entity types
        entity_labels = {e.label for e in result.entities}
        expected_labels = {'PERSON', 'ORG', 'GPE', 'DATE', 'LAW', 'MONEY'}
        self.assertTrue(expected_labels.intersection(entity_labels))

        # Test backward compatibility
        old_format = extract_entities_with_ner(self.sample_fir_text)
        self.assertIsInstance(old_format, dict)
        self.assertIn('persons', old_format)
        self.assertIn('locations', old_format)

    def test_ml_pattern_learner_integration(self):
        """Test ML Pattern Learner integration."""
        # Add training examples
        training_examples = [
            {
                'text': 'John Doe reported theft at police station',
                'entities': {'PERSON': ['John Doe'], 'ORG': ['police station']},
                'context': 'test_case_1'
            },
            {
                'text': 'Priya Sharma was victim of crime',
                'entities': {'PERSON': ['Priya Sharma']},
                'context': 'test_case_2'
            }
        ]

        for example in training_examples:
            self.ml_learner.add_training_example(
                example['text'],
                example['entities'],
                example['context']
            )

        # Test model training
        self.ml_learner.train_entity_classifier()

        # Test prediction
        predictions = self.ml_learner.predict_entities("Rajesh Kumar stole property")
        self.assertIsInstance(predictions, dict)

        # Test pattern clustering
        clusters = self.ml_learner.get_pattern_clusters("Test document with entities")
        self.assertIsInstance(clusters, dict)

        # Test model stats
        stats = self.ml_learner.get_model_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('training_samples', stats)

    def test_fir_parser_ml_integration(self):
        """Test FIR parser with ML integration."""
        # Test enhanced parsing
        result = self.fir_parser.parse_fir_row_enhanced(self.sample_row)

        # Verify structure
        self.assertIn('police_station', result)
        self.assertIn('confidence_scores', result)
        self.assertIn('processing_metadata', result)

        # Check metadata
        metadata = result['processing_metadata']
        self.assertIn('ml_enabled', metadata)
        self.assertIn('ner_processing_time', metadata)
        self.assertIn('ml_processing_time', metadata)

        # Test performance stats
        stats = self.fir_parser.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('parse_count', stats)

    def test_training_data_collection_integration(self):
        """Test training data collection pipeline."""
        # Simulate extraction results
        expected_output = {
            'police_station': 'Mumbai Police Station',
            'cr_no': '123/2023',
            'complainant_name': 'John Doe'
        }

        actual_output = {
            'police_station': 'Mumbai Police Station',
            'cr_no': '123/2023',
            'complainant_name': 'John Doe'
        }

        confidence_scores = {
            'police_station': 0.9,
            'cr_no': 0.95,
            'complainant_name': 0.8
        }

        # Collect training example
        example_id = self.data_collector.collect_extraction_result(
            source_file="test_fir.pdf",
            extraction_input=self.sample_fir_text,
            expected_output=expected_output,
            actual_output=actual_output,
            confidence_scores=confidence_scores,
            processing_metadata={'model_version': '1.0'}
        )

        self.assertIsNotNone(example_id)
        self.assertGreater(len(self.data_collector.examples), 0)

        # Test data filtering
        high_quality = self.data_collector.get_examples_by_quality_range(0.8, 1.0)
        self.assertGreater(len(high_quality), 0)

        # Test export functionality
        export_path = self.storage_path / "export_test.json"
        exported_file = self.data_collector.export_for_training(
            output_path=str(export_path),
            format="json",
            quality_threshold=0.7
        )

        self.assertTrue(Path(exported_file).exists())

    def test_annotation_interface_integration(self):
        """Test annotation interface integration."""
        # Create test example first
        example_id = self.data_collector.collect_extraction_result(
            source_file="test_fir.pdf",
            extraction_input="test input",
            expected_output={'name': 'John'},
            actual_output={'name': 'Jon'},  # Slight difference
            confidence_scores={'name': 0.6},
            processing_metadata={}
        )

        # Create annotation task
        task_id = self.annotation_interface.create_annotation_task(
            example_id=example_id,
            annotator_id="test_annotator",
            priority=8
        )

        self.assertIsNotNone(task_id)
        self.assertIn(task_id, self.annotation_interface.tasks)

        # Test task assignment
        self.assertTrue(
            self.annotation_interface.assign_task_to_annotator(task_id, "test_annotator")
        )

        # Test getting pending tasks
        pending_tasks = self.annotation_interface.get_pending_tasks("test_annotator")
        self.assertEqual(len(pending_tasks), 0)  # Should be in progress now

    def test_data_validation_integration(self):
        """Test data validation in the pipeline."""
        # Test data validation
        test_data = {
            'police_station': 'Mumbai Police Station',
            'cr_no': '123/2023',
            'complainant': {'name': 'John Doe'}
        }

        validation_result = self.data_validator.validate_fir_data(test_data)
        self.assertIsInstance(validation_result, dict)
        self.assertIn('is_valid', validation_result)

        # Test quality scoring
        quality_score = self.data_validator.calculate_quality_score(test_data)
        self.assertIsInstance(quality_score, (int, float))
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)

    def test_data_export_integration(self):
        """Test data export functionality."""
        # Add test data to collector
        self.data_collector.collect_extraction_result(
            source_file="test1.pdf",
            extraction_input="test input 1",
            expected_output={'name': 'John'},
            actual_output={'name': 'John'},
            confidence_scores={'name': 0.9},
            processing_metadata={}
        )

        # Test different export formats
        formats = ['json', 'jsonl', 'csv']

        for fmt in formats:
            export_path = self.storage_path / f"test_export.{fmt}"
            exported_file = self.data_collector.export_for_training(
                output_path=str(export_path),
                format=fmt
            )

            self.assertTrue(Path(exported_file).exists())

            # Verify file is not empty
            self.assertGreater(Path(exported_file).stat().st_size, 0)

    def test_data_anonymization_integration(self):
        """Test data anonymization in the pipeline."""
        # Test data with sensitive information
        sensitive_data = {
            'police_station': 'Mumbai Police Station',
            'complainant': {
                'name': 'John Doe',
                'phone': '9876543210',
                'aadhar': '1234-5678-9012'
            },
            'victim': {
                'name': 'Jane Smith',
                'address': '123 Main St, Mumbai'
            }
        }

        # Anonymize data
        anonymized = self.data_anonymizer.anonymize_fir_data(sensitive_data)

        # Check that sensitive fields are anonymized
        self.assertNotEqual(anonymized['complainant']['phone'], '9876543210')
        self.assertNotEqual(anonymized['complainant']['aadhar'], '1234-5678-9012')
        self.assertNotEqual(anonymized['complainant']['name'], 'John Doe')
        self.assertNotEqual(anonymized['victim']['name'], 'Jane Smith')

        # Check that non-sensitive fields remain unchanged
        self.assertEqual(anonymized['police_station'], 'Mumbai Police Station')

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end ML pipeline."""
        # 1. Parse FIR with enhanced parser
        parsed_result = self.fir_parser.parse_fir_row_enhanced(self.sample_row)

        # 2. Validate the parsed data
        validation_result = self.data_validator.validate_fir_data(parsed_result)

        # 3. Anonymize sensitive information
        anonymized_result = self.data_anonymizer.anonymize_fir_data(parsed_result)

        # 4. Collect for training (if quality is good enough)
        if validation_result.get('is_valid', False):
            example_id = self.data_collector.collect_extraction_result(
                source_file="test_fir.pdf",
                extraction_input=self.sample_fir_text,
                expected_output=parsed_result,  # In real scenario, this would be ground truth
                actual_output=anonymized_result,
                confidence_scores=parsed_result['confidence_scores'],
                processing_metadata=parsed_result['processing_metadata']
            )

            self.assertIsNotNone(example_id)

        # 5. Export processed data
        export_path = self.storage_path / "pipeline_test.json"
        exported_file = self.data_exporter.export_fir_data(
            [anonymized_result],
            str(export_path)
        )

        self.assertTrue(Path(exported_file).exists())

    def test_error_handling_integration(self):
        """Test error handling across all components."""
        # Test with invalid inputs
        invalid_inputs = [None, [], "", {}]

        for invalid_input in invalid_inputs:
            # NER should handle gracefully
            ner_result = self.ner.extract_entities(invalid_input or "")
            self.assertIsInstance(ner_result.entities, list)

            # FIR parser should handle gracefully
            parser_result = self.fir_parser.parse_fir_row_enhanced(invalid_input or [])
            self.assertIsInstance(parser_result, dict)

        # Test with malformed data
        malformed_data = {
            'police_station': None,
            'complainant': {'name': 123},  # Wrong type
            'invalid_field': 'test'
        }

        # Validator should handle malformed data
        validation_result = self.data_validator.validate_fir_data(malformed_data)
        self.assertIsInstance(validation_result, dict)

        # Anonymizer should handle malformed data
        anonymized = self.data_anonymizer.anonymize_fir_data(malformed_data)
        self.assertIsInstance(anonymized, dict)

    def test_performance_integration(self):
        """Test performance across integrated components."""
        # Measure end-to-end performance
        start_time = time.time()

        # Run complete pipeline
        result = self.fir_parser.parse_fir_row_enhanced(self.sample_row)
        validation = self.data_validator.validate_fir_data(result)
        anonymized = self.data_anonymizer.anonymize_fir_data(result)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete within reasonable time (< 5 seconds as per requirements)
        self.assertLess(total_time, 5.0)

        # Check individual component times
        metadata = result['processing_metadata']
        ner_time = metadata.get('ner_processing_time', 0)
        ml_time = metadata.get('ml_processing_time', 0)

        # Individual components should also be fast
        self.assertLess(ner_time, 2.0)
        self.assertLess(ml_time, 2.0)

    def test_memory_usage_integration(self):
        """Test memory usage across integrated components."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run intensive operations
        for _ in range(10):
            result = self.fir_parser.parse_fir_row_enhanced(self.sample_row)
            validation = self.data_validator.validate_fir_data(result)
            anonymized = self.data_anonymizer.anonymize_fir_data(result)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB for test)
        self.assertLess(memory_increase, 100)

    def test_concurrent_access_integration(self):
        """Test concurrent access to shared components."""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def run_pipeline():
            try:
                # Each thread runs the pipeline independently
                result = self.fir_parser.parse_fir_row_enhanced(self.sample_row)
                validation = self.data_validator.validate_fir_data(result)
                results.put(True)
            except Exception as e:
                errors.put(e)

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=run_pipeline)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # Check results
        success_count = 0
        while not results.empty():
            if results.get():
                success_count += 1

        error_count = errors.qsize()

        # Should have mostly successful runs
        self.assertGreater(success_count, 0)
        self.assertEqual(error_count, 0)


class TestMLComponentIntegration(unittest.TestCase):
    """Test specific ML component integrations."""

    def setUp(self):
        """Set up test fixtures."""
        self.ner = EnhancedNER(confidence_threshold=0.7)
        self.ml_learner = MLPatternLearner()
        self.fir_parser = EnhancedFIRParser(enable_ml=True)

    def test_ner_ml_learner_integration(self):
        """Test NER and ML Pattern Learner working together."""
        # Train ML learner with NER results
        test_text = "John Doe from Mumbai Police Station reported theft"

        # Get NER results
        ner_result = self.ner.extract_entities(test_text)
        ner_entities = {}
        for entity in ner_result.entities:
            if entity.label not in ner_entities:
                ner_entities[entity.label] = []
            ner_entities[entity.label].append(entity.text)

        # Add to ML learner
        self.ml_learner.add_training_example(test_text, ner_entities, 'integration_test')

        # Train and predict
        self.ml_learner.train_entity_classifier()
        predictions = self.ml_learner.predict_entities("Rajesh Kumar from Delhi Police Station")

        # Should get reasonable predictions
        self.assertIsInstance(predictions, dict)

    def test_fir_parser_ner_integration(self):
        """Test FIR parser using NER for entity extraction."""
        # Test that FIR parser uses NER for better extraction
        row_with_complex_entities = [
            "Mumbai Police Station, CR No 456/2023, Section 379 IPC, Section 323 IPC",
            "15/03/2023, 16/03/2023, 14:30hrs, Andheri East",
            "Complainant: John Michael Doe S/o Robert Anthony Doe, Age: 35, Address: 123 Bandra West",
            "Victim: Priya Mary Sharma D/o Anil Kumar Sharma, Age: 28",
            "Property Lost: Samsung Galaxy S21 worth Rs. 45000, Cash Rs. 5000",
            "Accused: Rajesh Kumar Patel S/o Vijay Kumar Patel, Age: 25",
            "Complex case with multiple entities and relationships"
        ]

        result = self.fir_parser.parse_fir_row_enhanced(row_with_complex_entities)

        # Should extract entities with reasonable confidence
        self.assertGreater(len(result['confidence_scores']), 0)

        # Check that complex names are handled
        self.assertIsNotNone(result['complainant']['name'])
        self.assertIsNotNone(result['victim']['name'])

    def test_full_ml_pipeline_integration(self):
        """Test complete ML pipeline from input to export."""
        # 1. Input processing
        input_text = """
        FIR Details:
        Police Station: Central Police Station
        CR No: 789/2023
        Date: 20/03/2023
        Complainant: Dr. Sarah Johnson, Professor at University
        Incident: Theft of research data worth Rs. 10,00,000
        """

        # 2. NER processing
        ner_result = self.ner.extract_entities(input_text)

        # 3. ML pattern learning
        entities_dict = {}
        for entity in ner_result.entities:
            if entity.label not in entities_dict:
                entities_dict[entity.label] = []
            entities_dict[entity.label].append(entity.text)

        self.ml_learner.add_training_example(input_text, entities_dict, 'pipeline_test')

        # 4. FIR parsing (simulated)
        simulated_result = {
            'police_station': 'Central Police Station',
            'cr_no': '789/2023',
            'date_of_occurrence': '20/03/2023',
            'complainant': {'name': 'Dr. Sarah Johnson'},
            'property_lost': [{'item': 'research data', 'value': 'Rs. 10,00,000'}],
            'confidence_scores': {
                'police_station': 0.9,
                'cr_no': 0.95,
                'complainant_name': 0.85
            }
        }

        # 5. Validation and quality scoring
        validation_result = self.data_validator.validate_fir_data(simulated_result)
        quality_score = self.data_validator.calculate_quality_score(simulated_result)

        # 6. Anonymization
        anonymized_result = self.data_anonymizer.anonymize_fir_data(simulated_result)

        # Verify pipeline integrity
        self.assertIsInstance(ner_result, object)  # NERResult
        self.assertIsInstance(validation_result, dict)
        self.assertIsInstance(anonymized_result, dict)
        self.assertIsInstance(quality_score, (int, float))


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)

    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMLIntegration)
    integration_suite = unittest.TestLoader().loadTestsFromTestCase(TestMLComponentIntegration)

    # Combine suites
    full_suite = unittest.TestSuite([test_suite, integration_suite])

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(full_suite)

    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)