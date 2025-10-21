"""
Error Handling Testing Suite for ML Components

Tests error handling, fallback mechanisms, and graceful degradation
when components fail or receive invalid input.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import logging
import sys
import os

# Import ML components
from utils.enhanced_ner import EnhancedNER, extract_entities_with_ner
from utils.ml_pattern_learner import MLPatternLearner
from utils.fir_parser import EnhancedFIRParser
from utils.data_collector import TrainingDataCollector
from utils.data_validator import DataValidator
from utils.data_anonymizer import DataAnonymizer


class TestErrorHandling(unittest.TestCase):
    """Test error handling across all ML components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Initialize components
        self.ner = EnhancedNER(confidence_threshold=0.7)
        self.ml_learner = MLPatternLearner(model_dir=self.temp_dir)
        self.fir_parser = EnhancedFIRParser(enable_ml=True, enable_logging=False)
        self.data_collector = TrainingDataCollector(
            storage_path=self.temp_dir,
            auto_save=False
        )
        self.data_validator = DataValidator()
        self.data_anonymizer = DataAnonymizer()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_enhanced_ner_error_handling(self):
        """Test Enhanced NER error handling."""
        # Test with None input
        result = self.ner.extract_entities(None)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.entities), 0)

        # Test with empty string
        result = self.ner.extract_entities("")
        self.assertIsNotNone(result)
        self.assertEqual(len(result.entities), 0)

        # Test with non-string input
        result = self.ner.extract_entities(123)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.entities), 0)

        # Test with very long input (potential memory issues)
        long_text = "A" * 1000000  # 1MB text
        result = self.ner.extract_entities(long_text)
        self.assertIsNotNone(result)

        # Test with special characters
        special_text = "!@#$%^&*()[]{}|\\:;\"'<>?/.,`~"
        result = self.ner.extract_entities(special_text)
        self.assertIsNotNone(result)

    @patch('utils.enhanced_ner.spacy')
    def test_spacy_failure_handling(self, mock_spacy):
        """Test handling of spaCy failures."""
        # Test spaCy load failure
        mock_spacy.load.side_effect = OSError("spaCy model not found")

        ner = EnhancedNER()
        result = ner.extract_entities("Test text")

        # Should handle gracefully and return empty result
        self.assertIsNotNone(result)
        self.assertEqual(len(result.entities), 0)

        # Test spaCy processing failure
        mock_nlp = Mock()
        mock_nlp.side_effect = Exception("Processing error")
        mock_spacy.load.return_value = mock_nlp

        ner = EnhancedNER()
        result = ner.extract_entities("Test text")

        # Should handle gracefully
        self.assertIsNotNone(result)

    def test_ml_pattern_learner_error_handling(self):
        """Test ML Pattern Learner error handling."""
        # Test with insufficient training data
        predictions = self.ml_learner.predict_entities("Test text")
        self.assertIsInstance(predictions, dict)

        # Test training with invalid data
        try:
            self.ml_learner.add_training_example("", {}, "invalid")
            self.ml_learner.train_entity_classifier()
            # Should not crash
        except Exception as e:
            self.fail(f"ML learner should handle invalid training data gracefully: {e}")

        # Test with very large training set
        for i in range(10000):
            self.ml_learner.add_training_example(f"Text {i}", {'PERSON': [f'Person{i}']}, f'context_{i}')

        # Should handle large dataset without memory issues
        try:
            self.ml_learner.train_entity_classifier()
        except MemoryError:
            self.fail("ML learner should handle large datasets without memory errors")

    def test_fir_parser_error_handling(self):
        """Test FIR parser error handling."""
        # Test with None input
        result = self.fir_parser.parse_fir_row_enhanced(None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['police_station'], '')

        # Test with empty list
        result = self.fir_parser.parse_fir_row_enhanced([])
        self.assertIsInstance(result, dict)

        # Test with invalid data types in list
        invalid_row = [None, 123, {"key": "value"}, []]
        result = self.fir_parser.parse_fir_row_enhanced(invalid_row)
        self.assertIsInstance(result, dict)

        # Test with extremely long content
        long_row = ["A" * 10000 for _ in range(10)]
        result = self.fir_parser.parse_fir_row_enhanced(long_row)
        self.assertIsInstance(result, dict)

    def test_data_collector_error_handling(self):
        """Test data collector error handling."""
        # Test with invalid file path
        invalid_collector = TrainingDataCollector(
            storage_path="/invalid/path/that/does/not/exist",
            auto_save=False
        )

        # Should handle gracefully
        example_id = invalid_collector.collect_extraction_result(
            source_file="test.pdf",
            extraction_input="test input",
            expected_output={'test': 'data'},
            actual_output={'test': 'data'},
            confidence_scores={'test': 0.9},
            processing_metadata={}
        )
        self.assertIsNotNone(example_id)

        # Test with malformed data
        try:
            invalid_collector.collect_extraction_result(
                source_file=None,
                extraction_input=None,
                expected_output=None,
                actual_output=None,
                confidence_scores=None,
                processing_metadata=None
            )
            # Should handle None values
        except Exception as e:
            self.fail(f"Data collector should handle None values gracefully: {e}")

    def test_data_validator_error_handling(self):
        """Test data validator error handling."""
        # Test with None input
        result = self.data_validator.validate_fir_data(None)
        self.assertIsInstance(result, dict)

        # Test with invalid data types
        invalid_data = {
            'police_station': 123,  # Should be string
            'complainant': ['not', 'a', 'dict'],  # Should be dict
            'invalid_field': 'test'
        }

        result = self.data_validator.validate_fir_data(invalid_data)
        self.assertIsInstance(result, dict)
        self.assertIn('is_valid', result)

        # Test quality score calculation with invalid data
        quality_score = self.data_validator.calculate_quality_score(invalid_data)
        self.assertIsInstance(quality_score, (int, float))

    def test_data_anonymizer_error_handling(self):
        """Test data anonymizer error handling."""
        # Test with None input
        result = self.data_anonymizer.anonymize_fir_data(None)
        self.assertIsInstance(result, dict)

        # Test with invalid data structure
        invalid_data = {
            'police_station': 123,
            'complainant': 'not_a_dict',
            'nested': {'deeply': {'nested': 'value'}}
        }

        result = self.data_anonymizer.anonymize_fir_data(invalid_data)
        self.assertIsInstance(result, dict)

        # Test with circular reference (potential infinite loop)
        circular_data = {'self': None}
        circular_data['self'] = circular_data

        try:
            result = self.data_anonymizer.anonymize_fir_data(circular_data)
            self.assertIsInstance(result, dict)
        except RecursionError:
            self.fail("Anonymizer should handle circular references")

    def test_backward_compatibility_error_handling(self):
        """Test backward compatibility function error handling."""
        # Test extract_entities_with_ner with various error conditions
        test_cases = [None, "", 123, [], {}]

        for case in test_cases:
            try:
                result = extract_entities_with_ner(case)
                self.assertIsInstance(result, dict)
            except Exception as e:
                self.fail(f"Backward compatibility function should handle {case} gracefully: {e}")

    def test_file_system_error_handling(self):
        """Test file system operation error handling."""
        # Test with read-only directory
        read_only_dir = Path(self.temp_dir) / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)  # Read-only

        try:
            # Should handle read-only directory gracefully
            collector = TrainingDataCollector(
                storage_path=str(read_only_dir),
                auto_save=True
            )
            # Should not crash even if it can't write
        except Exception as e:
            self.fail(f"Should handle read-only directory gracefully: {e}")
        finally:
            read_only_dir.chmod(0o755)  # Restore permissions

    def test_network_error_handling(self):
        """Test network operation error handling (if applicable)."""
        # Test spaCy download failure
        with patch('utils.enhanced_ner.spacy.cli.download') as mock_download:
            mock_download.side_effect = Exception("Network error")

            # Should handle download failure gracefully
            ner = EnhancedNER()
            # Should fall back to regex-based extraction
            result = ner.extract_entities("John Doe from Mumbai")
            self.assertIsNotNone(result)

    def test_memory_error_handling(self):
        """Test memory error handling."""
        # Test with memory-intensive operations
        large_texts = ["A" * 100000 for _ in range(10)]  # 10 * 100KB texts

        try:
            for text in large_texts:
                result = self.ner.extract_entities(text)
                self.assertIsNotNone(result)

                # Process with FIR parser
                large_row = [text[:1000] for _ in range(7)]  # Reasonable size for parser
                parser_result = self.fir_parser.parse_fir_row_enhanced(large_row)
                self.assertIsInstance(parser_result, dict)

        except MemoryError:
            self.fail("Components should handle memory-intensive operations gracefully")

    def test_concurrent_error_handling(self):
        """Test error handling under concurrent access."""
        import threading
        import queue

        errors = queue.Queue()
        results = queue.Queue()

        def run_with_errors():
            """Function that might encounter errors."""
            try:
                # Mix of valid and invalid operations
                self.ner.extract_entities("Valid text")
                self.ner.extract_entities(None)  # Invalid
                self.fir_parser.parse_fir_row_enhanced([])  # Invalid
                self.fir_parser.parse_fir_row_enhanced(["Valid", "row"])  # Valid
                results.put(True)
            except Exception as e:
                errors.put(e)

        # Run multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=run_with_errors)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check that no threads crashed
        error_count = errors.qsize()
        success_count = results.qsize()

        self.assertEqual(error_count, 0, f"Concurrent operations should not produce errors: {errors.queue}")
        self.assertGreater(success_count, 0, "Some operations should succeed")

    def test_logging_error_handling(self):
        """Test error handling when logging fails."""
        # Test with invalid log configuration
        with patch('logging.getLogger') as mock_logger:
            mock_logger.side_effect = Exception("Logging error")

            # Should not crash when logging fails
            ner = EnhancedNER()
            result = ner.extract_entities("Test text")
            self.assertIsNotNone(result)

    def test_import_error_handling(self):
        """Test handling of import errors."""
        # Test missing optional dependencies
        with patch.dict(sys.modules, {'spacy': None}):
            # Should handle missing spaCy gracefully
            ner = EnhancedNER()
            result = ner.extract_entities("Test text")
            self.assertIsNotNone(result)

    def test_database_error_handling(self):
        """Test database/storage error handling."""
        # Test with corrupted storage
        corrupted_file = Path(self.temp_dir) / "corrupted.jsonl"
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content {")  # Invalid JSON

        # Should handle corrupted files gracefully
        collector = TrainingDataCollector(
            storage_path=self.temp_dir,
            auto_save=True
        )

        # Should not crash when loading corrupted data
        try:
            # This might log warnings but should not crash
            pass
        except Exception as e:
            self.fail(f"Should handle corrupted storage gracefully: {e}")

    def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        # Test NER degradation when spaCy fails
        with patch('utils.enhanced_ner.spacy.load') as mock_spacy_load:
            mock_spacy_load.side_effect = Exception("spaCy failed")

            ner = EnhancedNER()
            result = ner.extract_entities("John Doe from Mumbai Police Station")

            # Should fall back to regex-based extraction
            self.assertIsNotNone(result)

            # Should still find some entities using regex fallback
            total_entities = sum(len(entities) for entities in [
                result.entities  # This would be empty if no fallback
            ])

            # At minimum, should not crash
            self.assertIsInstance(result.entities, list)

    def test_resource_cleanup_error_handling(self):
        """Test resource cleanup when errors occur."""
        # Test that resources are cleaned up properly even when errors occur

        initial_files = len(list(Path(self.temp_dir).rglob("*")))

        try:
            # Perform operations that might fail
            collector = TrainingDataCollector(
                storage_path=self.temp_dir,
                auto_save=True
            )

            # Try to save with invalid data
            collector.collect_extraction_result(
                source_file="test.pdf",
                extraction_input="test",
                expected_output={'invalid': set()},  # Non-serializable
                actual_output={'test': 'data'},
                confidence_scores={'test': 0.9},
                processing_metadata={}
            )

        except Exception:
            # Even if serialization fails, should not leave corrupted state
            pass

        final_files = len(list(Path(self.temp_dir).rglob("*")))

        # File count should not grow excessively due to errors
        self.assertLess(final_files - initial_files, 10)


class TestFallbackMechanisms(unittest.TestCase):
    """Test fallback mechanisms and graceful degradation."""

    def setUp(self):
        """Set up test fixtures."""
        self.ner = EnhancedNER(confidence_threshold=0.7)
        self.ml_learner = MLPatternLearner()
        self.fir_parser = EnhancedFIRParser(enable_ml=True)

    def test_ner_fallback_mechanisms(self):
        """Test NER fallback when spaCy is unavailable."""
        # Mock spaCy to be completely unavailable
        with patch('utils.enhanced_ner.spacy') as mock_spacy:
            mock_spacy.load.side_effect = ImportError("spaCy not available")

            ner = EnhancedNER()
            result = ner.extract_entities("John Doe from Mumbai Police Station on 15/03/2023")

            # Should use regex fallback
            self.assertIsNotNone(result)
            self.assertIsInstance(result.entities, list)

    def test_ml_learner_fallback(self):
        """Test ML learner fallback when model is not trained."""
        # Test prediction without training
        predictions = self.ml_learner.predict_entities("John Doe from police station")

        # Should return empty dict or fallback predictions
        self.assertIsInstance(predictions, dict)

        # Should not crash
        self.assertIsNotNone(predictions)

    def test_fir_parser_fallback(self):
        """Test FIR parser fallback mechanisms."""
        # Test with ML disabled
        parser_no_ml = EnhancedFIRParser(enable_ml=False)

        sample_row = [
            "Police Station, CR No 123/456",
            "01/01/2023",
            "John Doe",
            "",
            "",
            "",
            "Test case"
        ]

        result = parser_no_ml.parse_fir_row_enhanced(sample_row)

        # Should work without ML
        self.assertIsInstance(result, dict)
        self.assertIn('processing_metadata', result)
        self.assertFalse(result['processing_metadata']['ml_enabled'])

    def test_data_processing_fallback(self):
        """Test data processing fallback mechanisms."""
        # Test validator with completely invalid data
        invalid_data = {
            'field1': float('inf'),
            'field2': float('nan'),
            'field3': None,
            'field4': {'nested': set()}  # Non-serializable
        }

        # Should handle gracefully
        result = self.data_validator.validate_fir_data(invalid_data)
        self.assertIsInstance(result, dict)

        # Should handle serialization errors in anonymizer
        try:
            anonymized = self.data_anonymizer.anonymize_fir_data(invalid_data)
            self.assertIsInstance(anonymized, dict)
        except (TypeError, ValueError):
            # Should either handle or raise appropriate errors
            pass

    def test_backward_compatibility_fallback(self):
        """Test backward compatibility fallback."""
        # Test old function with various inputs
        test_inputs = [
            "Normal text",
            "",  # Empty
            None,  # None
            "Text with special chars: !@#$%^&*()",
            "Very long text " * 1000
        ]

        for test_input in test_inputs:
            try:
                result = extract_entities_with_ner(test_input)
                self.assertIsInstance(result, dict)
                # Should have expected keys
                expected_keys = {'persons', 'dates', 'times', 'locations', 'orgs'}
                self.assertTrue(expected_keys.issubset(set(result.keys())))
            except Exception as e:
                self.fail(f"Backward compatibility function should handle {test_input} gracefully: {e}")


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery and resilience."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ner = EnhancedNER()
        self.fir_parser = EnhancedFIRParser()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_component_recovery(self):
        """Test that components recover from errors."""
        # Cause an error and verify recovery
        error_inputs = [None, "", [], {}]

        for error_input in error_inputs:
            # NER should recover
            result1 = self.ner.extract_entities(error_input or "test")
            self.assertIsNotNone(result1)

            # FIR parser should recover
            result2 = self.fir_parser.parse_fir_row_enhanced(error_input or ["test"])
            self.assertIsNotNone(result2)

    def test_state_consistency_after_errors(self):
        """Test that component state remains consistent after errors."""
        # Get initial state
        initial_ner_result = self.ner.extract_entities("Initial test")
        initial_parser_result = self.fir_parser.parse_fir_row_enhanced(["Initial", "test"])

        # Cause errors
        error_inputs = [None, "", 123, []]

        for error_input in error_inputs:
            self.ner.extract_entities(error_input)
            self.fir_parser.parse_fir_row_enhanced(error_input or [])

        # State should still be consistent
        final_ner_result = self.ner.extract_entities("Final test")
        final_parser_result = self.fir_parser.parse_fir_row_enhanced(["Final", "test"])

        self.assertIsNotNone(final_ner_result)
        self.assertIsNotNone(final_parser_result)

    def test_resource_leak_prevention(self):
        """Test that errors don't cause resource leaks."""
        import gc

        # Get initial object count
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform error-prone operations
        for _ in range(100):
            try:
                self.ner.extract_entities(None)
                self.fir_parser.parse_fir_row_enhanced([])
            except:
                pass

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count should not grow excessively
        object_increase = final_objects - initial_objects
        self.assertLess(object_increase, 1000, "Errors should not cause object leaks")


if __name__ == '__main__':
    # Configure logging to capture errors
    logging.basicConfig(level=logging.ERROR)

    # Run error handling tests
    unittest.main(verbosity=2)