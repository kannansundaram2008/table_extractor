"""
Performance Testing Suite for ML Components

Tests processing speed, memory usage, and scalability of all ML components
to ensure they meet the <5 second processing requirement and other performance criteria.
"""

import unittest
import time
import psutil
import os
import gc
from typing import List, Dict, Any
from unittest.mock import Mock
import tempfile
import shutil
from pathlib import Path

# Import ML components
from utils.enhanced_ner import EnhancedNER
from utils.ml_pattern_learner import MLPatternLearner
from utils.fir_parser import EnhancedFIRParser
from utils.data_collector import TrainingDataCollector
from utils.data_validator import DataValidator
from utils.data_anonymizer import DataAnonymizer


class PerformanceTestCase(unittest.TestCase):
    """Base class for performance tests with timing utilities."""

    def setUp(self):
        """Set up performance test environment."""
        self.process = psutil.Process(os.getpid())
        self.baselines = {}

    def measure_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    def measure_memory(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def measure_cpu_percent(self):
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()

    def assert_performance(self, execution_time, max_time, operation_name):
        """Assert that operation meets performance requirements."""
        self.assertLess(
            execution_time, max_time,
            f"{operation_name} took {execution_time:.3f}s, exceeding limit of {max_time:.3f}s"
        )


class TestMLPerformance(PerformanceTestCase):
    """Performance tests for individual ML components."""

    def setUp(self):
        """Set up ML components for performance testing."""
        super().setUp()
        self.ner = EnhancedNER(confidence_threshold=0.7)
        self.ml_learner = MLPatternLearner()
        self.fir_parser = EnhancedFIRParser(enable_ml=True)
        self.data_validator = DataValidator()
        self.data_anonymizer = DataAnonymizer()

        # Test data
        self.short_text = "John Doe reported theft at police station on 15/03/2023."
        self.long_text = "A" * 10000  # 10KB text

        # Sample FIR row for parsing tests
        self.sample_row = [
            "Mumbai Police Station, CR No 123/2023, Section 379 IPC",
            "15/03/2023, 16/03/2023, 14:30hrs, Andheri East",
            "John Doe S/o Robert Doe, Age 35, Address: Bandra West",
            "Priya Sharma D/o Anil Sharma, Age 28, Address: Andheri East",
            "Mobile Phone worth Rs. 15000",
            "Rajesh Kumar S/o Vijay Kumar, Age 25, Address: Jogeshwari",
            "Accused stole victim's mobile phone and injured her"
        ]

    def test_enhanced_ner_performance(self):
        """Test Enhanced NER performance requirements."""
        # Test with short text
        _, short_time = self.measure_time(self.ner.extract_entities, self.short_text)
        self.assert_performance(short_time, 1.0, "Short text NER")

        # Test with long text
        _, long_time = self.measure_time(self.ner.extract_entities, self.long_text)
        self.assert_performance(long_time, 2.0, "Long text NER")

        # Test memory usage
        initial_memory = self.measure_memory()
        for _ in range(10):
            self.ner.extract_entities(self.short_text)
        final_memory = self.measure_memory()

        memory_increase = final_memory - initial_memory
        self.assertLess(memory_increase, 50, "NER memory usage should be reasonable")

    def test_ml_pattern_learner_performance(self):
        """Test ML Pattern Learner performance."""
        # Add training data
        for i in range(50):  # Reasonable training set
            text = f"Training example {i} with person name and location"
            entities = {
                'PERSON': [f'Person{i}'],
                'GPE': [f'Location{i}']
            }
            self.ml_learner.add_training_example(text, entities, f'context_{i}')

        # Test training performance
        _, train_time = self.measure_time(self.ml_learner.train_entity_classifier)
        self.assert_performance(train_time, 10.0, "ML model training")

        # Test prediction performance
        _, predict_time = self.measure_time(
            self.ml_learner.predict_entities,
            "Test prediction with multiple entities"
        )
        self.assert_performance(predict_time, 1.0, "ML prediction")

    def test_fir_parser_performance(self):
        """Test FIR parser performance requirements."""
        # Test single row parsing
        _, parse_time = self.measure_time(
            self.fir_parser.parse_fir_row_enhanced,
            self.sample_row
        )
        self.assert_performance(parse_time, 2.0, "Single FIR row parsing")

        # Test batch processing (requirement: <5 seconds total)
        batch_size = 10
        batch_rows = [self.sample_row for _ in range(batch_size)]

        start_time = time.time()
        results = []
        for row in batch_rows:
            result = self.fir_parser.parse_fir_row_enhanced(row)
            results.append(result)
        batch_time = time.time() - start_time

        self.assert_performance(batch_time, 5.0, f"Batch processing {batch_size} rows")
        self.assertEqual(len(results), batch_size)

    def test_data_processing_performance(self):
        """Test data processing component performance."""
        # Test data validation performance
        test_data = {
            'police_station': 'Mumbai Police Station',
            'cr_no': '123/2023',
            'complainant': {'name': 'John Doe', 'age': '35'},
            'victims': [{'name': 'Jane Doe', 'age': '30'}],
            'accused': [{'name': 'Richard Roe', 'age': '40'}]
        }

        _, validation_time = self.measure_time(
            self.data_validator.validate_fir_data,
            test_data
        )
        self.assert_performance(validation_time, 0.5, "Data validation")

        # Test data anonymization performance
        _, anonymization_time = self.measure_time(
            self.data_anonymizer.anonymize_fir_data,
            test_data
        )
        self.assert_performance(anonymization_time, 0.5, "Data anonymization")

    def test_end_to_end_performance(self):
        """Test complete end-to-end pipeline performance."""
        # This test ensures the entire pipeline meets <5 second requirement
        start_time = time.time()

        # Complete pipeline
        parsed = self.fir_parser.parse_fir_row_enhanced(self.sample_row)
        validated = self.data_validator.validate_fir_data(parsed)
        anonymized = self.data_anonymizer.anonymize_fir_data(parsed)

        total_time = time.time() - start_time

        # Must complete within 5 seconds as per requirements
        self.assert_performance(total_time, 5.0, "End-to-end pipeline")

        # Verify all steps completed successfully
        self.assertIsNotNone(parsed)
        self.assertIsNotNone(validated)
        self.assertIsNotNone(anonymized)

    def test_scalability_performance(self):
        """Test performance with increasing data sizes."""
        # Test with different text lengths
        text_sizes = [100, 1000, 10000, 50000]

        for size in text_sizes:
            test_text = "A" * size

            _, ner_time = self.measure_time(self.ner.extract_entities, test_text)

            # Performance should degrade gracefully, not exponentially
            if size <= 1000:
                self.assert_performance(ner_time, 1.0, f"NER with {size} chars")
            elif size <= 10000:
                self.assert_performance(ner_time, 2.0, f"NER with {size} chars")
            else:
                self.assert_performance(ner_time, 5.0, f"NER with {size} chars")

    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        import threading
        import queue

        def run_ner_test():
            """Function to run in each thread."""
            for _ in range(5):
                self.ner.extract_entities(self.short_text)

        # Start multiple threads
        threads = []
        num_threads = 5

        start_time = time.time()

        for _ in range(num_threads):
            thread = threading.Thread(target=run_ner_test)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        concurrent_time = time.time() - start_time

        # Concurrent execution should still be reasonable
        self.assert_performance(concurrent_time, 10.0, f"Concurrent execution with {num_threads} threads")

    def test_memory_efficiency(self):
        """Test memory efficiency of ML components."""
        # Measure memory before operations
        gc.collect()
        initial_memory = self.measure_memory()

        # Perform intensive operations
        for _ in range(100):
            # NER processing
            self.ner.extract_entities(self.short_text)

            # FIR parsing
            self.fir_parser.parse_fir_row_enhanced(self.sample_row)

            # Data validation
            test_data = {'police_station': 'Test', 'cr_no': '123/456'}
            self.data_validator.validate_fir_data(test_data)

        # Force garbage collection
        gc.collect()
        final_memory = self.measure_memory()

        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB for intensive operations)
        self.assertLess(memory_increase, 100, "Memory usage should remain reasonable")

    def test_cpu_efficiency(self):
        """Test CPU efficiency during processing."""
        # Monitor CPU usage during intensive operations
        cpu_before = self.measure_cpu_percent()
        time.sleep(0.1)  # Allow CPU measurement to stabilize

        start_time = time.time()

        # Intensive processing
        for _ in range(50):
            self.ner.extract_entities(self.long_text)
            self.fir_parser.parse_fir_row_enhanced(self.sample_row)

        processing_time = time.time() - start_time
        cpu_after = self.measure_cpu_percent()

        # CPU usage should be reasonable
        avg_cpu = (cpu_before + cpu_after) / 2
        self.assertLess(avg_cpu, 80, "Average CPU usage should be reasonable")

        # Processing should complete in reasonable time
        self.assert_performance(processing_time, 15.0, "CPU-intensive processing")


class TestPerformanceBenchmarks(PerformanceTestCase):
    """Benchmark tests for performance regression detection."""

    def setUp(self):
        """Set up for benchmark testing."""
        super().setUp()
        self.benchmark_results = {}

    def record_baseline(self, operation_name: str, execution_time: float):
        """Record baseline performance for regression testing."""
        self.benchmark_results[operation_name] = execution_time

    def test_ner_benchmark(self):
        """Benchmark NER performance."""
        test_cases = [
            ("short_text", "John Doe reported incident."),
            ("medium_text", "A" * 1000),
            ("long_text", "A" * 10000)
        ]

        ner = EnhancedNER()

        for case_name, text in test_cases:
            _, execution_time = self.measure_time(ner.extract_entities, text)
            self.record_baseline(f"ner_{case_name}", execution_time)

            # Set performance expectations based on text length
            if case_name == "short_text":
                self.assert_performance(execution_time, 0.5, f"NER {case_name}")
            elif case_name == "medium_text":
                self.assert_performance(execution_time, 1.0, f"NER {case_name}")
            else:
                self.assert_performance(execution_time, 2.0, f"NER {case_name}")

    def test_fir_parser_benchmark(self):
        """Benchmark FIR parser performance."""
        parser = EnhancedFIRParser()

        # Test with different complexity levels
        simple_row = [
            "Police Station, CR No 123/456",
            "01/01/2023",
            "John Doe",
            "",
            "",
            "",
            "Simple case"
        ]

        complex_row = [
            "Mumbai Central Police Station, CR No 123/2023, Section 379 IPC, Section 323 IPC",
            "15/03/2023, 16/03/2023, 14:30hrs, 15:45hrs, Andheri East, Mumbai",
            "John Michael Doe S/o Robert Anthony Doe, Age 35, Address: 123 Bandra West, Mumbai",
            "Priya Mary Sharma D/o Anil Kumar Sharma, Age 28, Address: 456 Andheri East",
            "Property Lost: Samsung Galaxy S21 worth Rs. 45000, Cash Rs. 5000, Gold Chain worth Rs. 75000",
            "Rajesh Kumar Patel S/o Vijay Kumar Patel, Age 25, Address: Jogeshwari, Mumbai",
            "Complex case with multiple victims, accused, and detailed property information"
        ]

        # Benchmark simple parsing
        _, simple_time = self.measure_time(parser.parse_fir_row_enhanced, simple_row)
        self.record_baseline("fir_parser_simple", simple_time)
        self.assert_performance(simple_time, 1.0, "Simple FIR parsing")

        # Benchmark complex parsing
        _, complex_time = self.measure_time(parser.parse_fir_row_enhanced, complex_row)
        self.record_baseline("fir_parser_complex", complex_time)
        self.assert_performance(complex_time, 3.0, "Complex FIR parsing")

    def test_ml_training_benchmark(self):
        """Benchmark ML training performance."""
        ml_learner = MLPatternLearner()

        # Add training data
        for i in range(100):
            text = f"Training example {i} with entities for benchmarking"
            entities = {
                'PERSON': [f'Person{i}'],
                'ORG': [f'Organization{i}'],
                'GPE': [f'Location{i}']
            }
            ml_learner.add_training_example(text, entities, f'benchmark_{i}')

        # Benchmark training
        _, training_time = self.measure_time(ml_learner.train_entity_classifier)
        self.record_baseline("ml_training", training_time)
        self.assert_performance(training_time, 15.0, "ML model training")

        # Benchmark prediction
        _, prediction_time = self.measure_time(
            ml_learner.predict_entities,
            "Benchmark prediction test with multiple entities"
        )
        self.record_baseline("ml_prediction", prediction_time)
        self.assert_performance(prediction_time, 1.0, "ML prediction")

    def test_data_processing_benchmark(self):
        """Benchmark data processing performance."""
        validator = DataValidator()
        anonymizer = DataAnonymizer()

        # Test data of varying complexity
        simple_data = {
            'police_station': 'Test Station',
            'cr_no': '123/456'
        }

        complex_data = {
            'police_station': 'Mumbai Central Police Station',
            'cr_no': '123/2023',
            'complainant': {
                'name': 'John Michael Doe',
                'age': '35',
                'address': '123 Main Street, Bandra West, Mumbai',
                'phone': '9876543210',
                'aadhar': '1234-5678-9012'
            },
            'victims': [
                {'name': 'Jane Mary Smith', 'age': '30', 'address': '456 Oak Avenue'},
                {'name': 'Bob Wilson', 'age': '45', 'address': '789 Pine Road'}
            ],
            'accused': [
                {'name': 'Richard Brown', 'age': '40', 'address': '321 Elm Street'}
            ],
            'property_lost': [
                {'item': 'Laptop', 'value': 'Rs. 50000'},
                {'item': 'Mobile Phone', 'value': 'Rs. 15000'}
            ]
        }

        # Benchmark validation
        _, simple_validation_time = self.measure_time(validator.validate_fir_data, simple_data)
        _, complex_validation_time = self.measure_time(validator.validate_fir_data, complex_data)

        self.record_baseline("validation_simple", simple_validation_time)
        self.record_baseline("validation_complex", complex_validation_time)

        self.assert_performance(simple_validation_time, 0.1, "Simple data validation")
        self.assert_performance(complex_validation_time, 0.5, "Complex data validation")

        # Benchmark anonymization
        _, simple_anonymization_time = self.measure_time(anonymizer.anonymize_fir_data, simple_data)
        _, complex_anonymization_time = self.measure_time(anonymizer.anonymize_fir_data, complex_data)

        self.record_baseline("anonymization_simple", simple_anonymization_time)
        self.record_baseline("anonymization_complex", complex_anonymization_time)

        self.assert_performance(simple_anonymization_time, 0.1, "Simple data anonymization")
        self.assert_performance(complex_anonymization_time, 0.5, "Complex data anonymization")

    def test_throughput_benchmark(self):
        """Test throughput under load."""
        # Test NER throughput
        ner = EnhancedNER()
        test_texts = [f"Test document {i} with entities" for i in range(100)]

        start_time = time.time()
        results = []
        for text in test_texts:
            result = ner.extract_entities(text)
            results.append(result)
        throughput_time = time.time() - start_time

        # Calculate throughput (documents per second)
        throughput = len(test_texts) / throughput_time

        self.record_baseline("ner_throughput", throughput)

        # Should process at least 10 documents per second
        self.assertGreater(throughput, 10, f"NER throughput {throughput:.2f} docs/sec is below requirement")

        # Test FIR parser throughput
        parser = EnhancedFIRParser()
        test_rows = [self.sample_row for _ in range(50)]

        start_time = time.time()
        parsed_results = []
        for row in test_rows:
            result = parser.parse_fir_row_enhanced(row)
            parsed_results.append(result)
        parser_throughput_time = time.time() - start_time

        parser_throughput = len(test_rows) / parser_throughput_time
        self.record_baseline("fir_parser_throughput", parser_throughput)

        # Should process at least 5 FIR rows per second
        self.assertGreater(parser_throughput, 5, f"FIR parser throughput {parser_throughput:.2f} rows/sec is below requirement")

    def generate_performance_report(self):
        """Generate a performance report for CI/CD integration."""
        report = {
            'timestamp': time.time(),
            'system_info': {
                'cpu_count': os.cpu_count(),
                'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
            },
            'benchmarks': self.benchmark_results,
            'performance_targets': {
                'ner_short_text': 0.5,  # seconds
                'ner_long_text': 2.0,   # seconds
                'fir_parsing': 2.0,     # seconds
                'ml_training': 15.0,    # seconds
                'end_to_end': 5.0,      # seconds
                'ner_throughput': 10,   # docs/second
                'parser_throughput': 5  # rows/second
            }
        }

        # Check if targets are met
        targets_met = True
        for benchmark_name, target in report['performance_targets'].items():
            if benchmark_name in self.benchmark_results:
                actual = self.benchmark_results[benchmark_name]
                if isinstance(target, (int, float)) and actual > target:
                    targets_met = False
                    break

        report['targets_met'] = targets_met
        return report


class TestStressPerformance(PerformanceTestCase):
    """Stress tests for extreme conditions."""

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large training dataset
        ml_learner = MLPatternLearner()

        large_training_set = []
        for i in range(1000):  # Large but reasonable dataset
            text = f"Large dataset training example {i} with comprehensive entity information"
            entities = {
                'PERSON': [f'Person{i}'],
                'ORG': [f'Organization{i}'],
                'GPE': [f'Location{i}'],
                'DATE': [f'2023-{i%12+1:02d}-{i%28+1:02d}']
            }
            large_training_set.append((text, entities, f'context_{i}'))

        # Add all training examples
        start_time = time.time()
        for text, entities, context in large_training_set:
            ml_learner.add_training_example(text, entities, context)
        add_time = time.time() - start_time

        self.assert_performance(add_time, 10.0, "Adding large training set")

        # Train on large dataset
        _, train_time = self.measure_time(ml_learner.train_entity_classifier)
        self.assert_performance(train_time, 30.0, "Training on large dataset")

    def test_memory_stress_test(self):
        """Test memory usage under stress."""
        gc.collect()
        initial_memory = self.measure_memory()

        # Create and process many documents
        ner = EnhancedNER()
        documents = []

        for i in range(200):  # Process many documents
            doc_text = f"Stress test document {i} with repeated content " * 10
            result = ner.extract_entities(doc_text)
            documents.append(result)

        # Force garbage collection
        del documents
        gc.collect()

        final_memory = self.measure_memory()
        memory_increase = final_memory - initial_memory

        # Memory increase should still be reasonable
        self.assertLess(memory_increase, 200, "Memory usage should remain manageable under stress")

    def test_long_running_performance(self):
        """Test performance over extended periods."""
        ner = EnhancedNER()
        parser = EnhancedFIRParser()

        start_time = time.time()

        # Simulate long-running processing
        for hour in range(24):  # Simulate 24 hours of processing
            for minute in range(60):  # Process every minute
                # Mix of operations
                ner.extract_entities(f"Document {hour}:{minute}")
                if minute % 10 == 0:  # Every 10 minutes, do intensive processing
                    parser.parse_fir_row_enhanced(self.sample_row)

        total_time = time.time() - start_time

        # Should complete within reasonable time (much less than actual 24 hours)
        self.assertLess(total_time, 60, "Long-running test should complete quickly in test environment")


if __name__ == '__main__':
    # Run performance tests
    unittest.main(verbosity=2)