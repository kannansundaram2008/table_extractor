"""
Data Flow Testing Suite for ML Components

Tests data flow between all ML components to ensure proper integration
and data transformation across the entire pipeline.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock

# Import ML components
from utils.enhanced_ner import EnhancedNER
from utils.ml_pattern_learner import MLPatternLearner
from utils.fir_parser import EnhancedFIRParser
from utils.data_collector import TrainingDataCollector
from utils.data_validator import DataValidator
from utils.data_exporter import DataExporter
from utils.data_anonymizer import DataAnonymizer
from utils.annotation_interface import AnnotationInterface


class DataFlowTestCase(unittest.TestCase):
    """Base class for data flow testing."""

    def assert_data_integrity(self, data: Dict[str, Any], required_fields: List[str]):
        """Assert that data contains required fields."""
        for field in required_fields:
            self.assertIn(field, data, f"Required field '{field}' missing from data")

    def assert_data_type(self, data: Dict[str, Any], field: str, expected_type: type):
        """Assert that field has correct data type."""
        if field in data:
            self.assertIsInstance(data[field], expected_type,
                                f"Field '{field}' should be {expected_type.__name__}, got {type(data[field]).__name__}")


class TestComponentDataFlow(DataFlowTestCase):
    """Test data flow between individual components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ner = EnhancedNER(confidence_threshold=0.7)
        self.ml_learner = MLPatternLearner(model_dir=self.temp_dir)
        self.fir_parser = EnhancedFIRParser(enable_ml=True, enable_logging=False)
        self.data_collector = TrainingDataCollector(
            storage_path=self.temp_dir,
            auto_save=False
        )
        self.data_validator = DataValidator()
        self.data_exporter = DataExporter()
        self.data_anonymizer = DataAnonymizer()

        # Sample data for testing
        self.sample_text = """
        On 15/03/2023 at 14:30hrs in Mumbai Police Station, complainant John Doe S/o Robert Doe
        aged 35 years residing at Bandra West reported that accused Rajesh Kumar aged 25 years
        at Andheri East stole mobile phone worth Rs. 15000. Victim Priya Sharma D/o Anil Sharma
        was injured. Section 379 IPC, Section 323 IPC applies. CR No 123/2023.
        """

        self.sample_fir_row = [
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
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ner_to_ml_learner_flow(self):
        """Test data flow from NER to ML Pattern Learner."""
        # 1. NER extracts entities
        ner_result = self.ner.extract_entities(self.sample_text)

        # Verify NER output structure
        self.assertIsNotNone(ner_result)
        self.assertTrue(hasattr(ner_result, 'entities'))
        self.assertTrue(hasattr(ner_result, 'confidence_scores'))
        self.assertIsInstance(ner_result.entities, list)

        # 2. Convert NER result to ML learner format
        entities_dict = {}
        for entity in ner_result.entities:
            if entity.label not in entities_dict:
                entities_dict[entity.label] = []
            entities_dict[entity.label].append(entity.text)

        # 3. Add to ML learner
        self.ml_learner.add_training_example(
            self.sample_text,
            entities_dict,
            'flow_test_1'
        )

        # 4. Train ML learner
        self.ml_learner.train_entity_classifier()

        # 5. Test prediction (data flows back)
        predictions = self.ml_learner.predict_entities("Rajesh Kumar from police station")

        # Verify prediction output
        self.assertIsInstance(predictions, dict)

        # Data should flow correctly through the pipeline
        self.assertGreater(len(ner_result.entities), 0)

    def test_fir_parser_data_flow(self):
        """Test data flow through FIR parser components."""
        # Test enhanced FIR parsing
        result = self.fir_parser.parse_fir_row_enhanced(self.sample_fir_row)

        # Verify output structure
        required_fields = [
            'police_station', 'cr_no', 'section_of_law', 'date_of_occurrence',
            'complainant', 'victims', 'accused', 'confidence_scores',
            'processing_metadata'
        ]

        self.assert_data_integrity(result, required_fields)

        # Verify data types
        self.assert_data_type(result, 'police_station', str)
        self.assert_data_type(result, 'cr_no', str)
        self.assert_data_type(result, 'confidence_scores', dict)
        self.assert_data_type(result, 'processing_metadata', dict)

        # Verify nested structures
        self.assert_data_type(result, 'complainant', dict)
        if result['complainant']:
            self.assertIn('name', result['complainant'])

    def test_training_data_collection_flow(self):
        """Test data flow through training data collection."""
        # 1. Parse FIR data
        parsed_data = self.fir_parser.parse_fir_row_enhanced(self.sample_fir_row)

        # 2. Validate data
        validation_result = self.data_validator.validate_fir_data(parsed_data)

        # 3. Collect for training
        example_id = self.data_collector.collect_extraction_result(
            source_file="test_fir.pdf",
            extraction_input=self.sample_text,
            expected_output=parsed_data,  # In real scenario, this would be ground truth
            actual_output=parsed_data,
            confidence_scores=parsed_data['confidence_scores'],
            processing_metadata=parsed_data['processing_metadata']
        )

        # Verify collection
        self.assertIsNotNone(example_id)
        self.assertGreater(len(self.data_collector.examples), 0)

        # 4. Export data
        export_path = Path(self.temp_dir) / "export_test.json"
        exported_file = self.data_collector.export_for_training(
            output_path=str(export_path),
            format="json"
        )

        # Verify export
        self.assertTrue(Path(exported_file).exists())

        # 5. Verify exported data structure
        with open(exported_file, 'r') as f:
            exported_data = json.load(f)

        self.assertIsInstance(exported_data, list)
        if exported_data:
            self.assert_data_integrity(exported_data[0], ['input', 'expected_output', 'metadata'])

    def test_anonymization_flow(self):
        """Test data flow through anonymization."""
        # 1. Start with sensitive data
        sensitive_data = {
            'police_station': 'Mumbai Police Station',
            'complainant': {
                'name': 'John Doe',
                'phone': '9876543210',
                'aadhar': '1234-5678-9012',
                'address': '123 Main St, Mumbai'
            },
            'victim': {
                'name': 'Jane Smith',
                'phone': '9123456789'
            }
        }

        # 2. Anonymize data
        anonymized_data = self.data_anonymizer.anonymize_fir_data(sensitive_data)

        # 3. Verify anonymization
        self.assertIsInstance(anonymized_data, dict)
        self.assert_data_integrity(anonymized_data, ['police_station', 'complainant', 'victim'])

        # 4. Verify sensitive fields are anonymized
        self.assertNotEqual(anonymized_data['complainant']['phone'], '9876543210')
        self.assertNotEqual(anonymized_data['complainant']['aadhar'], '1234-5678-9012')
        self.assertNotEqual(anonymized_data['complainant']['name'], 'John Doe')
        self.assertNotEqual(anonymized_data['victim']['name'], 'Jane Smith')

        # 5. Verify non-sensitive fields remain unchanged
        self.assertEqual(anonymized_data['police_station'], 'Mumbai Police Station')

    def test_validation_flow(self):
        """Test data flow through validation."""
        # Test with valid data
        valid_data = {
            'police_station': 'Mumbai Police Station',
            'cr_no': '123/2023',
            'complainant': {'name': 'John Doe'},
            'section_of_law': 'IPC 379'
        }

        validation_result = self.data_validator.validate_fir_data(valid_data)

        # Verify validation output
        self.assertIsInstance(validation_result, dict)
        self.assertIn('is_valid', validation_result)
        self.assertIn('errors', validation_result)
        self.assertIn('warnings', validation_result)

        # Test quality scoring
        quality_score = self.data_validator.calculate_quality_score(valid_data)
        self.assertIsInstance(quality_score, (int, float))
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)

    def test_export_flow(self):
        """Test data flow through export functionality."""
        # 1. Collect test data
        test_data = [
            {
                'police_station': 'Station 1',
                'cr_no': '001/2023',
                'complainant': {'name': 'Person 1'}
            },
            {
                'police_station': 'Station 2',
                'cr_no': '002/2023',
                'complainant': {'name': 'Person 2'}
            }
        ]

        # 2. Export in different formats
        formats = ['json', 'jsonl', 'csv']
        exported_files = []

        for fmt in formats:
            export_path = Path(self.temp_dir) / f"test_export.{fmt}"
            exported_file = self.data_exporter.export_fir_data(
                test_data,
                str(export_path)
            )
            exported_files.append(exported_file)

        # 3. Verify all exports succeeded
        for exported_file in exported_files:
            self.assertTrue(Path(exported_file).exists())

            # Verify file is not empty
            file_size = Path(exported_file).stat().st_size
            self.assertGreater(file_size, 0)

        # 4. Verify data integrity in exported files
        json_file = Path(self.temp_dir) / "test_export.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                imported_data = json.load(f)

            self.assertEqual(len(imported_data), len(test_data))

            # Verify structure is preserved
            for item in imported_data:
                self.assert_data_integrity(item, ['police_station', 'cr_no', 'complainant'])


class TestEndToEndDataFlow(DataFlowTestCase):
    """Test complete end-to-end data flow."""

    def setUp(self):
        """Set up complete pipeline."""
        self.temp_dir = tempfile.mkdtemp()
        self.ner = EnhancedNER(confidence_threshold=0.7)
        self.ml_learner = MLPatternLearner(model_dir=self.temp_dir)
        self.fir_parser = EnhancedFIRParser(enable_ml=True, enable_logging=False)
        self.data_collector = TrainingDataCollector(
            storage_path=self.temp_dir,
            auto_save=False
        )
        self.data_validator = DataValidator()
        self.data_anonymizer = DataAnonymizer()
        self.data_exporter = DataExporter()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_pipeline_flow(self):
        """Test complete data flow through entire pipeline."""
        # Input data
        input_text = """
        FIR Details:
        Police Station: Central Police Station, Mumbai
        CR No: 456/2023
        Date: 20/03/2023
        Complainant: Dr. Sarah Johnson, Phone: 9876543210, Aadhar: 1234-5678-9012
        Incident: Theft of laptop worth Rs. 75000
        """

        fir_row = [
            "Central Police Station, Mumbai, CR No 456/2023, Section 379 IPC",
            "20/03/2023, 21/03/2023, 10:00hrs, Mumbai",
            "Dr. Sarah Johnson, Age 40, Address: University Campus",
            "",
            "Laptop worth Rs. 75000",
            "",
            "Theft of laptop from office"
        ]

        # 1. FIR Parsing with ML integration
        parsed_result = self.fir_parser.parse_fir_row_enhanced(fir_row)

        # Verify parsing output
        self.assert_data_integrity(parsed_result, [
            'police_station', 'cr_no', 'confidence_scores', 'processing_metadata'
        ])

        # 2. Data Validation
        validation_result = self.data_validator.validate_fir_data(parsed_result)

        # Verify validation output
        self.assertIsInstance(validation_result, dict)
        self.assertIn('is_valid', validation_result)

        # 3. Data Anonymization
        anonymized_result = self.data_anonymizer.anonymize_fir_data(parsed_result)

        # Verify anonymization
        self.assertIsInstance(anonymized_result, dict)

        # Sensitive fields should be anonymized
        if 'complainant' in anonymized_result and 'name' in anonymized_result['complainant']:
            self.assertNotEqual(anonymized_result['complainant']['name'], 'Dr. Sarah Johnson')

        # 4. Training Data Collection
        example_id = self.data_collector.collect_extraction_result(
            source_file="test_pipeline.pdf",
            extraction_input=input_text,
            expected_output=parsed_result,  # Ground truth would come from manual annotation
            actual_output=anonymized_result,
            confidence_scores=parsed_result['confidence_scores'],
            processing_metadata=parsed_result['processing_metadata']
        )

        # Verify collection
        self.assertIsNotNone(example_id)

        # 5. Data Export
        export_path = Path(self.temp_dir) / "pipeline_output.json"
        exported_file = self.data_collector.export_for_training(
            output_path=str(export_path),
            format="json"
        )

        # Verify export
        self.assertTrue(Path(exported_file).exists())

        # 6. Verify complete data transformation
        with open(exported_file, 'r') as f:
            final_data = json.load(f)

        self.assertIsInstance(final_data, list)
        if final_data:
            # Data should have been transformed through the entire pipeline
            self.assert_data_integrity(final_data[0], ['input', 'expected_output', 'actual_output', 'metadata'])

    def test_data_transformation_integrity(self):
        """Test that data transformations preserve integrity."""
        # Original data
        original_data = {
            'police_station': 'Test Police Station',
            'cr_no': '123/456',
            'complainant': {
                'name': 'John Doe',
                'phone': '9876543210',
                'address': '123 Test St'
            },
            'section_of_law': 'IPC 379'
        }

        # Transform through pipeline
        validation_result = self.data_validator.validate_fir_data(original_data)
        anonymized_data = self.data_anonymizer.anonymize_fir_data(original_data)

        # Verify key fields are preserved (non-sensitive ones)
        self.assertEqual(anonymized_data['police_station'], 'Test Police Station')
        self.assertEqual(anonymized_data['cr_no'], '123/456')
        self.assertEqual(anonymized_data['section_of_law'], 'IPC 379')

        # Verify structure is maintained
        self.assertIn('complainant', anonymized_data)
        self.assertIn('name', anonymized_data['complainant'])
        self.assertIn('phone', anonymized_data['complainant'])
        self.assertIn('address', anonymized_data['complainant'])

    def test_error_propagation_flow(self):
        """Test how errors propagate through the data flow."""
        # Test with invalid data
        invalid_data = {
            'police_station': None,
            'complainant': 'not_a_dict',
            'invalid_field': float('inf')
        }

        # 1. Validation should catch errors
        validation_result = self.data_validator.validate_fir_data(invalid_data)

        # 2. Anonymization should handle errors gracefully
        anonymized_data = self.data_anonymizer.anonymize_fir_data(invalid_data)

        # 3. Collection should handle problematic data
        try:
            example_id = self.data_collector.collect_extraction_result(
                source_file="test.pdf",
                extraction_input="test",
                expected_output=invalid_data,
                actual_output=anonymized_data,
                confidence_scores={},
                processing_metadata={}
            )
            self.assertIsNotNone(example_id)
        except Exception as e:
            self.fail(f"Data collection should handle invalid data gracefully: {e}")

    def test_batch_processing_flow(self):
        """Test data flow with batch processing."""
        # Create batch of FIR rows
        batch_rows = [
            [
                "Police Station 1, CR No 001/2023, Section 379 IPC",
                "01/01/2023, Station Area 1",
                "Person 1, Age 30",
                "", "", "", "", "Case 1"
            ],
            [
                "Police Station 2, CR No 002/2023, Section 323 IPC",
                "02/01/2023, Station Area 2",
                "Person 2, Age 35",
                "", "", "", "", "Case 2"
            ],
            [
                "Police Station 3, CR No 003/2023, Section 420 IPC",
                "03/01/2023, Station Area 3",
                "Person 3, Age 40",
                "", "", "", "", "Case 3"
            ]
        ]

        # Process batch
        processed_results = []
        for i, row in enumerate(batch_rows):
            result = self.fir_parser.parse_fir_row_enhanced(row)

            # Collect each result
            example_id = self.data_collector.collect_extraction_result(
                source_file=f"batch_test_{i}.pdf",
                extraction_input=f"Input text for case {i}",
                expected_output=result,
                actual_output=result,
                confidence_scores=result['confidence_scores'],
                processing_metadata=result['processing_metadata']
            )

            processed_results.append((result, example_id))

        # Verify batch processing
        self.assertEqual(len(processed_results), len(batch_rows))

        for result, example_id in processed_results:
            self.assertIsNotNone(example_id)
            self.assertIsInstance(result, dict)

        # Export batch
        export_path = Path(self.temp_dir) / "batch_export.json"
        exported_file = self.data_collector.export_for_training(
            output_path=str(export_path),
            format="json"
        )

        self.assertTrue(Path(exported_file).exists())

        # Verify batch export
        with open(exported_file, 'r') as f:
            batch_export_data = json.load(f)

        self.assertEqual(len(batch_export_data), len(batch_rows))


class TestDataConsistencyFlow(DataFlowTestCase):
    """Test data consistency throughout the flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ner = EnhancedNER()
        self.fir_parser = EnhancedFIRParser()
        self.data_validator = DataValidator()
        self.data_anonymizer = DataAnonymizer()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_field_consistency(self):
        """Test that field names remain consistent throughout flow."""
        # Define expected field names
        expected_fields = {
            'police_station', 'cr_no', 'section_of_law', 'date_of_occurrence',
            'date_of_report', 'place_of_occurrence', 'complainant', 'victims',
            'accused', 'property_lost', 'property_recovered', 'property_seized', 'gist'
        }

        # Process through pipeline
        fir_row = [
            "Test Police Station, CR No 123/456, Section 379 IPC",
            "01/01/2023, Test Location",
            "John Doe, Age 30",
            "", "", "", "", "Test case"
        ]

        parsed = self.fir_parser.parse_fir_row_enhanced(fir_row)
        validated = self.data_validator.validate_fir_data(parsed)
        anonymized = self.data_anonymizer.anonymize_fir_data(parsed)

        # Check field consistency
        for data in [parsed, anonymized]:
            self.assertTrue(
                expected_fields.issubset(set(data.keys())),
                f"Missing fields in {list(expected_fields - set(data.keys()))}"
            )

    def test_nested_data_consistency(self):
        """Test consistency of nested data structures."""
        # Test complainant structure
        complainant_data = {
            'name': 'John Doe',
            'age': '30',
            'address': '123 Test St',
            'sex': 'Male'
        }

        # Process through validation and anonymization
        test_record = {'complainant': complainant_data}
        validated = self.data_validator.validate_fir_data(test_record)
        anonymized = self.data_anonymizer.anonymize_fir_data(test_record)

        # Verify nested structure consistency
        for data in [anonymized]:
            if 'complainant' in data:
                complainant = data['complainant']
                self.assertIsInstance(complainant, dict)
                self.assertIn('name', complainant)
                self.assertIn('age', complainant)

    def test_confidence_score_flow(self):
        """Test that confidence scores flow correctly through pipeline."""
        # Parse with confidence scores
        fir_row = [
            "Test Police Station, CR No 123/456",
            "01/01/2023",
            "John Doe",
            "", "", "", "", "Test"
        ]

        parsed = self.fir_parser.parse_fir_row_enhanced(fir_row)

        # Verify confidence scores exist
        self.assertIn('confidence_scores', parsed)
        self.assertIsInstance(parsed['confidence_scores'], dict)

        # Verify confidence scores are reasonable (0-1 range)
        for field, score in parsed['confidence_scores'].items():
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestIntegrationDataFlow(DataFlowTestCase):
    """Test data flow in integrated scenarios."""

    def setUp(self):
        """Set up integrated test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.ner = EnhancedNER()
        self.ml_learner = MLPatternLearner(model_dir=self.temp_dir)
        self.fir_parser = EnhancedFIRParser(enable_ml=True, enable_logging=False)
        self.data_collector = TrainingDataCollector(
            storage_path=self.temp_dir,
            auto_save=False
        )
        self.annotation_interface = AnnotationInterface(
            storage_path=self.temp_dir + "/annotations",
            auto_create_tasks=False
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ml_enhanced_workflow(self):
        """Test complete workflow with ML enhancement."""
        # 1. Train ML learner with sample data
        training_cases = [
            ("John Doe reported incident at police station", {'PERSON': ['John Doe'], 'ORG': ['police station']}, 'train_1'),
            ("Priya Sharma was victim", {'PERSON': ['Priya Sharma']}, 'train_2'),
            ("Incident at Andheri on 15/03/2023", {'GPE': ['Andheri'], 'DATE': ['15/03/2023']}, 'train_3')
        ]

        for text, entities, context in training_cases:
            self.ml_learner.add_training_example(text, entities, context)

        self.ml_learner.train_entity_classifier()

        # 2. Process FIR with ML enhancement
        fir_row = [
            "Andheri Police Station, CR No 789/2023, Section 379 IPC",
            "15/03/2023, Andheri",
            "John Doe, Age 35",
            "Priya Sharma, Age 28",
            "",
            "",
            "Theft case"
        ]

        result = self.fir_parser.parse_fir_row_enhanced(fir_row)

        # 3. Verify ML enhancement in results
        self.assertIn('processing_metadata', result)
        metadata = result['processing_metadata']
        self.assertTrue(metadata['ml_enabled'])

        # 4. Collect and verify data quality
        example_id = self.data_collector.collect_extraction_result(
            source_file="ml_test.pdf",
            extraction_input="Test input with ML enhancement",
            expected_output=result,
            actual_output=result,
            confidence_scores=result['confidence_scores'],
            processing_metadata=result['processing_metadata']
        )

        self.assertIsNotNone(example_id)

    def test_annotation_workflow_flow(self):
        """Test data flow through annotation workflow."""
        # 1. Create low-quality example (needs annotation)
        low_quality_data = {
            'police_station': 'Test Station',
            'cr_no': '123/456',
            'complainant': {'name': 'Jon Doe'}  # Slight error for annotation
        }

        example_id = self.data_collector.collect_extraction_result(
            source_file="annotation_test.pdf",
            extraction_input="Test input",
            expected_output={'complainant': {'name': 'John Doe'}},  # Correct version
            actual_output=low_quality_data,
            confidence_scores={'complainant_name': 0.6},  # Low confidence
            processing_metadata={}
        )

        # 2. Create annotation task
        task_id = self.annotation_interface.create_annotation_task(
            example_id=example_id,
            annotator_id="test_annotator",
            priority=8
        )

        self.assertIsNotNone(task_id)

        # 3. Simulate annotation submission
        corrections = [
            {
                'field_name': 'complainant_name',
                'original_value': 'Jon Doe',
                'corrected_value': 'John Doe',
                'correction_type': 'correction',
                'confidence': 0.9
            }
        ]

        success = self.annotation_interface.submit_annotation(
            task_id=task_id,
            annotator_id="test_annotator",
            corrections=corrections,
            notes="Fixed typo in name"
        )

        self.assertTrue(success)

        # 4. Verify corrected data
        corrected_example = None
        for example in self.data_collector.examples:
            if example.id == example_id:
                corrected_example = example
                break

        self.assertIsNotNone(corrected_example)
        self.assertTrue(corrected_example.is_corrected)


if __name__ == '__main__':
    # Run data flow tests
    unittest.main(verbosity=2)