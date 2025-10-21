import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
from utils.fir_parser import parse_fir_row, extract_entities_with_ner, EnhancedFIRParser, parse_fir_row_enhanced
from utils.enhanced_ner import EnhancedNER
from utils.ml_pattern_learner import MLPatternLearner

class TestFirParser(unittest.TestCase):

    def test_parse_basic_fields(self):
        row_cells = [
            "",  # 0
            "",  # 1
            "",  # 2
            "",  # 3
            "Police Station: Central, CR No: 123/456, Section of Law: IPC 420",  # 4
            "Date of Occurrence: 2023-06-01, Date of Report: 2023-06-02, DO Time: 10:00 AM, DR Time: 11:00 AM, Place of Occurrence: Downtown",  # 5
            "Complainant Name: John Doe, Complainant Address: 123 Main St, Complainant Sex: M, Complainant Age: 35",  # 6
            "Victim 1: Name: Jane Doe, Age: 30, Address: 456 Elm St, Sex: F",  # 7
            "Property Lost: Item: Laptop, Quantity: 1, Property Recovered: Item: Laptop, Quantity: 1, Property Seized: Item: Phone, Quantity: 2",  # 8
            "Accused 1: Name: Richard Roe, Age: 40, Address: 789 Oak St, Sex: M",  # 9
            "Theft reported"  # 10
        ]
        parsed = parse_fir_row(row_cells)
        self.assertEqual(parsed['police_station'], 'Central')
        self.assertEqual(parsed['cr_no'], '123/456')
        self.assertEqual(parsed['section_of_law'], 'IPC 420')
        self.assertEqual(parsed['date_of_occurrence'], '2023-06-01')
        self.assertEqual(parsed['date_of_report'], '2023-06-02')
        self.assertEqual(parsed['do_time'], '10:00 AM')
        self.assertEqual(parsed['dr_time'], '11:00 AM')
        self.assertEqual(parsed['place_of_occurrence'], 'Downtown')
        self.assertEqual(parsed['complainant']['name'], 'John Doe')
        self.assertEqual(parsed['complainant']['address'], '123 Main St')
        self.assertEqual(parsed['complainant']['sex'], 'M')
        self.assertEqual(parsed['complainant']['age'], '35')
        self.assertEqual(parsed['victims_count'], 1)
        self.assertEqual(len(parsed['victims']), 1)
        self.assertEqual(parsed['victims'][0]['name'], 'Jane Doe')
        self.assertEqual(parsed['accused'][0]['name'], 'Richard Roe')
        self.assertEqual(parsed['property_lost'][0]['item'], 'Laptop')
        self.assertEqual(parsed['property_recovered'][0]['item'], 'Laptop')
        self.assertEqual(parsed['property_seized'][0]['item'], 'Phone')
        self.assertEqual(parsed['gist'], 'Theft reported')

    def test_parse_empty(self):
        parsed = parse_fir_row([])
        self.assertEqual(parsed['police_station'], '')
        self.assertEqual(parsed['victims_count'], 0)
        self.assertEqual(parsed['victims'], [])
        self.assertEqual(parsed['accused'], [])
        self.assertEqual(parsed['property_lost'], [])
        self.assertEqual(parsed['gist'], '')

    @patch('utils.fir_parser.spacy')
    def test_extract_entities_with_ner_spacy_available(self, mock_spacy):
        """Test NER extraction when spaCy is available"""
        # Mock spaCy components
        mock_nlp = Mock()
        mock_doc = Mock()

        # Mock entities
        mock_person = Mock()
        mock_person.text = 'John Doe'
        mock_person.label_ = 'PERSON'

        mock_date = Mock()
        mock_date.text = 'June 5th'
        mock_date.label_ = 'DATE'

        mock_location = Mock()
        mock_location.text = 'Downtown'
        mock_location.label_ = 'GPE'

        mock_doc.ents = [mock_person, mock_date, mock_location]
        mock_nlp.return_value = mock_doc
        mock_spacy.load.return_value = mock_nlp

        text = "John Doe reported the incident on June 5th at Downtown."
        entities = extract_entities_with_ner(text)

        self.assertIn('persons', entities)
        self.assertIn('dates', entities)
        self.assertIn('locations', entities)
        self.assertIn('John Doe', entities['persons'])
        self.assertIn('June 5th', entities['dates'])
        self.assertIn('Downtown', entities['locations'])

    @patch('utils.fir_parser.spacy')
    def test_extract_entities_with_ner_spacy_unavailable(self, mock_spacy):
        """Test NER extraction when spaCy is not available"""
        mock_spacy.load.side_effect = ImportError("spaCy not installed")

        text = "John Doe reported the incident on June 5th at Downtown."
        entities = extract_entities_with_ner(text)

        self.assertEqual(entities, {})

    @patch('utils.fir_parser.spacy')
    def test_extract_entities_with_ner_processing_error(self, mock_spacy):
        """Test NER extraction with processing error"""
        mock_nlp = Mock()
        mock_nlp.side_effect = Exception("Processing error")
        mock_spacy.load.return_value = mock_nlp

        text = "John Doe reported the incident."
        entities = extract_entities_with_ner(text)

        # Should return empty dict on error
        self.assertEqual(entities, {})

    def test_parse_partial_data(self):
        # Test with partial data, missing some fields
        row_cells = [
            "", "", "", "",
            "Police Station: Central",  # Only police station
            "Date of Occurrence: 2023-06-01",  # Only one date
            "Complainant Name: John Doe",  # Only name
            "",  # Empty victim
            "",  # Empty property
            "",  # Empty accused
            "Theft"  # Gist
        ]
        parsed = parse_fir_row(row_cells)
        self.assertEqual(parsed['police_station'], 'Central')
        self.assertEqual(parsed['date_of_occurrence'], '2023-06-01')
        self.assertEqual(parsed['complainant']['name'], 'John Doe')
        self.assertEqual(parsed['victims_count'], 0)
        self.assertEqual(parsed['gist'], 'Theft')

    def test_parse_malformed_lines(self):
        # Test with malformed lines, extra colons, etc.
        row_cells = [
            "", "", "", "",
            "Police Station:: Central",  # Extra colon
            "Date of Occurrence: 2023-06-01, Invalid: ",  # Invalid line
            "Complainant Name:: John Doe",  # Extra colon
            "Victim 1: Name:: Jane Doe",  # Extra colon
            "Property Lost: Item:: Laptop",  # Extra colon
            "Accused 1: Name:: Richard Roe",  # Extra colon
            "Theft reported"
        ]
        parsed = parse_fir_row(row_cells)
        self.assertEqual(parsed['police_station'], ': Central')  # Extra colon
        self.assertEqual(parsed['date_of_occurrence'], '2023-06-01')
        self.assertEqual(parsed['complainant']['name'], ': John Doe')  # Extra colon
        self.assertEqual(parsed['victims'][0]['name'], ': Jane Doe')  # Extra colon
        self.assertEqual(parsed['property_lost'][0]['item'], ': Laptop')  # Extra colon
        self.assertEqual(parsed['accused'][0]['name'], ': Richard Roe')  # Extra colon

    def test_parse_row_with_none_values(self):
        """Test parsing row with None values"""
        row_cells = [None, None, "Police Station: Central", None, None]
        parsed = parse_fir_row(row_cells)

        # Should handle None values gracefully
        self.assertEqual(parsed['police_station'], 'Central')
        self.assertEqual(parsed['cr_no'], '')

    def test_parse_row_with_empty_strings(self):
        """Test parsing row with empty strings"""
        row_cells = ["", "", "Police Station: Central", "", ""]
        parsed = parse_fir_row(row_cells)

        self.assertEqual(parsed['police_station'], 'Central')

    def test_parse_row_with_mixed_valid_invalid_data(self):
        """Test parsing row with mix of valid and invalid data"""
        row_cells = [
            "Invalid data",
            "More invalid",
            "Police Station: Central, CR No: 123/456",
            "Date of Occurrence: 2023-06-01",
            "Complainant Name: John Doe",
            "Victim 1: Name: Jane Doe",
            "Property Lost: Item: Laptop",
            "Accused 1: Name: Richard Roe",
            "Valid gist"
        ]
        parsed = parse_fir_row(row_cells)

        # Should extract what it can
        self.assertEqual(parsed['police_station'], 'Central')
        self.assertEqual(parsed['cr_no'], '123/456')
        self.assertEqual(parsed['complainant']['name'], 'John Doe')

    def test_parse_row_extreme_malformation(self):
        """Test parsing extremely malformed row data"""
        row_cells = [
            "::::::::::::::::::::",
            "Police Station:::::::::::::::: Central",
            "Date of Occurrence: 2023-06-01",
            "Complainant Name::::::::::::::::::::::::: John Doe",
            "This is not a proper format at all",
            "Random text with no structure",
            "More random text",
            "Even more random text",
            "Gist with no meaning"
        ]
        parsed = parse_fir_row(row_cells)

        # Should handle gracefully without crashing
        self.assertIsInstance(parsed, dict)
        self.assertIn('police_station', parsed)
        self.assertIn('complainant', parsed)

    def test_parse_row_unicode_handling(self):
        """Test parsing row with Unicode characters"""
        row_cells = [
            "",
            "",
            "Police Station: केंद्र पुलिस स्टेशन, CR No: १२३/४५६",
            "Date of Occurrence: २०२३-०६-०१",
            "Complainant Name: जॉन डो",
            "",
            "",
            "",
            "यूनिकोड टेस्ट"
        ]
        parsed = parse_fir_row(row_cells)

        # Should handle Unicode properly
        self.assertEqual(parsed['police_station'], 'केंद्र पुलिस स्टेशन')
        self.assertEqual(parsed['cr_no'], '१२३/४५६')
        self.assertEqual(parsed['complainant']['name'], 'जॉन डो')

    def test_parse_row_very_long_content(self):
        """Test parsing row with very long content"""
        long_text = "A" * 10000  # Very long string
        row_cells = [
            "",
            "",
            f"Police Station: {long_text}",
            f"Date of Occurrence: {long_text}",
            f"Complainant Name: {long_text}",
            "",
            "",
            "",
            long_text
        ]
        parsed = parse_fir_row(row_cells)

        # Should handle long content without performance issues
        self.assertEqual(parsed['police_station'], long_text)
        self.assertEqual(parsed['gist'], long_text)

    def test_parse_row_special_characters(self):
        """Test parsing row with special characters"""
        row_cells = [
            "",
            "",
            "Police Station: Test@#$%^&*(), CR No: 123/456",
            "Date of Occurrence: 2023-06-01",
            "Complainant Name: O'Connor-Smith Jr.",
            "",
            "",
            "",
            "Special chars: @#$%^&*()"
        ]
        parsed = parse_fir_row(row_cells)

        # Should handle special characters
        self.assertEqual(parsed['police_station'], 'Test@#$%^&*()')
        self.assertEqual(parsed['complainant']['name'], 'O\'Connor-Smith Jr.')
        self.assertIn('@#$%^&*()', parsed['gist'])

    def test_parse_row_sql_injection_like(self):
        """Test parsing row with SQL injection-like content"""
        malicious_content = "'; DROP TABLE users; --"
        row_cells = [
            "",
            "",
            f"Police Station: {malicious_content}",
            "Date of Occurrence: 2023-06-01",
            f"Complainant Name: {malicious_content}",
            "",
            "",
            "",
            malicious_content
        ]
        parsed = parse_fir_row(row_cells)

        # Should handle potentially malicious content safely
        self.assertEqual(parsed['police_station'], malicious_content)
        self.assertEqual(parsed['complainant']['name'], malicious_content)
        self.assertEqual(parsed['gist'], malicious_content)

    def test_parse_row_empty_after_strip(self):
        """Test parsing row where content becomes empty after stripping"""
        row_cells = [
            "   ",
            "\t\n",
            "Police Station: Central",
            "  \t  ",
            "Complainant Name: John Doe",
            "",
            "",
            "",
            "   "
        ]
        parsed = parse_fir_row(row_cells)

        # Should handle whitespace-only fields
        self.assertEqual(parsed['police_station'], 'Central')
        self.assertEqual(parsed['complainant']['name'], 'John Doe')

    def test_parse_row_inconsistent_field_count(self):
        """Test parsing row with inconsistent number of fields"""
        # Less fields than expected
        row_cells = [
            "Police Station: Central",
            "Date of Occurrence: 2023-06-01"
        ]
        parsed = parse_fir_row(row_cells)

        self.assertEqual(parsed['police_station'], 'Central')
        self.assertEqual(parsed['date_of_occurrence'], '2023-06-01')
        # Missing fields should be empty
        self.assertEqual(parsed['cr_no'], '')

    def test_parse_row_with_regex_special_chars(self):
        """Test parsing row with regex special characters"""
        special_content = "Police Station: Test.+*?[](){}^$|"
        row_cells = [
            "",
            "",
            special_content,
            "Date of Occurrence: 2023-06-01",
            "Complainant Name: John Doe",
            "",
            "",
            "",
            "Gist with special chars: .+*?[](){}^$|"
        ]
        parsed = parse_fir_row(row_cells)

        # Should handle regex special characters
        self.assertEqual(parsed['police_station'], 'Test.+*?[](){}^$|')
        self.assertIn('.+*?[](){}^$|', parsed['gist'])


class TestEnhancedFIRParser(unittest.TestCase):
    """Test cases for the enhanced FIR parser with ML integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.enhanced_parser = EnhancedFIRParser(
            confidence_threshold=0.7,
            enable_ml=True,
            enable_logging=False
        )
        self.ner = EnhancedNER(confidence_threshold=0.7)
        self.ml_learner = MLPatternLearner(model_dir=self.temp_dir)

        # Sample test data
        self.complex_fir_row = [
            "Mumbai Police Station, CR No 123/2023, Section 379 IPC, Section 323 IPC",
            "15/03/2023, 16/03/2023, 14:30hrs, 15:45hrs, Andheri East, Mumbai",
            "John Michael Doe S/o Robert Anthony Doe, Age 35, Address: 123 Bandra West, Mumbai, Phone: 9876543210",
            "Priya Mary Sharma D/o Anil Kumar Sharma, Age 28, Address: 456 Andheri East, Mumbai",
            "Property Lost: Samsung Galaxy S21 worth Rs. 45000, Cash Rs. 5000, Gold Chain worth Rs. 75000",
            "Rajesh Kumar Patel S/o Vijay Kumar Patel, Age 25, Address: Jogeshwari, Mumbai",
            "Complex case with multiple victims, accused, and detailed property information involving theft and assault"
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_enhanced_parsing_with_confidence_scores(self):
        """Test enhanced parsing with confidence scores."""
        result = self.enhanced_parser.parse_fir_row_enhanced(self.complex_fir_row)

        # Verify structure
        self.assertIn('confidence_scores', result)
        self.assertIn('processing_metadata', result)
        self.assertIsInstance(result['confidence_scores'], dict)
        self.assertIsInstance(result['processing_metadata'], dict)

        # Verify confidence scores are reasonable
        for field, score in result['confidence_scores'].items():
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        # Verify metadata
        metadata = result['processing_metadata']
        self.assertIn('ml_enabled', metadata)
        self.assertIn('ner_processing_time', metadata)
        self.assertIn('ml_processing_time', metadata)

    def test_ml_enhanced_entity_extraction(self):
        """Test ML-enhanced entity extraction."""
        # Train ML learner with sample data
        training_data = [
            ("John Doe reported incident", {'PERSON': ['John Doe']}, 'train_1'),
            ("Police station in Mumbai", {'ORG': ['Police station'], 'GPE': ['Mumbai']}, 'train_2'),
            ("Section 379 IPC applies", {'LAW': ['Section 379 IPC']}, 'train_3')
        ]

        for text, entities, context in training_data:
            self.ml_learner.add_training_example(text, entities, context)

        self.ml_learner.train_entity_classifier()

        # Test enhanced parsing with ML
        result = self.enhanced_parser.parse_fir_row_enhanced(self.complex_fir_row)

        # Should have ML processing time recorded
        metadata = result['processing_metadata']
        self.assertTrue(metadata['ml_enabled'])
        self.assertGreaterEqual(metadata['ml_processing_time'], 0.0)

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        # Reset performance stats
        self.enhanced_parser.reset_performance_stats()

        # Process multiple rows
        for _ in range(5):
            self.enhanced_parser.parse_fir_row_enhanced(self.complex_fir_row)

        # Get performance stats
        stats = self.enhanced_parser.get_performance_stats()

        # Verify stats structure
        expected_fields = [
            'total_parsing_time', 'total_ner_time', 'total_ml_time',
            'parse_count', 'ner_call_count', 'ml_call_count'
        ]

        for field in expected_fields:
            self.assertIn(field, stats)

        # Verify stats are reasonable
        self.assertGreater(stats['parse_count'], 0)
        self.assertGreaterEqual(stats['total_parsing_time'], 0.0)

        if stats['parse_count'] > 0:
            self.assertIn('avg_parsing_time', stats)
            self.assertGreater(stats['avg_parsing_time'], 0.0)

    def test_error_handling_in_enhanced_parser(self):
        """Test error handling in enhanced parser."""
        # Test with None input
        result = self.enhanced_parser.parse_fir_row_enhanced(None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['police_station'], '')

        # Test with empty list
        result = self.enhanced_parser.parse_fir_row_enhanced([])
        self.assertIsInstance(result, dict)

        # Test with invalid data types
        result = self.enhanced_parser.parse_fir_row_enhanced([123, "test", {}])
        self.assertIsInstance(result, dict)

    def test_ml_integration_toggle(self):
        """Test ML integration can be toggled."""
        # Test with ML enabled
        parser_with_ml = EnhancedFIRParser(enable_ml=True)
        result_with_ml = parser_with_ml.parse_fir_row_enhanced(self.complex_fir_row)

        # Test with ML disabled
        parser_without_ml = EnhancedFIRParser(enable_ml=False)
        result_without_ml = parser_without_ml.parse_fir_row_enhanced(self.complex_fir_row)

        # Both should work
        self.assertIsInstance(result_with_ml, dict)
        self.assertIsInstance(result_without_ml, dict)

        # ML-enabled should have ML metadata
        self.assertTrue(result_with_ml['processing_metadata']['ml_enabled'])
        self.assertFalse(result_without_ml['processing_metadata']['ml_enabled'])

    def test_ner_integration_in_parsing(self):
        """Test NER integration in FIR parsing."""
        # Test that NER enhances entity extraction
        result = self.enhanced_parser.parse_fir_row_enhanced(self.complex_fir_row)

        # Should extract entities with reasonable confidence
        confidence_scores = result['confidence_scores']

        # Key fields should have confidence scores
        key_fields = ['police_station', 'cr_no', 'complainant_name', 'date_of_occurrence']
        for field in key_fields:
            if result.get(field.replace('_name', '')):  # Handle complainant_name -> complainant
                confidence_key = field if field != 'complainant_name' else 'complainant_name'
                if confidence_key in confidence_scores:
                    self.assertGreater(confidence_scores[confidence_key], 0.0)

    def test_complex_entity_extraction(self):
        """Test extraction of complex entities."""
        # Test with complex names and addresses
        complex_row = [
            "Bandra West Police Station, CR No 456/2023, Section 379 IPC, Section 420 IPC",
            "20/03/2023, 21/03/2023, 09:15hrs, Bandra West, Mumbai, Maharashtra",
            "Dr. Priyanka Singhania D/o Dr. Rajesh Singhania, MBBS, MD, Age 32, Address: 789 Carter Road, Bandra West",
            "Mrs. Anita Deshpande W/o Mr. Sanjay Deshpande, Age 45, Address: 321 Linking Road, Bandra",
            "Jewelry worth Rs. 15,00,000 and cash Rs. 2,50,000",
            "Mr. Rohit Malhotra S/o Mr. Suresh Malhotra, Age 28, Address: Andheri West",
            "Sophisticated theft involving multiple high-value items and complex planning"
        ]

        result = self.enhanced_parser.parse_fir_row_enhanced(complex_row)

        # Should handle complex entities
        self.assertIsNotNone(result['police_station'])
        self.assertIsNotNone(result['complainant']['name'])
        self.assertIsNotNone(result['victim']['name'])

        # Should extract multiple sections
        self.assertIn('Section 379 IPC', result['section_of_law'])
        self.assertIn('Section 420 IPC', result['section_of_law'])

    def test_confidence_scoring_accuracy(self):
        """Test accuracy of confidence scoring."""
        # Test with clearly identifiable patterns (should have high confidence)
        clear_row = [
            "Mumbai Police Station, CR No 123/2023, Section 379 IPC",
            "15/03/2023, Mumbai",
            "John Doe, Age 30",
            "",
            "",
            "",
            "Clear case"
        ]

        result = self.enhanced_parser.parse_fir_row_enhanced(clear_row)

        # Clear patterns should have high confidence
        if result['police_station']:
            self.assertGreaterEqual(result['confidence_scores'].get('police_station', 0), 0.7)
        if result['cr_no']:
            self.assertGreaterEqual(result['confidence_scores'].get('cr_no', 0), 0.8)

    def test_fallback_mechanisms(self):
        """Test fallback mechanisms when ML components fail."""
        # Test with ML learner that has no training data
        untrained_parser = EnhancedFIRParser(enable_ml=True)

        # Should still work without trained ML models
        result = untrained_parser.parse_fir_row_enhanced(self.complex_fir_row)

        self.assertIsInstance(result, dict)
        self.assertIn('confidence_scores', result)
        self.assertIn('processing_metadata', result)

        # Should have fallback processing
        metadata = result['processing_metadata']
        self.assertTrue(metadata['ml_enabled'])  # ML enabled but may not be effective

    def test_memory_efficiency(self):
        """Test memory efficiency of enhanced parser."""
        import gc

        gc.collect()
        initial_objects = len(gc.get_objects())

        # Process many documents
        for i in range(50):
            row = [
                f"Police Station {i}, CR No {i}/2023",
                f"0{i%9+1}/03/2023",
                f"Person {i}",
                "",
                "",
                "",
                f"Case {i}"
            ]
            result = self.enhanced_parser.parse_fir_row_enhanced(row)
            self.assertIsInstance(result, dict)

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory increase should be reasonable
        object_increase = final_objects - initial_objects
        self.assertLess(object_increase, 1000, "Enhanced parser should not leak memory")


class TestMLComponentIntegrationInFIRParser(unittest.TestCase):
    """Test ML component integration specifically in FIR parser context."""

    def setUp(self):
        """Set up integrated test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.ner = EnhancedNER()
        self.ml_learner = MLPatternLearner(model_dir=self.temp_dir)
        self.fir_parser = EnhancedFIRParser(enable_ml=True, enable_logging=False)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ner_ml_fir_integration(self):
        """Test complete NER -> ML -> FIR parser integration."""
        # 1. Train ML learner with NER results
        training_text = "John Doe from Mumbai Police Station reported theft under Section 379 IPC"
        ner_result = self.ner.extract_entities(training_text)

        # Convert to training format
        entities_dict = {}
        for entity in ner_result.entities:
            if entity.label not in entities_dict:
                entities_dict[entity.label] = []
            entities_dict[entity.label].append(entity.text)

        self.ml_learner.add_training_example(training_text, entities_dict, 'integration_test')
        self.ml_learner.train_entity_classifier()

        # 2. Test FIR parsing with trained ML
        fir_row = [
            "Mumbai Police Station, CR No 789/2023, Section 379 IPC",
            "25/03/2023, Mumbai",
            "John Doe, Age 35",
            "",
            "",
            "",
            "Integration test case"
        ]

        result = self.fir_parser.parse_fir_row_enhanced(fir_row)

        # 3. Verify integration worked
        self.assertIsInstance(result, dict)
        self.assertIn('confidence_scores', result)
        self.assertIn('processing_metadata', result)

        # 4. Verify ML enhancement
        metadata = result['processing_metadata']
        self.assertTrue(metadata['ml_enabled'])
        self.assertGreaterEqual(metadata['ml_processing_time'], 0.0)

    def test_cross_component_data_consistency(self):
        """Test data consistency across integrated components."""
        # Test that data flows consistently between components
        test_row = [
            "Test Police Station, CR No 999/2023, Section 420 IPC",
            "30/03/2023, Test Location",
            "Test Person, Age 40",
            "",
            "Test Property worth Rs. 10000",
            "",
            "Data consistency test"
        ]

        # Process through enhanced parser
        result = self.fir_parser.parse_fir_row_enhanced(test_row)

        # Verify all expected fields are present
        expected_fields = [
            'police_station', 'cr_no', 'section_of_law', 'date_of_occurrence',
            'place_of_occurrence', 'complainant', 'confidence_scores', 'processing_metadata'
        ]

        for field in expected_fields:
            self.assertIn(field, result, f"Missing field: {field}")

        # Verify data types
        self.assertIsInstance(result['police_station'], str)
        self.assertIsInstance(result['cr_no'], str)
        self.assertIsInstance(result['complainant'], dict)
        self.assertIsInstance(result['confidence_scores'], dict)
        self.assertIsInstance(result['processing_metadata'], dict)

    def test_error_propagation_across_components(self):
        """Test how errors propagate across integrated components."""
        # Test with problematic input
        problematic_row = [
            None,  # None value
            "01/01/2023",
            {"name": "Invalid", "type": "dict"},  # Wrong type
            "",
            "",
            "",
            "Error propagation test"
        ]

        # Should handle gracefully without crashing
        result = self.fir_parser.parse_fir_row_enhanced(problematic_row)

        self.assertIsInstance(result, dict)
        self.assertIn('confidence_scores', result)
        self.assertIn('processing_metadata', result)

        # Should have fallback processing
        metadata = result['processing_metadata']
        self.assertIsNotNone(metadata)


if __name__ == '__main__':
    unittest.main()
