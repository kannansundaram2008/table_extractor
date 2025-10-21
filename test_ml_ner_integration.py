#!/usr/bin/env python3
"""
Test script for ML-NER integration verification.
Tests the enhanced NER system and FIR parser integration.
"""

import sys
import os
import logging
from typing import Dict, Any

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from enhanced_ner import EnhancedNER, extract_entities_with_ner
from fir_parser import parse_fir_row_enhanced, EnhancedFIRParser
from ml_pattern_learner import MLPatternLearner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_enhanced_ner():
    """Test the enhanced NER system."""
    logger.info("Testing Enhanced NER System...")

    # Test text with various entities
    test_text = """
    On 15/03/2023 at 14:30hrs in Mumbai Police Station, complainant John Doe S/o Robert Doe
    reported that accused Rajesh Kumar aged 25 years at Andheri East stole property worth Rs. 5000.
    Victim Priya Sharma D/o Anil Sharma was injured. Section 379 IPC applies.
    """

    try:
        # Test enhanced NER
        ner = EnhancedNER(confidence_threshold=0.7)
        result = ner.extract_entities(test_text)

        logger.info(f"Enhanced NER found {len(result.entities)} entities:")
        for entity in result.entities:
            logger.info(f"  - {entity.text} ({entity.label}): {entity.confidence:.3f}")

        # Test backward compatibility function
        old_format = extract_entities_with_ner(test_text)
        logger.info(f"Backward compatibility format: {len(old_format['persons'])} persons, {len(old_format['locations'])} locations")

        return True

    except Exception as e:
        logger.error(f"Enhanced NER test failed: {e}")
        return False


def test_fir_parser_integration():
    """Test the enhanced FIR parser integration."""
    logger.info("Testing Enhanced FIR Parser Integration...")

    # Sample FIR row data
    sample_row = [
        "Mumbai Police Station, CR No 123/2023, Section 379 IPC",
        "15/03/2023, 16/03/2023, 14:30hrs, Andheri East",
        "John Doe S/o Robert Doe, Age 35, Address: Bandra West",
        "Priya Sharma D/o Anil Sharma, Age 28, Address: Andheri East",
        "Mobile Phone worth Rs. 15000",
        "Rajesh Kumar S/o Vijay Kumar, Age 25, Address: Jogeshwari",
        "Accused stole victim's mobile phone and injured her"
    ]

    try:
        # Test enhanced parser
        parser = EnhancedFIRParser(confidence_threshold=0.7, enable_ml=True)
        result = parser.parse_fir_row_enhanced(sample_row)

        logger.info("Enhanced FIR Parser Results:")
        logger.info(f"  Police Station: {result['police_station']} (confidence: {result['confidence_scores'].get('police_station', 'N/A')})")
        logger.info(f"  CR No: {result['cr_no']} (confidence: {result['confidence_scores'].get('cr_no', 'N/A')})")
        logger.info(f"  Section of Law: {result['section_of_law']} (confidence: {result['confidence_scores'].get('section_of_law', 'N/A')})")
        logger.info(f"  Date of Occurrence: {result['date_of_occurrence']} (confidence: {result['confidence_scores'].get('date_of_occurrence', 'N/A')})")
        logger.info(f"  Place of Occurrence: {result['place_of_occurrence']} (confidence: {result['confidence_scores'].get('place_of_occurrence', 'N/A')})")
        logger.info(f"  Complainant: {result['complainant']['name']} (confidence: {result['confidence_scores'].get('complainant_name', 'N/A')})")
        logger.info(f"  Victims Count: {result['victims_count']} (confidence: {result['confidence_scores'].get('victims_count', 'N/A')})")
        logger.info(f"  Accused Count: {len(result['accused'])} (confidence: {result['confidence_scores'].get('accused_count', 'N/A')})")

        # Check processing metadata
        metadata = result['processing_metadata']
        logger.info(f"  ML Enabled: {metadata['ml_enabled']}")
        logger.info(f"  NER Processing Time: {metadata['ner_processing_time']:.3f}s")
        logger.info(f"  ML Processing Time: {metadata['ml_processing_time']:.3f}s")
        logger.info(f"  Fallback Used: {metadata['fallback_used']}")

        # Get performance stats
        stats = parser.get_performance_stats()
        logger.info(f"Performance Stats: {stats['parse_count']} parses, {stats['avg_parsing_time']:.3f}s avg time")

        return True

    except Exception as e:
        logger.error(f"FIR parser integration test failed: {e}")
        return False


def test_ml_pattern_learner():
    """Test the ML pattern learner integration."""
    logger.info("Testing ML Pattern Learner...")

    try:
        # Initialize ML learner
        ml_learner = MLPatternLearner()

        # Test with sample text
        test_text = "John Doe reported theft of mobile phone worth Rs. 5000 from Andheri"

        # Add training example
        entities = {
            'PERSON': ['John Doe'],
            'GPE': ['Andheri'],
            'MONEY': ['Rs. 5000']
        }
        ml_learner.add_training_example(test_text, entities, 'test_context')

        # Test prediction
        predictions = ml_learner.predict_entities(test_text)
        logger.info(f"ML Predictions: {predictions}")

        # Get model stats
        stats = ml_learner.get_model_stats()
        logger.info(f"ML Model Stats: {stats}")

        return True

    except Exception as e:
        logger.error(f"ML pattern learner test failed: {e}")
        return False


def test_spacy_integration():
    """Test that spaCy is being used instead of simple regex."""
    logger.info("Testing spaCy Integration...")

    test_text = "John Doe from Mumbai reported the incident at Andheri Police Station on 15/03/2023"

    try:
        # Test enhanced NER
        ner = EnhancedNER()
        result = ner.extract_entities(test_text)

        # Check if we got spaCy-style entities (more sophisticated than simple regex)
        entity_types = set(e.label for e in result.entities)
        entity_confidences = [e.confidence for e in result.entities]

        logger.info(f"Entity types found: {entity_types}")
        logger.info(f"Confidence scores: {entity_confidences}")

        # Check if we have proper entity labels (not just basic regex matches)
        has_proper_labels = any(label in ['PERSON', 'GPE', 'ORG', 'DATE'] for label in entity_types)

        if has_proper_labels and len(entity_confidences) > 0:
            logger.info("✅ spaCy integration verified - proper entity labels and confidence scores")
            return True
        else:
            logger.warning("❌ spaCy integration may not be working properly")
            return False

    except Exception as e:
        logger.error(f"spaCy integration test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and fallback mechanisms."""
    logger.info("Testing Error Handling and Fallback...")

    try:
        # Test with invalid input
        parser = EnhancedFIRParser()

        # Test with empty input
        result1 = parser.parse_fir_row_enhanced([])
        assert isinstance(result1, dict)
        logger.info("✅ Empty input handled correctly")

        # Test with None input
        result2 = parser.parse_fir_row_enhanced(None)
        assert isinstance(result2, dict)
        logger.info("✅ None input handled correctly")

        # Test enhanced NER with invalid input
        ner = EnhancedNER()
        result3 = ner.extract_entities("")
        assert result3.entities == []
        logger.info("✅ Empty text handled correctly")

        # Test backward compatibility function with error
        try:
            old_result = extract_entities_with_ner("test text")
            assert isinstance(old_result, dict)
            logger.info("✅ Backward compatibility maintained")
        except Exception as e:
            logger.error(f"❌ Backward compatibility broken: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting ML-NER Integration Tests...")

    tests = [
        ("Enhanced NER System", test_enhanced_ner),
        ("spaCy Integration", test_spacy_integration),
        ("FIR Parser Integration", test_fir_parser_integration),
        ("ML Pattern Learner", test_ml_pattern_learner),
        ("Error Handling", test_error_handling),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info('='*50)

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info('='*60)

    passed = 0
    failed = 0

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:25} {status}")

        if success:
            passed += 1
        else:
            failed += 1

    logger.info(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("🎉 All tests passed! ML-NER integration is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)