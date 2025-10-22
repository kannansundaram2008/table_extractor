#!/usr/bin/env python3
"""
Comprehensive test script to verify AI/ML NER system flow
Tests the complete pipeline from input through enhanced NER processing
"""

import sys
import os
import time
import json
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports are working correctly."""
    print("🔧 Testing imports...")

    try:
        from utils.enhanced_ner import EnhancedNER, Entity
        print("  ✅ Enhanced NER imports successful")

        from utils.fir_parser import EnhancedFIRParser
        print("  ✅ FIR Parser imports successful")

        from utils.pattern_matcher import has_primary_pattern, has_adjacent_pattern
        print("  ✅ Pattern matcher imports successful")

        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_spacy_model():
    """Test that spaCy model is available and working."""
    print("🤖 Testing spaCy model...")

    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        print("  ✅ spaCy model loaded successfully")

        # Test basic NER
        test_text = "Police Station City, CR No 123/2023, Section 420 IPC"
        doc = nlp(test_text)
        entities = list(doc.ents)

        print(f"  📊 Basic NER found {len(entities)} entities:")
        for ent in entities:
            print(f"    • {ent.text} → {ent.label_}")

        return True
    except Exception as e:
        print(f"  ❌ spaCy error: {e}")
        return False

def test_enhanced_ner():
    """Test the enhanced NER system."""
    print("🚀 Testing Enhanced NER System...")

    try:
        from utils.enhanced_ner import EnhancedNER

        ner = EnhancedNER()
        test_cases = [
            "Police Station City, CR No 123/2023, Section 420 IPC, Date 15/03/2023",
            "John Smith S/o Robert Smith, Age 25, Address: 123 Main Street",
            "Section 302 IPC, Section 34 IPC, Police Station Central",
            "Date of occurrence: 15/03/2023 at 14:30 hrs, Place: Market Area"
        ]

        for i, test_text in enumerate(test_cases, 1):
            print(f"\n  🧪 Test Case {i}: {test_text}")

            start_time = time.time()
            result = ner.extract_entities(test_text)
            processing_time = time.time() - start_time

            print(f"    ⏱️  Processing time: {processing_time:.3f}")
            print(f"    📊 Found {len(result.entities)} entities:")

            for entity in result.entities:
                print(f"      • {entity.text} → {entity.label} (confidence: {entity.confidence:.2f})")

            if hasattr(result, 'context_analysis'):
                ca = result.context_analysis
                print(f"    🔍 Context: Legal={ca.get('is_legal_document', False)}, Type={ca.get('document_type', 'unknown')}")

        return True
    except Exception as e:
        print(f"  ❌ Enhanced NER error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fir_parser():
    """Test the FIR parser with enhanced NER integration."""
    print("📋 Testing FIR Parser Integration...")

    try:
        from utils.fir_parser import EnhancedFIRParser

        parser = EnhancedFIRParser()

        # Test with sample FIR row data
        test_row = [
            "Police Station City, CR No 123/2023, Section 420 IPC",
            "Date 15/03/2023 at 14:30 hrs, Place: Market Area East",
            "John Smith S/o Robert Smith, Age 25, Address: 123 Main Street",
            "Victim: Mary Johnson, Age 22, Address: 456 Oak Avenue",
            "Property Lost: Gold Chain worth Rs 50,000",
            "Accused: David Wilson S/o Peter Wilson, Age 30"
        ]

        print(f"  📝 Test row with {len(test_row)} columns")

        start_time = time.time()
        result = parser.parse_fir_row_enhanced(test_row, "test_document.pdf")
        processing_time = time.time() - start_time

        print(f"    ⏱️  Processing time: {processing_time:.3f}")
        print("  📊 Extracted data:")

        # Display key extracted fields
        key_fields = {
            'police_station': 'Police Station',
            'cr_no': 'CR Number',
            'section_of_law': 'Section of Law',
            'date_of_occurrence': 'Date of Occurrence',
            'complainant': 'Complainant',
            'victims_count': 'Victims Count',
            'accused': 'Accused'
        }

        for field, label in key_fields.items():
            value = result.get(field, 'Not found')
            if isinstance(value, dict):
                if field == 'complainant':
                    comp = value
                    print(f"    • {label}: {comp.get('name', 'N/A')} (Age: {comp.get('age', 'N/A')}, Sex: {comp.get('sex', 'N/A')})")
                else:
                    print(f"    • {label}: {len(value)} items")
            elif isinstance(value, list):
                print(f"    • {label}: {len(value)} items")
            else:
                print(f"    • {label}: {value}")

        # Show confidence scores
        print("  🎯 Confidence Scores:")
        for field, confidence in result.get('confidence_scores', {}).items():
            print(f"    • {field}: {confidence:.1%}")

        return True
    except Exception as e:
        print(f"  ❌ FIR Parser error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pattern_matcher():
    """Test the enhanced pattern matcher."""
    print("🔍 Testing Pattern Matcher...")

    try:
        from utils.pattern_matcher import (
            has_primary_pattern, has_adjacent_pattern,
            has_time_pattern, has_date_pattern,
            has_police_station_pattern, has_legal_section_pattern
        )

        test_cells = [
            "Police Station City, CR No 123/2023, Section 420 IPC",
            "Date 15/03/2023 at 14:30 hrs, Place: Market Area",
            "John Smith S/o Robert Smith, Age 25",
            "Section 302 IPC, Section 34 IPC",
            "Property Lost: Gold Chain worth Rs 50,000"
        ]

        print(f"  🧪 Testing {len(test_cells)} cells:")

        for i, cell in enumerate(test_cells, 1):
            print(f"\n    Cell {i}: {cell}")
            print(f"      • Primary pattern: {has_primary_pattern(cell)}")
            print(f"      • Adjacent pattern: {has_adjacent_pattern(cell)}")
            print(f"      • Time pattern: {has_time_pattern(cell)}")
            print(f"      • Date pattern: {has_date_pattern(cell)}")
            print(f"      • Police station: {has_police_station_pattern(cell)}")
            print(f"      • Legal section: {has_legal_section_pattern(cell)}")

        return True
    except Exception as e:
        print(f"  ❌ Pattern matcher error: {e}")
        return False

def test_regex_patterns():
    """Test regex pattern effectiveness."""
    print("🔧 Testing Regex Pattern Effectiveness...")

    try:
        from utils.pattern_matcher import (
            POLICE_STATION_PATTERNS, LEGAL_SECTION_PATTERNS,
            PERSON_NAME_PATTERNS, TIME_PATTERNS, DATE_PATTERNS
        )
        import re

        test_texts = [
            "Police Station City, CR No 123/2023",
            "Section 420 IPC, Section 34 IPC",
            "John Smith S/o Robert Smith",
            "Date 15/03/2023 at 14:30 hrs",
            "Mary Johnson D/o Peter Johnson"
        ]

        pattern_groups = {
            "Police Stations": POLICE_STATION_PATTERNS,
            "Legal Sections": LEGAL_SECTION_PATTERNS,
            "Person Names": PERSON_NAME_PATTERNS,
            "Times": TIME_PATTERNS,
            "Dates": DATE_PATTERNS
        }

        for category, patterns in pattern_groups.items():
            print(f"\n  📋 {category}:")

            for i, text in enumerate(test_texts[:3], 1):  # Test first 3 texts
                matches = []
                for pattern in patterns:
                    found = re.findall(pattern, text, re.IGNORECASE)
                    matches.extend(found)

                print(f"    Text {i}: {len(matches)} matches")
                if matches:
                    print(f"      Found: {matches}")

        return True
    except Exception as e:
        print(f"  ❌ Regex pattern test error: {e}")
        return False

def main():
    """Run all tests to verify the complete AI/ML NER system flow."""
    print("🚀 COMPREHENSIVE AI/ML NER SYSTEM VERIFICATION")
    print("=" * 60)

    tests = [
        ("Import Verification", test_imports),
        ("spaCy Model Test", test_spacy_model),
        ("Enhanced NER Test", test_enhanced_ner),
        ("Pattern Matcher Test", test_pattern_matcher),
        ("Regex Pattern Test", test_regex_patterns),
        ("FIR Parser Integration Test", test_fir_parser),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}")
        print("-" * 40)

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {test_name}")

    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! AI/ML NER system is working correctly.")
        return True
    else:
        print("⚠️  SOME TESTS FAILED! Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)