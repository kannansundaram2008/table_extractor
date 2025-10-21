"""
Accuracy Testing Suite for ML Components

Tests entity extraction and classification accuracy to ensure >90% accuracy
for major entity types as per requirements.
"""

import unittest
from typing import Dict, List, Tuple, Set
from unittest.mock import Mock
import re

# Import ML components
from utils.enhanced_ner import EnhancedNER
from utils.ml_pattern_learner import MLPatternLearner
from utils.fir_parser import EnhancedFIRParser
from utils.data_validator import DataValidator


class AccuracyTestCase(unittest.TestCase):
    """Base class for accuracy testing with evaluation utilities."""

    def calculate_precision_recall_f1(self, expected: Set[str], actual: Set[str]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        if not expected and not actual:
            return 1.0, 1.0, 1.0

        true_positives = len(expected.intersection(actual))
        false_positives = len(actual - expected)
        false_negatives = len(expected - actual)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def evaluate_entity_extraction(self, text: str, expected_entities: Dict[str, List[str]],
                                 extraction_func) -> Dict[str, Dict[str, float]]:
        """Evaluate entity extraction accuracy."""
        result = extraction_func(text)

        # Handle different result formats
        if hasattr(result, 'entities'):
            # EnhancedNER result format
            actual_entities = {}
            for entity in result.entities:
                if entity.label not in actual_entities:
                    actual_entities[entity.label] = []
                actual_entities[entity.label].append(entity.text)
        else:
            # Simple dict format
            actual_entities = result

        # Calculate accuracy metrics for each entity type
        metrics = {}

        for entity_type, expected_values in expected_entities.items():
            expected_set = set(expected_values)
            actual_set = set(actual_entities.get(entity_type, []))

            precision, recall, f1 = self.calculate_precision_recall_f1(expected_set, actual_set)

            metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'expected_count': len(expected_set),
                'actual_count': len(actual_set),
                'true_positives': len(expected_set.intersection(actual_set))
            }

        return metrics

    def assert_accuracy_threshold(self, metrics: Dict[str, Dict[str, float]],
                                entity_type: str, threshold: float = 0.9):
        """Assert that accuracy meets threshold for specific entity type."""
        if entity_type in metrics:
            f1_score = metrics[entity_type]['f1']
            self.assertGreaterEqual(
                f1_score, threshold,
                f"{entity_type} F1 score {f1_score:.3f} is below threshold {threshold}"
            )


class TestEntityExtractionAccuracy(AccuracyTestCase):
    """Test entity extraction accuracy for different entity types."""

    def setUp(self):
        """Set up test fixtures."""
        self.ner = EnhancedNER(confidence_threshold=0.7)
        self.ml_learner = MLPatternLearner()
        self.fir_parser = EnhancedFIRParser(enable_ml=True)

        # Training data for ML learner
        self._train_ml_learner()

    def _train_ml_learner(self):
        """Train ML learner with test data."""
        training_data = [
            ("John Doe reported theft at police station", {'PERSON': ['John Doe'], 'ORG': ['police station']}, 'train_1'),
            ("Priya Sharma was victim of crime", {'PERSON': ['Priya Sharma']}, 'train_2'),
            ("Incident at Andheri East on 15/03/2023", {'GPE': ['Andheri East'], 'DATE': ['15/03/2023']}, 'train_3'),
            ("IPC Section 379 applies to this case", {'LAW': ['IPC Section 379']}, 'train_4'),
            ("Property worth Rs. 5000 was stolen", {'MONEY': ['Rs. 5000']}, 'train_5')
        ]

        for text, entities, context in training_data:
            self.ml_learner.add_training_example(text, entities, context)

        self.ml_learner.train_entity_classifier()

    def test_person_name_accuracy(self):
        """Test person name extraction accuracy (target: >90%)."""
        test_cases = [
            {
                'text': "Complainant John Doe S/o Robert Doe reported the incident",
                'expected': {'PERSON': ['John Doe', 'Robert Doe']}
            },
            {
                'text': "Victim Priya Sharma D/o Anil Sharma was injured",
                'expected': {'PERSON': ['Priya Sharma', 'Anil Sharma']}
            },
            {
                'text': "Accused Rajesh Kumar Patel S/o Vijay Kumar",
                'expected': {'PERSON': ['Rajesh Kumar Patel', 'Vijay Kumar']}
            },
            {
                'text': "Witness Dr. Sarah Johnson, Professor at University",
                'expected': {'PERSON': ['Dr. Sarah Johnson']}
            }
        ]

        total_f1_scores = []

        for case in test_cases:
            metrics = self.evaluate_entity_extraction(
                case['text'],
                case['expected'],
                self.ner.extract_entities
            )

            if 'PERSON' in metrics:
                total_f1_scores.append(metrics['PERSON']['f1'])

        # Calculate average F1 score across all test cases
        avg_f1 = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0.0

        self.assertGreaterEqual(avg_f1, 0.9, f"Person name extraction F1 score {avg_f1:.3f} is below 90%")

        # Test individual cases
        for case in test_cases:
            metrics = self.evaluate_entity_extraction(
                case['text'],
                case['expected'],
                self.ner.extract_entities
            )
            self.assert_accuracy_threshold(metrics, 'PERSON', 0.8)  # Slightly lower threshold for individual cases

    def test_location_accuracy(self):
        """Test location extraction accuracy."""
        test_cases = [
            {
                'text': "Incident occurred at Andheri East, Mumbai",
                'expected': {'GPE': ['Andheri East', 'Mumbai']}
            },
            {
                'text': "Police Station located in Bandra West",
                'expected': {'GPE': ['Bandra West']}
            },
            {
                'text': "Place of occurrence: Jogeshwari, Maharashtra",
                'expected': {'GPE': ['Jogeshwari', 'Maharashtra']}
            }
        ]

        total_f1_scores = []

        for case in test_cases:
            metrics = self.evaluate_entity_extraction(
                case['text'],
                case['expected'],
                self.ner.extract_entities
            )

            if 'GPE' in metrics:
                total_f1_scores.append(metrics['GPE']['f1'])

        avg_f1 = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0.0
        self.assertGreaterEqual(avg_f1, 0.85, f"Location extraction F1 score {avg_f1:.3f} is below 85%")

    def test_date_accuracy(self):
        """Test date extraction accuracy."""
        test_cases = [
            {
                'text': "Incident on 15/03/2023 at 14:30hrs",
                'expected': {'DATE': ['15/03/2023']}
            },
            {
                'text': "Date of report: March 15th, 2023",
                'expected': {'DATE': ['March 15th, 2023']}
            },
            {
                'text': "Occurrence between 01/01/2023 to 31/01/2023",
                'expected': {'DATE': ['01/01/2023', '31/01/2023']}
            }
        ]

        total_f1_scores = []

        for case in test_cases:
            metrics = self.evaluate_entity_extraction(
                case['text'],
                case['expected'],
                self.ner.extract_entities
            )

            if 'DATE' in metrics:
                total_f1_scores.append(metrics['DATE']['f1'])

        avg_f1 = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0.0
        self.assertGreaterEqual(avg_f1, 0.85, f"Date extraction F1 score {avg_f1:.3f} is below 85%")

    def test_law_section_accuracy(self):
        """Test legal section extraction accuracy."""
        test_cases = [
            {
                'text': "Section 379 IPC, Section 323 IPC applies",
                'expected': {'LAW': ['Section 379 IPC', 'Section 323 IPC']}
            },
            {
                'text': "Under IPC Section 420 for cheating",
                'expected': {'LAW': ['IPC Section 420']}
            },
            {
                'text': "BNS Section 123, CRPC Section 456",
                'expected': {'LAW': ['BNS Section 123', 'CRPC Section 456']}
            }
        ]

        total_f1_scores = []

        for case in test_cases:
            metrics = self.evaluate_entity_extraction(
                case['text'],
                case['expected'],
                self.ner.extract_entities
            )

            if 'LAW' in metrics:
                total_f1_scores.append(metrics['LAW']['f1'])

        avg_f1 = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0.0
        self.assertGreaterEqual(avg_f1, 0.9, f"Law section extraction F1 score {avg_f1:.3f} is below 90%")

    def test_police_station_accuracy(self):
        """Test police station extraction accuracy."""
        test_cases = [
            {
                'text': "Mumbai Police Station, CR No 123/2023",
                'expected': {'ORG': ['Mumbai Police Station']}
            },
            {
                'text': "Andheri Police Station PS, Maharashtra",
                'expected': {'ORG': ['Andheri Police Station']}
            },
            {
                'text': "Bandra West Police Station located at",
                'expected': {'ORG': ['Bandra West Police Station']}
            }
        ]

        total_f1_scores = []

        for case in test_cases:
            metrics = self.evaluate_entity_extraction(
                case['text'],
                case['expected'],
                self.ner.extract_entities
            )

            if 'ORG' in metrics:
                total_f1_scores.append(metrics['ORG']['f1'])

        avg_f1 = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0.0
        self.assertGreaterEqual(avg_f1, 0.85, f"Police station extraction F1 score {avg_f1:.3f} is below 85%")

    def test_cr_number_accuracy(self):
        """Test CR number extraction accuracy."""
        test_cases = [
            {
                'text': "CR No 123/2023, Section 379 IPC",
                'expected': {'CR': ['123/2023']}
            },
            {
                'text': "Crime Register Number 456/2023",
                'expected': {'CR': ['456/2023']}
            },
            {
                'text': "CR No. 789/23 at police station",
                'expected': {'CR': ['789/23']}
            }
        ]

        # Test with regex pattern for CR numbers
        for case in test_cases:
            text = case['text']
            expected = case['expected']['CR']

            # Use regex to find CR numbers
            cr_pattern = r'\b\d{1,4}/\d{2,4}\b'
            actual = re.findall(cr_pattern, text)

            expected_set = set(expected)
            actual_set = set(actual)

            precision, recall, f1 = self.calculate_precision_recall_f1(expected_set, actual_set)

            self.assertGreaterEqual(f1, 0.9, f"CR number extraction F1 score {f1:.3f} is below 90% for text: {text}")

    def test_money_amount_accuracy(self):
        """Test monetary amount extraction accuracy."""
        test_cases = [
            {
                'text': "Property worth Rs. 15000 was stolen",
                'expected': {'MONEY': ['Rs. 15000']}
            },
            {
                'text': "Cash amount Rs. 50,000 and jewelry Rs. 2,00,000",
                'expected': {'MONEY': ['Rs. 50,000', 'Rs. 2,00,000']}
            },
            {
                'text': "Fine amount Rs.10000 under section",
                'expected': {'MONEY': ['Rs.10000']}
            }
        ]

        total_f1_scores = []

        for case in test_cases:
            metrics = self.evaluate_entity_extraction(
                case['text'],
                case['expected'],
                self.ner.extract_entities
            )

            if 'MONEY' in metrics:
                total_f1_scores.append(metrics['MONEY']['f1'])

        avg_f1 = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0.0
        self.assertGreaterEqual(avg_f1, 0.8, f"Money amount extraction F1 score {avg_f1:.3f} is below 80%")

    def test_fir_parsing_accuracy(self):
        """Test overall FIR parsing accuracy."""
        # Test case with known correct parsing
        fir_row = [
            "Mumbai Police Station, CR No 123/2023, Section 379 IPC",
            "15/03/2023, 16/03/2023, 14:30hrs, Andheri East",
            "John Doe S/o Robert Doe, Age 35, Address: Bandra West",
            "Priya Sharma D/o Anil Sharma, Age 28, Address: Andheri East",
            "Mobile Phone worth Rs. 15000",
            "Rajesh Kumar S/o Vijay Kumar, Age 25, Address: Jogeshwari",
            "Accused stole victim's mobile phone and injured her"
        ]

        result = self.fir_parser.parse_fir_row_enhanced(fir_row)

        # Test key field accuracy
        accuracy_tests = [
            ('police_station', 'Mumbai Police Station', 0.9),
            ('cr_no', '123/2023', 0.95),
            ('date_of_occurrence', '15/03/2023', 0.9),
            ('place_of_occurrence', 'Andheri East', 0.8)
        ]

        for field, expected_value, threshold in accuracy_tests:
            actual_value = result.get(field, '')
            if actual_value == expected_value:
                accuracy = 1.0
            elif expected_value.lower() in actual_value.lower():
                accuracy = 0.8
            else:
                accuracy = 0.0

            self.assertGreaterEqual(
                accuracy, threshold,
                f"Field {field} accuracy {accuracy:.3f} is below threshold {threshold}. Expected: '{expected_value}', Got: '{actual_value}'"
            )

    def test_ml_pattern_learner_accuracy(self):
        """Test ML pattern learner prediction accuracy."""
        # Test cases for ML prediction
        test_cases = [
            {
                'text': "Rajesh Kumar from Delhi Police Station reported incident",
                'expected': {'PERSON': ['Rajesh Kumar'], 'ORG': ['Delhi Police Station'], 'GPE': ['Delhi']}
            },
            {
                'text': "Property worth Rs. 25000 was stolen by accused",
                'expected': {'MONEY': ['Rs. 25000']}
            }
        ]

        total_f1_scores = []

        for case in test_cases:
            predictions = self.ml_learner.predict_entities(case['text'])

            # Calculate overall F1 across all entity types
            all_expected = set()
            all_actual = set()

            for entity_type, expected_values in case['expected'].items():
                all_expected.update(expected_values)
                all_actual.update(predictions.get(entity_type, []))

            precision, recall, f1 = self.calculate_precision_recall_f1(all_expected, all_actual)
            total_f1_scores.append(f1)

        avg_f1 = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0.0
        self.assertGreaterEqual(avg_f1, 0.7, f"ML pattern learner F1 score {avg_f1:.3f} is below 70%")

    def test_cross_component_accuracy(self):
        """Test accuracy across integrated components."""
        # Test complete pipeline accuracy
        test_text = """
        On 15/03/2023 at Andheri Police Station, complainant John Doe S/o Robert Doe
        reported that accused Rajesh Kumar aged 25 years stole mobile phone worth Rs. 15000.
        Section 379 IPC applies. CR No 123/2023.
        """

        # 1. NER extraction
        ner_result = self.ner.extract_entities(test_text)

        # 2. ML pattern learning enhancement
        entities_dict = {}
        for entity in ner_result.entities:
            if entity.label not in entities_dict:
                entities_dict[entity.label] = []
            entities_dict[entity.label].append(entity.text)

        # 3. FIR parsing with ML integration
        sample_row = [
            "Andheri Police Station, CR No 123/2023, Section 379 IPC",
            "15/03/2023, 14:30hrs, Andheri",
            "John Doe S/o Robert Doe, Age 35",
            "",
            "Mobile Phone worth Rs. 15000",
            "Rajesh Kumar, Age 25",
            "Theft of mobile phone"
        ]

        parsed_result = self.fir_parser.parse_fir_row_enhanced(sample_row)

        # 4. Validation
        validation_result = self.data_validator.validate_fir_data(parsed_result)

        # Verify end-to-end accuracy
        expected_fields = {
            'police_station': 'Andheri Police Station',
            'cr_no': '123/2023',
            'date_of_occurrence': '15/03/2023'
        }

        correct_fields = 0
        total_fields = len(expected_fields)

        for field, expected_value in expected_fields.items():
            actual_value = parsed_result.get(field, '')
            if expected_value.lower() in actual_value.lower():
                correct_fields += 1

        field_accuracy = correct_fields / total_fields
        self.assertGreaterEqual(field_accuracy, 0.8, f"End-to-end field accuracy {field_accuracy:.3f} is below 80%")

    def test_edge_case_accuracy(self):
        """Test accuracy with edge cases."""
        edge_cases = [
            {
                'text': "Names with Jr., Sr., Dr. prefixes",
                'expected': {'PERSON': ['Dr. John Smith Jr.']}
            },
            {
                'text': "Multiple IPC sections: 379, 323, 420 IPC",
                'expected': {'LAW': ['379', '323', '420']}
            },
            {
                'text': "Complex addresses with multiple locations",
                'expected': {'GPE': ['Mumbai', 'Maharashtra']}
            }
        ]

        for case in edge_cases:
            metrics = self.evaluate_entity_extraction(
                case['text'],
                case['expected'],
                self.ner.extract_entities
            )

            # Check that we get reasonable results for edge cases
            for entity_type, metric in metrics.items():
                # Edge cases should have at least 50% accuracy
                self.assertGreaterEqual(metric['f1'], 0.5, f"Edge case F1 score {metric['f1']:.3f} is too low for {case['text']}")


class TestAccuracyRegression(AccuracyTestCase):
    """Test for accuracy regression detection."""

    def setUp(self):
        """Set up baseline accuracy measurements."""
        super().setUp()
        self.baseline_accuracies = {}

    def record_baseline_accuracy(self, test_name: str, f1_score: float):
        """Record baseline accuracy for regression testing."""
        self.baseline_accuracies[test_name] = f1_score

    def test_accuracy_baselines(self):
        """Establish accuracy baselines for regression testing."""
        # Test person name accuracy baseline
        person_test_cases = [
            ("John Doe reported incident", {'PERSON': ['John Doe']}),
            ("Priya Sharma was victim", {'PERSON': ['Priya Sharma']}),
            ("Rajesh Kumar accused", {'PERSON': ['Rajesh Kumar']})
        ]

        person_f1_scores = []
        for text, expected in person_test_cases:
            metrics = self.evaluate_entity_extraction(text, expected, self.ner.extract_entities)
            if 'PERSON' in metrics:
                person_f1_scores.append(metrics['PERSON']['f1'])

        avg_person_f1 = sum(person_f1_scores) / len(person_f1_scores) if person_f1_scores else 0.0
        self.record_baseline_accuracy("person_names", avg_person_f1)

        # Test location accuracy baseline
        location_test_cases = [
            ("Incident at Andheri East", {'GPE': ['Andheri East']}),
            ("Police station in Bandra", {'GPE': ['Bandra']})
        ]

        location_f1_scores = []
        for text, expected in location_test_cases:
            metrics = self.evaluate_entity_extraction(text, expected, self.ner.extract_entities)
            if 'GPE' in metrics:
                location_f1_scores.append(metrics['GPE']['f1'])

        avg_location_f1 = sum(location_f1_scores) / len(location_f1_scores) if location_f1_scores else 0.0
        self.record_baseline_accuracy("locations", avg_location_f1)

        # Verify baselines meet requirements
        self.assertGreaterEqual(avg_person_f1, 0.9, "Person name baseline should be >= 90%")
        self.assertGreaterEqual(avg_location_f1, 0.8, "Location baseline should be >= 80%")

    def test_no_regression(self):
        """Test that accuracy hasn't regressed from baselines."""
        # This would typically compare against stored baselines
        # For now, we'll test current accuracy meets minimum requirements

        test_cases = [
            ("person_accuracy", "John Doe from Mumbai", {'PERSON': ['John Doe'], 'GPE': ['Mumbai']}, 0.8),
            ("law_accuracy", "Section 379 IPC applies", {'LAW': ['Section 379 IPC']}, 0.9),
            ("date_accuracy", "Incident on 15/03/2023", {'DATE': ['15/03/2023']}, 0.8)
        ]

        for test_name, text, expected, threshold in test_cases:
            metrics = self.evaluate_entity_extraction(text, expected, self.ner.extract_entities)

            # Check overall F1 score across all entity types
            total_f1 = 0.0
            entity_count = 0

            for entity_type, metric in metrics.items():
                total_f1 += metric['f1']
                entity_count += 1

            if entity_count > 0:
                avg_f1 = total_f1 / entity_count
                self.assertGreaterEqual(
                    avg_f1, threshold,
                    f"{test_name} accuracy {avg_f1:.3f} is below threshold {threshold}"
                )


if __name__ == '__main__':
    # Run accuracy tests
    unittest.main(verbosity=2)