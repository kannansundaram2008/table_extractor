"""
Data Validation and Quality Checks for Training Data Collection

This module provides comprehensive validation and quality checking capabilities
for FIR document training data. It ensures data integrity, consistency, and
quality before using it for model training.
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import logging

# Import data collector
from utils.data_collector import TrainingExample, get_data_collector

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float
    field_scores: Dict[str, float]

@dataclass
class QualityMetrics:
    """Quality metrics for training data."""
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    format_compliance_score: float
    overall_score: float

class TrainingDataValidator:
    """
    Comprehensive validator for training data quality and integrity.
    """

    def __init__(self):
        """Initialize the data validator."""
        self.validation_rules = self._load_validation_rules()
        self.fir_fields = {
            'police_station', 'cr_no', 'section_of_law', 'date_of_occurrence',
            'date_of_report', 'do_time', 'dr_time', 'place_of_occurrence',
            'complainant', 'victims', 'accused', 'property_lost',
            'property_recovered', 'property_seized', 'gist'
        }

        # PII fields that may need anonymization
        self.pii_fields = {
            'complainant_name', 'victim_name', 'accused_name',
            'complainant_address', 'victim_address', 'accused_address'
        }

        logger.info("TrainingDataValidator initialized")

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for different field types."""
        return {
            'cr_no': {
                'required': True,
                'pattern': r'^\d{1,4}/\d{2,4}$',
                'description': 'CR number format: digits/digits (e.g., 123/2024)'
            },
            'police_station': {
                'required': False,
                'min_length': 3,
                'max_length': 200,
                'description': 'Police station name'
            },
            'section_of_law': {
                'required': False,
                'min_length': 3,
                'max_length': 500,
                'description': 'Legal sections (IPC, BNS, etc.)'
            },
            'date_of_occurrence': {
                'required': False,
                'date_format': True,
                'description': 'Date of occurrence'
            },
            'date_of_report': {
                'required': False,
                'date_format': True,
                'description': 'Date of report'
            },
            'do_time': {
                'required': False,
                'time_format': True,
                'description': 'Time of occurrence'
            },
            'dr_time': {
                'required': False,
                'time_format': True,
                'description': 'Time of report'
            },
            'place_of_occurrence': {
                'required': False,
                'min_length': 3,
                'max_length': 200,
                'description': 'Place/location of occurrence'
            },
            'complainant': {
                'required': False,
                'structure_check': True,
                'description': 'Complainant information'
            },
            'victims': {
                'required': False,
                'list_check': True,
                'description': 'Victim information list'
            },
            'accused': {
                'required': False,
                'list_check': True,
                'description': 'Accused information list'
            },
            'property_lost': {
                'required': False,
                'list_check': True,
                'description': 'Lost property list'
            },
            'property_recovered': {
                'required': False,
                'list_check': True,
                'description': 'Recovered property list'
            },
            'property_seized': {
                'required': False,
                'list_check': True,
                'description': 'Seized property list'
            },
            'gist': {
                'required': False,
                'min_length': 10,
                'max_length': 2000,
                'description': 'Case summary/gist'
            }
        }

    def validate_training_example(self, example: TrainingExample) -> ValidationResult:
        """
        Validate a single training example.

        Args:
            example: Training example to validate

        Returns:
            ValidationResult with errors, warnings, and quality score
        """
        errors = []
        warnings = []
        field_scores = {}

        # Validate each field
        for field_name in self.fir_fields:
            field_result = self._validate_field(field_name, example)
            field_scores[field_name] = field_result['score']

            if field_result['errors']:
                errors.extend(field_result['errors'])
            if field_result['warnings']:
                warnings.extend(field_result['warnings'])

        # Validate overall structure
        structure_result = self._validate_structure(example)
        if structure_result['errors']:
            errors.extend(structure_result['errors'])
        if structure_result['warnings']:
            warnings.extend(structure_result['warnings'])

        # Validate metadata
        metadata_result = self._validate_metadata(example)
        if metadata_result['errors']:
            errors.extend(metadata_result['errors'])
        if metadata_result['warnings']:
            warnings.extend(metadata_result['warnings'])

        # Calculate overall quality score
        quality_score = self._calculate_overall_quality_score(field_scores, errors, warnings)

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            field_scores=field_scores
        )

    def _validate_field(self, field_name: str, example: TrainingExample) -> Dict[str, Any]:
        """Validate a specific field."""
        errors = []
        warnings = []
        score = 1.0

        # Get field value from actual output
        actual_value = example.actual_output.get(field_name, "")
        expected_value = example.expected_output.get(field_name, "")
        confidence = example.confidence_scores.get(field_name, 0.0)

        # Get validation rules for this field
        rules = self.validation_rules.get(field_name, {})

        # Required field check
        if rules.get('required', False) and not actual_value:
            errors.append(f"Field '{field_name}' is required but empty")
            score -= 0.5

        # Skip further validation if field is empty and not required
        if not actual_value and not rules.get('required', False):
            return {'errors': errors, 'warnings': warnings, 'score': max(0.0, score)}

        # Pattern validation
        if 'pattern' in rules:
            if not re.match(rules['pattern'], str(actual_value)):
                errors.append(f"Field '{field_name}' does not match required pattern: {rules['description']}")
                score -= 0.3

        # Length validation
        if 'min_length' in rules:
            if len(str(actual_value)) < rules['min_length']:
                errors.append(f"Field '{field_name}' is too short (min: {rules['min_length']})")
                score -= 0.2

        if 'max_length' in rules:
            if len(str(actual_value)) > rules['max_length']:
                warnings.append(f"Field '{field_name}' is very long (max recommended: {rules['max_length']})")
                score -= 0.1

        # Date format validation
        if rules.get('date_format', False):
            if not self._validate_date_format(actual_value):
                errors.append(f"Field '{field_name}' has invalid date format")
                score -= 0.3

        # Time format validation
        if rules.get('time_format', False):
            if not self._validate_time_format(actual_value):
                errors.append(f"Field '{field_name}' has invalid time format")
                score -= 0.2

        # Structure validation for complex fields
        if rules.get('structure_check', False):
            struct_result = self._validate_complainant_structure(actual_value)
            if struct_result['errors']:
                errors.extend([f"{field_name}: {err}" for err in struct_result['errors']])
                score -= 0.2
            if struct_result['warnings']:
                warnings.extend([f"{field_name}: {warn}" for warn in struct_result['warnings']])
                score -= 0.1

        # List validation
        if rules.get('list_check', False):
            list_result = self._validate_list_structure(actual_value)
            if list_result['errors']:
                errors.extend([f"{field_name}: {err}" for err in list_result['errors']])
                score -= 0.2

        # Confidence-based scoring
        if confidence < 0.5:
            warnings.append(f"Field '{field_name}' has low confidence score: {confidence:.3f}")
            score -= 0.2

        # Consistency check between expected and actual
        if expected_value and actual_value != expected_value:
            if confidence > 0.8:
                warnings.append(f"Field '{field_name}' differs from expected value despite high confidence")
                score -= 0.1

        return {
            'errors': errors,
            'warnings': warnings,
            'score': max(0.0, min(1.0, score))
        }

    def _validate_structure(self, example: TrainingExample) -> Dict[str, Any]:
        """Validate overall structure of the training example."""
        errors = []
        warnings = []

        # Check if at least one key field is present
        key_fields = ['cr_no', 'police_station', 'complainant']
        has_key_field = any(
            example.actual_output.get(field) for field in key_fields
        )

        if not has_key_field:
            errors.append("No key identification fields found (CR No, Police Station, or Complainant)")

        # Check for reasonable data size
        total_text_length = sum(
            len(str(value)) for value in example.actual_output.values()
        )

        if total_text_length < 10:
            errors.append("Extracted data is too short - possible parsing failure")
        elif total_text_length > 10000:
            warnings.append("Extracted data is very long - possible data concatenation issue")

        # Check confidence scores consistency
        confidence_values = list(example.confidence_scores.values())
        if confidence_values:
            avg_confidence = sum(confidence_values) / len(confidence_values)
            if avg_confidence < 0.3:
                errors.append(f"Overall confidence too low: {avg_confidence:.3f}")
            elif avg_confidence < 0.6:
                warnings.append(f"Overall confidence is moderate: {avg_confidence:.3f}")

        return {'errors': errors, 'warnings': warnings}

    def _validate_metadata(self, example: TrainingExample) -> Dict[str, Any]:
        """Validate metadata of the training example."""
        errors = []
        warnings = []

        # Check timestamp
        if not example.timestamp:
            errors.append("Missing timestamp")
        else:
            try:
                datetime.fromisoformat(example.timestamp.replace('Z', '+00:00'))
            except ValueError:
                errors.append("Invalid timestamp format")

        # Check data hash
        if not example.data_hash:
            warnings.append("Missing data hash for integrity checking")

        # Check processing metadata
        if not example.processing_metadata:
            warnings.append("Missing processing metadata")
        else:
            required_meta = ['ml_enabled', 'ner_processing_time']
            for meta_field in required_meta:
                if meta_field not in example.processing_metadata:
                    warnings.append(f"Missing processing metadata field: {meta_field}")

        return {'errors': errors, 'warnings': warnings}

    def _validate_date_format(self, date_value: Any) -> bool:
        """Validate date format."""
        if not date_value:
            return True  # Empty is acceptable for optional dates

        date_str = str(date_value)

        # Common date patterns
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY or DD/MM/YYYY
            r'^\d{1,2}-\d{1,2}-\d{4}$',  # MM-DD-YYYY or DD-MM-YYYY
            r'^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$',  # DD Mon YYYY
        ]

        return any(re.match(pattern, date_str) for pattern in date_patterns)

    def _validate_time_format(self, time_value: Any) -> bool:
        """Validate time format."""
        if not time_value:
            return True  # Empty is acceptable for optional times

        time_str = str(time_value)

        # Time patterns
        time_patterns = [
            r'^\d{1,2}:\d{2}\s*(?:AM|PM|hrs)?$',  # 12-hour format
            r'^\d{1,2}:\d{2}$',  # 24-hour format
            r'^\d{1,2}\.\d{2}\s*hrs?$',  # Decimal hours
        ]

        return any(re.match(pattern, time_str, re.IGNORECASE) for pattern in time_patterns)

    def _validate_complainant_structure(self, complainant_value: Any) -> Dict[str, Any]:
        """Validate complainant structure."""
        errors = []
        warnings = []

        if not isinstance(complainant_value, dict):
            errors.append("Complainant should be a dictionary")
            return {'errors': errors, 'warnings': warnings}

        # Check for name
        if not complainant_value.get('name'):
            errors.append("Complainant name is missing")

        # Validate age if present
        age = complainant_value.get('age')
        if age:
            try:
                age_num = int(age)
                if age_num < 1 or age_num > 120:
                    warnings.append(f"Complainant age seems unusual: {age}")
            except (ValueError, TypeError):
                warnings.append(f"Complainant age is not a valid number: {age}")

        # Validate sex if present
        sex = complainant_value.get('sex', '').lower()
        if sex and sex not in ['male', 'female', 'm', 'f']:
            warnings.append(f"Complainant sex value is unusual: {sex}")

        return {'errors': errors, 'warnings': warnings}

    def _validate_list_structure(self, list_value: Any) -> Dict[str, Any]:
        """Validate list structure for victims, accused, property."""
        errors = []
        warnings = []

        if not isinstance(list_value, list):
            errors.append("Field should be a list")
            return {'errors': errors, 'warnings': warnings}

        if len(list_value) == 0:
            return {'errors': errors, 'warnings': warnings}  # Empty list is acceptable

        # Validate each item in the list
        for i, item in enumerate(list_value):
            if not isinstance(item, dict):
                errors.append(f"List item {i} should be a dictionary")
                continue

            # Check for name field in person-related lists
            if 'name' not in item or not item['name']:
                warnings.append(f"List item {i} missing name field")

        return {'errors': errors, 'warnings': warnings}

    def _calculate_overall_quality_score(
        self,
        field_scores: Dict[str, float],
        errors: List[str],
        warnings: List[str]
    ) -> float:
        """Calculate overall quality score."""
        if not field_scores:
            return 0.0

        # Average field scores
        avg_field_score = sum(field_scores.values()) / len(field_scores)

        # Penalize for errors and warnings
        error_penalty = min(0.5, len(errors) * 0.1)
        warning_penalty = min(0.2, len(warnings) * 0.05)

        quality_score = avg_field_score - error_penalty - warning_penalty

        return max(0.0, min(1.0, quality_score))

    def validate_training_dataset(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """
        Validate an entire training dataset.

        Args:
            examples: List of training examples to validate

        Returns:
            Dictionary with validation summary and metrics
        """
        if not examples:
            return {"error": "No examples provided for validation"}

        validation_results = []
        total_quality_score = 0.0
        field_quality_scores = {}

        # Validate each example
        for example in examples:
            result = self.validate_training_example(example)
            validation_results.append(result)
            total_quality_score += result.quality_score

            # Aggregate field scores
            for field, score in result.field_scores.items():
                if field not in field_quality_scores:
                    field_quality_scores[field] = []
                field_quality_scores[field].append(score)

        # Calculate aggregate metrics
        avg_quality_score = total_quality_score / len(examples)

        # Average field quality scores
        avg_field_scores = {}
        for field, scores in field_quality_scores.items():
            avg_field_scores[field] = sum(scores) / len(scores)

        # Count issues
        total_errors = sum(len(result.errors) for result in validation_results)
        total_warnings = sum(len(result.warnings) for result in validation_results)
        valid_examples = sum(1 for result in validation_results if result.is_valid)

        # Calculate quality metrics
        quality_metrics = self._calculate_dataset_quality_metrics(examples, validation_results)

        return {
            "summary": {
                "total_examples": len(examples),
                "valid_examples": valid_examples,
                "invalid_examples": len(examples) - valid_examples,
                "average_quality_score": avg_quality_score,
                "total_errors": total_errors,
                "total_warnings": total_warnings
            },
            "field_quality_scores": avg_field_scores,
            "quality_metrics": quality_metrics,
            "validation_results": [
                {
                    "example_id": example.id,
                    "is_valid": result.is_valid,
                    "quality_score": result.quality_score,
                    "error_count": len(result.errors),
                    "warning_count": len(result.warnings)
                }
                for example, result in zip(examples, validation_results)
            ]
        }

    def _calculate_dataset_quality_metrics(
        self,
        examples: List[TrainingExample],
        validation_results: List[ValidationResult]
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics for the dataset."""

        # Completeness score: based on how many fields are populated
        total_possible_fields = len(self.fir_fields) * len(examples)
        actual_populated_fields = 0

        for example in examples:
            for field in self.fir_fields:
                if example.actual_output.get(field):
                    actual_populated_fields += 1

        completeness_score = actual_populated_fields / total_possible_fields if total_possible_fields > 0 else 0.0

        # Consistency score: based on validation results
        valid_examples = sum(1 for result in validation_results if result.is_valid)
        consistency_score = valid_examples / len(examples) if examples else 0.0

        # Accuracy score: based on average quality scores
        accuracy_score = sum(result.quality_score for result in validation_results) / len(validation_results)

        # Format compliance score: based on format validation errors
        format_errors = sum(
            1 for result in validation_results
            for error in result.errors
            if 'format' in error.lower() or 'pattern' in error.lower()
        )
        format_compliance_score = 1.0 - min(1.0, format_errors / len(examples) * 0.5)

        # Overall score
        overall_score = (completeness_score + consistency_score + accuracy_score + format_compliance_score) / 4.0

        return QualityMetrics(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            format_compliance_score=format_compliance_score,
            overall_score=overall_score
        )

    def identify_pii_fields(self, example: TrainingExample) -> Set[str]:
        """Identify fields that contain personally identifiable information."""
        pii_detected = set()

        # Check complainant information
        complainant = example.actual_output.get('complainant', {})
        if isinstance(complainant, dict):
            if complainant.get('name'):
                pii_detected.add('complainant_name')
            if complainant.get('address'):
                pii_detected.add('complainant_address')

        # Check victims
        victims = example.actual_output.get('victims', [])
        if isinstance(victims, list) and victims:
            for victim in victims:
                if isinstance(victim, dict):
                    if victim.get('name'):
                        pii_detected.add('victim_name')
                    if victim.get('address'):
                        pii_detected.add('victim_address')

        # Check accused
        accused = example.actual_output.get('accused', [])
        if isinstance(accused, list) and accused:
            for acc in accused:
                if isinstance(acc, dict):
                    if acc.get('name'):
                        pii_detected.add('accused_name')
                    if acc.get('address'):
                        pii_detected.add('accused_address')

        return pii_detected

    def generate_validation_report(self, examples: List[TrainingExample]) -> str:
        """Generate a detailed validation report."""
        validation_summary = self.validate_training_dataset(examples)

        report = []
        report.append("=" * 60)
        report.append("TRAINING DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")

        # Summary section
        summary = validation_summary["summary"]
        report.append("SUMMARY:")
        report.append(f"  Total Examples: {summary['total_examples']}")
        report.append(f"  Valid Examples: {summary['valid_examples']}")
        report.append(f"  Invalid Examples: {summary['invalid_examples']}")
        report.append(f"  Average Quality Score: {summary['average_quality_score']:.3f}")
        report.append(f"  Total Errors: {summary['total_errors']}")
        report.append(f"  Total Warnings: {summary['total_warnings']}")
        report.append("")

        # Quality metrics
        metrics = validation_summary["quality_metrics"]
        report.append("QUALITY METRICS:")
        report.append(f"  Completeness Score: {metrics.completeness_score:.3f}")
        report.append(f"  Consistency Score: {metrics.consistency_score:.3f}")
        report.append(f"  Accuracy Score: {metrics.accuracy_score:.3f}")
        report.append(f"  Format Compliance Score: {metrics.format_compliance_score:.3f}")
        report.append(f"  Overall Score: {metrics.overall_score:.3f}")
        report.append("")

        # Field quality scores
        report.append("FIELD QUALITY SCORES:")
        field_scores = validation_summary["field_quality_scores"]
        for field, score in sorted(field_scores.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {field}: {score:.3f}")
        report.append("")

        # Top issues
        all_errors = []
        all_warnings = []
        for result in validation_results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        if all_errors:
            report.append("COMMON ERRORS:")
            error_counts = {}
            for error in all_errors:
                error_counts[error] = error_counts.get(error, 0) + 1

            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"  ({count}) {error}")
            report.append("")

        if all_warnings:
            report.append("COMMON WARNINGS:")
            warning_counts = {}
            for warning in all_warnings:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1

            for warning, count in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"  ({count}) {warning}")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)

# Global validator instance
_data_validator = None

def get_data_validator() -> TrainingDataValidator:
    """Get or create global data validator instance."""
    global _data_validator
    if _data_validator is None:
        _data_validator = TrainingDataValidator()
    return _data_validator

def validate_training_example(example: TrainingExample) -> ValidationResult:
    """Convenience function to validate a training example."""
    validator = get_data_validator()
    return validator.validate_training_example(example)

def validate_training_dataset(examples: List[TrainingExample]) -> Dict[str, Any]:
    """Convenience function to validate a training dataset."""
    validator = get_data_validator()
    return validator.validate_training_dataset(examples)