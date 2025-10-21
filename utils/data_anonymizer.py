"""
Data Anonymization for Privacy Compliance

This module provides comprehensive data anonymization capabilities for
FIR document training data, ensuring privacy compliance while preserving
data utility for machine learning.
"""

import re
import json
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import logging

# Import data collector and validator
from utils.data_collector import TrainingExample, get_data_collector
from utils.data_validator import get_data_validator

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AnonymizationRule:
    """Rule for anonymizing specific types of PII."""
    field_name: str
    pii_type: str
    anonymization_method: str  # 'hash', 'replace', 'mask', 'tokenize'
    pattern: str = ""
    replacement: str = ""
    preserve_structure: bool = True

@dataclass
class AnonymizationResult:
    """Result of data anonymization."""
    original_example_id: str
    anonymized_example_id: str
    pii_fields_detected: Set[str]
    anonymization_methods_applied: Dict[str, str]
    success: bool
    error_message: str = ""

class DataAnonymizer:
    """
    Comprehensive data anonymization system for privacy compliance.
    """

    def __init__(self, preserve_utility: bool = True):
        """
        Initialize the data anonymizer.

        Args:
            preserve_utility: Whether to preserve data utility for ML training
        """
        self.preserve_utility = preserve_utility

        # Define PII detection patterns
        self.pii_patterns = self._load_pii_patterns()

        # Define anonymization rules
        self.anonymization_rules = self._load_anonymization_rules()

        # Track anonymized data
        self.anonymization_map = {}  # Maps original values to anonymized values

        logger.info("DataAnonymizer initialized")

    def _load_pii_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for detecting personally identifiable information."""
        return {
            'person_names': [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Multi-word names
                r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b',  # Names with initials
                r'\b[A-Z]+\s+[A-Z][a-z]+\b',  # Surname + name pattern
            ],
            'addresses': [
                r'\b\d+\s*,?\s*[A-Z][a-z\s]+(?:Road|Street|Avenue|Lane|Nagar|Colony|Village|District|City)\b',
                r'\b[A-Z][a-z\s]+(?:Road|Street|Avenue|Lane|Nagar|Colony|Village|District|City)\s+\d+\b',
                r'\bH\.?No\.?\s*:?\s*\d+[A-Z]?\b',  # House numbers
            ],
            'phone_numbers': [
                r'\b\d{10}\b',  # 10-digit mobile numbers
                r'\b\d{3,5}[-\s]\d{6,8}\b',  # Landline numbers
                r'\b\d{2}[-\s]\d{4}[-\s]\d{4}\b',  # Various formats
            ],
            'aadhar_numbers': [
                r'\b\d{4}\s*\d{4}\s*\d{4}\b',  # Aadhar format
            ],
            'vehicle_numbers': [
                r'\b[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}\b',  # Indian vehicle registration
            ],
            'bank_accounts': [
                r'\b\d{9,18}\b',  # Bank account numbers
            ]
        }

    def _load_anonymization_rules(self) -> List[AnonymizationRule]:
        """Load default anonymization rules."""
        return [
            AnonymizationRule(
                field_name="complainant_name",
                pii_type="person_name",
                anonymization_method="replace",
                replacement="[COMPLAINANT_NAME]",
                preserve_structure=True
            ),
            AnonymizationRule(
                field_name="victim_name",
                pii_type="person_name",
                anonymization_method="replace",
                replacement="[VICTIM_NAME]",
                preserve_structure=True
            ),
            AnonymizationRule(
                field_name="accused_name",
                pii_type="person_name",
                anonymization_method="replace",
                replacement="[ACCUSED_NAME]",
                preserve_structure=True
            ),
            AnonymizationRule(
                field_name="complainant_address",
                pii_type="address",
                anonymization_method="hash",
                preserve_structure=False
            ),
            AnonymizationRule(
                field_name="victim_address",
                pii_type="address",
                anonymization_method="hash",
                preserve_structure=False
            ),
            AnonymizationRule(
                field_name="accused_address",
                pii_type="address",
                anonymization_method="hash",
                preserve_structure=False
            ),
            AnonymizationRule(
                field_name="phone_numbers",
                pii_type="phone",
                anonymization_method="mask",
                replacement="XXX-XXX-XXXX",
                preserve_structure=True
            ),
            AnonymizationRule(
                field_name="aadhar_numbers",
                pii_type="id_number",
                anonymization_method="mask",
                replacement="XXXX-XXXX-XXXX",
                preserve_structure=True
            )
        ]

    def detect_pii_in_text(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII in text using pattern matching.

        Args:
            text: Text to analyze for PII

        Returns:
            Dictionary mapping PII types to detected values
        """
        detected_pii = {}

        for pii_type, patterns in self.pii_patterns.items():
            detected_values = []

            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                detected_values.extend(matches)

            if detected_values:
                detected_pii[pii_type] = list(set(detected_values))  # Remove duplicates

        return detected_pii

    def detect_pii_in_example(self, example: TrainingExample) -> Set[str]:
        """
        Detect PII fields in a training example.

        Args:
            example: Training example to analyze

        Returns:
            Set of field names containing PII
        """
        pii_fields = set()

        # Check each field in the actual output
        for field_name, field_value in example.actual_output.items():
            if self._field_contains_pii(field_name, field_value):
                pii_fields.add(field_name)

        # Also check input text for PII that might not be in structured fields
        input_pii = self.detect_pii_in_text(example.extraction_input)
        if input_pii:
            pii_fields.add("input_text")

        return pii_fields

    def _field_contains_pii(self, field_name: str, field_value: Any) -> bool:
        """Check if a specific field contains PII."""
        if not field_value:
            return False

        field_str = str(field_value)

        # Check against known PII patterns
        for pii_type, patterns in self.pii_patterns.items():
            for pattern in patterns:
                if re.search(pattern, field_str, re.IGNORECASE):
                    return True

        # Special checks for structured fields
        if field_name in ['complainant', 'victims', 'accused']:
            if isinstance(field_value, dict):
                for key, value in field_value.items():
                    if key in ['name', 'address'] and value:
                        return True
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if key in ['name', 'address'] and value:
                                return True

        return False

    def anonymize_example(self, example: TrainingExample, anonymization_level: str = "standard") -> Tuple[TrainingExample, AnonymizationResult]:
        """
        Anonymize a training example.

        Args:
            example: Training example to anonymize
            anonymization_level: Level of anonymization ('minimal', 'standard', 'aggressive')

        Returns:
            Tuple of (anonymized_example, anonymization_result)
        """
        try:
            # Create a copy of the example
            anonymized_example = TrainingExample(
                id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                source_file=example.source_file,
                extraction_input=self._anonymize_text(example.extraction_input, anonymization_level),
                expected_output=self._anonymize_output(example.expected_output, anonymization_level),
                actual_output=self._anonymize_output(example.actual_output, anonymization_level),
                confidence_scores=example.confidence_scores.copy(),
                processing_metadata=example.processing_metadata.copy(),
                annotations=example.annotations.copy(),
                data_quality_score=example.data_quality_score,
                is_corrected=example.is_corrected,
                correction_notes=example.correction_notes,
                model_version=example.model_version,
                data_hash=""  # Will be recalculated
            )

            # Recalculate data hash
            data_to_hash = anonymized_example.source_file + anonymized_example.extraction_input + json.dumps(anonymized_example.actual_output)
            anonymized_example.data_hash = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

            # Detect PII fields
            pii_fields = self.detect_pii_in_example(example)

            # Record anonymization methods applied
            methods_applied = {}
            for field in pii_fields:
                methods_applied[field] = "text_anonymization"

            result = AnonymizationResult(
                original_example_id=example.id,
                anonymized_example_id=anonymized_example.id,
                pii_fields_detected=pii_fields,
                anonymization_methods_applied=methods_applied,
                success=True
            )

            logger.info(f"Anonymized example {example.id} -> {anonymized_example.id}")
            return anonymized_example, result

        except Exception as e:
            logger.error(f"Error anonymizing example {example.id}: {e}")
            return example, AnonymizationResult(
                original_example_id=example.id,
                anonymized_example_id="",
                pii_fields_detected=set(),
                anonymization_methods_applied={},
                success=False,
                error_message=str(e)
            )

    def _anonymize_text(self, text: str, anonymization_level: str = "standard") -> str:
        """Anonymize PII in text."""
        if not text:
            return text

        anonymized_text = text

        # Apply different levels of anonymization
        if anonymization_level in ["standard", "aggressive"]:
            # Anonymize person names
            for pattern in self.pii_patterns['person_names']:
                matches = re.findall(pattern, anonymized_text)
                for match in matches:
                    if len(match.split()) >= 2:  # Likely a real name
                        anonymized_text = re.sub(
                            re.escape(match),
                            "[PERSON_NAME]",
                            anonymized_text,
                            flags=re.IGNORECASE
                        )

        if anonymization_level == "aggressive":
            # More aggressive anonymization
            for pii_type, patterns in self.pii_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, anonymized_text)
                    for match in matches:
                        replacement = f"[{pii_type.upper()}]"
                        anonymized_text = re.sub(
                            re.escape(match),
                            replacement,
                            anonymized_text,
                            flags=re.IGNORECASE
                        )

        return anonymized_text

    def _anonymize_output(self, output: Dict[str, Any], anonymization_level: str = "standard") -> Dict[str, Any]:
        """Anonymize PII in structured output."""
        if not output:
            return output

        anonymized_output = {}

        for field_name, field_value in output.items():
            if field_name in ['complainant', 'victims', 'accused']:
                anonymized_output[field_name] = self._anonymize_person_data(field_value, anonymization_level)
            elif field_name in ['complainant_address', 'victim_address', 'accused_address']:
                anonymized_output[field_name] = self._anonymize_address(field_value)
            elif field_name == 'phone_numbers':
                anonymized_output[field_name] = self._anonymize_phone_numbers(field_value)
            else:
                # For other fields, check if they contain PII
                if isinstance(field_value, str):
                    anonymized_output[field_name] = self._anonymize_text(field_value, anonymization_level)
                else:
                    anonymized_output[field_name] = field_value

        return anonymized_output

    def _anonymize_person_data(self, person_data: Any, anonymization_level: str = "standard") -> Any:
        """Anonymize person-related data."""
        if not person_data:
            return person_data

        if isinstance(person_data, dict):
            anonymized_person = {}
            for key, value in person_data.items():
                if key == 'name':
                    anonymized_person[key] = "[PERSON_NAME]" if value else value
                elif key == 'address':
                    anonymized_person[key] = self._anonymize_address(value)
                else:
                    anonymized_person[key] = value
            return anonymized_person

        elif isinstance(person_data, list):
            return [
                self._anonymize_person_data(item, anonymization_level)
                for item in person_data
            ]

        return person_data

    def _anonymize_address(self, address: Any) -> Any:
        """Anonymize address information."""
        if not address:
            return address

        # Hash the address to preserve uniqueness while removing PII
        address_str = str(address)
        return hashlib.sha256(address_str.encode('utf-8')).hexdigest()[:16]

    def _anonymize_phone_numbers(self, phone_data: Any) -> Any:
        """Anonymize phone number data."""
        if not phone_data:
            return phone_data

        if isinstance(phone_data, str):
            return "XXX-XXX-XXXX"
        elif isinstance(phone_data, list):
            return ["XXX-XXX-XXXX" for _ in phone_data]

        return phone_data

    def anonymize_dataset(
        self,
        examples: List[TrainingExample],
        anonymization_level: str = "standard",
        output_path: str = "models/training_data/anonymized"
    ) -> Dict[str, Any]:
        """
        Anonymize an entire dataset.

        Args:
            examples: List of examples to anonymize
            anonymization_level: Level of anonymization
            output_path: Path to save anonymized data

        Returns:
            Dictionary with anonymization summary
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        anonymized_examples = []
        anonymization_results = []
        total_pii_fields = set()

        for example in examples:
            anonymized_example, result = self.anonymize_example(example, anonymization_level)

            if result.success:
                anonymized_examples.append(anonymized_example)
                total_pii_fields.update(result.pii_fields_detected)

            anonymization_results.append(result)

        # Save anonymized examples
        anonymized_file = output_dir / "anonymized_examples.jsonl"
        with open(anonymized_file, 'w', encoding='utf-8') as f:
            for example in anonymized_examples:
                json.dump(example.__dict__, f, ensure_ascii=False)
                f.write('\n')

        # Save anonymization report
        report = {
            "anonymization_timestamp": datetime.now().isoformat(),
            "anonymization_level": anonymization_level,
            "total_examples": len(examples),
            "successfully_anonymized": len(anonymized_examples),
            "failed_anonymizations": len(examples) - len(anonymized_examples),
            "pii_fields_detected": list(total_pii_fields),
            "anonymization_methods": list(set(
                method for result in anonymization_results
                for method in result.anonymization_methods_applied.values()
            )),
            "results": [
                {
                    "original_id": result.original_example_id,
                    "anonymized_id": result.anonymized_example_id,
                    "pii_fields": list(result.pii_fields_detected),
                    "success": result.success
                }
                for result in anonymization_results
            ]
        }

        report_file = output_dir / "anonymization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Anonymized {len(anonymized_examples)}/{len(examples)} examples")
        return {
            "anonymized_file": str(anonymized_file),
            "report_file": str(report_file),
            "summary": report
        }

    def create_privacy_compliant_export(
        self,
        examples: List[TrainingExample],
        export_path: str,
        anonymization_level: str = "standard",
        model_type: str = "extraction"
    ) -> Dict[str, str]:
        """
        Create a privacy-compliant export of training data.

        Args:
            examples: Examples to export
            export_path: Path for the export
            anonymization_level: Level of anonymization
            model_type: Type of model data to prepare

        Returns:
            Dictionary with paths to exported files
        """
        # First anonymize the data
        anonymization_result = self.anonymize_dataset(
            examples,
            anonymization_level,
            output_path=f"{export_path}_anonymized"
        )

        # Load anonymized examples
        anonymized_examples = []
        anonymized_file = Path(anonymization_result["anonymized_file"])

        if anonymized_file.exists():
            with open(anonymized_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        example = TrainingExample(**data)
                        anonymized_examples.append(example)

        # Export anonymized data for model training
        from utils.data_exporter import get_data_exporter

        exporter = get_data_exporter()
        export_result = exporter.export_for_model_training(
            output_path=export_path,
            model_type=model_type,
            quality_threshold=0.0  # Include all anonymized examples
        )

        # Add privacy compliance metadata
        compliance_metadata = {
            "privacy_compliant": True,
            "anonymization_level": anonymization_level,
            "anonymization_timestamp": datetime.now().isoformat(),
            "original_examples": len(examples),
            "anonymized_examples": len(anonymized_examples),
            "pii_detection_summary": anonymization_result["summary"]
        }

        metadata_file = Path(export_path) / "privacy_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(compliance_metadata, f, ensure_ascii=False, indent=2)

        export_result["privacy_metadata"] = str(metadata_file)
        export_result["anonymization_report"] = anonymization_result["report_file"]

        logger.info(f"Created privacy-compliant export with {len(anonymized_examples)} anonymized examples")
        return export_result

    def generate_privacy_impact_assessment(self, examples: List[TrainingExample]) -> str:
        """Generate a privacy impact assessment report."""
        # Analyze PII in the dataset
        pii_analysis = self._analyze_dataset_pii(examples)

        # Generate assessment report
        report = []
        report.append("=" * 60)
        report.append("PRIVACY IMPACT ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("DATASET OVERVIEW:")
        report.append(f"  Total Examples: {len(examples)}")
        report.append(f"  Examples with PII: {pii_analysis['examples_with_pii']}")
        report.append(f"  PII Detection Rate: {pii_analysis['pii_rate']:.2%}")
        report.append("")

        report.append("PII TYPES DETECTED:")
        for pii_type, count in pii_analysis['pii_types'].items():
            report.append(f"  {pii_type}: {count} examples")
        report.append("")

        report.append("RECOMMENDED ANONYMIZATION:")
        report.append(f"  Suggested Level: {pii_analysis['recommended_anonymization_level']}")
        report.append("  Fields Requiring Anonymization:")
        for field in pii_analysis['critical_pii_fields']:
            report.append(f"    - {field}")
        report.append("")

        report.append("PRIVACY RISKS:")
        if pii_analysis['high_risk_examples'] > 0:
            report.append(f"  ⚠️  High Risk: {pii_analysis['high_risk_examples']} examples contain multiple PII types")
        if pii_analysis['pii_rate'] > 0.8:
            report.append("  ⚠️  High PII Prevalence: Most examples contain personal information")
        report.append("")

        report.append("COMPLIANCE RECOMMENDATIONS:")
        report.append("  1. Apply appropriate anonymization before model training")
        report.append("  2. Implement data minimization principles")
        report.append("  3. Ensure anonymized data cannot be re-identified")
        report.append("  4. Regular privacy audits of training data")
        report.append("  5. Document all data processing activities")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def _analyze_dataset_pii(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Analyze PII prevalence in the dataset."""
        total_examples = len(examples)
        examples_with_pii = 0
        pii_types = {}
        critical_pii_fields = set()
        high_risk_examples = 0

        for example in examples:
            pii_fields = self.detect_pii_in_example(example)

            if pii_fields:
                examples_with_pii += 1

                # Count PII types
                for field in pii_fields:
                    pii_types[field] = pii_types.get(field, 0) + 1

                    # Track critical PII fields
                    if field in ['complainant_name', 'victim_name', 'accused_name']:
                        critical_pii_fields.add(field)

                # Count high-risk examples (multiple PII types)
                if len(pii_fields) > 2:
                    high_risk_examples += 1

        pii_rate = examples_with_pii / total_examples if total_examples > 0 else 0.0

        # Determine recommended anonymization level
        if pii_rate > 0.8:
            recommended_level = "aggressive"
        elif pii_rate > 0.5:
            recommended_level = "standard"
        else:
            recommended_level = "minimal"

        return {
            "total_examples": total_examples,
            "examples_with_pii": examples_with_pii,
            "pii_rate": pii_rate,
            "pii_types": pii_types,
            "critical_pii_fields": list(critical_pii_fields),
            "high_risk_examples": high_risk_examples,
            "recommended_anonymization_level": recommended_level
        }

# Global anonymizer instance
_data_anonymizer = None

def get_data_anonymizer(preserve_utility: bool = True) -> DataAnonymizer:
    """Get or create global data anonymizer instance."""
    global _data_anonymizer
    if _data_anonymizer is None:
        _data_anonymizer = DataAnonymizer(preserve_utility)
    return _data_anonymizer

def anonymize_training_example(example: TrainingExample, anonymization_level: str = "standard") -> Tuple[TrainingExample, AnonymizationResult]:
    """Convenience function to anonymize a training example."""
    anonymizer = get_data_anonymizer()
    return anonymizer.anonymize_example(example, anonymization_level)

def create_privacy_compliant_dataset(
    examples: List[TrainingExample],
    export_path: str,
    anonymization_level: str = "standard"
) -> Dict[str, str]:
    """Convenience function to create a privacy-compliant dataset."""
    anonymizer = get_data_anonymizer()
    return anonymizer.create_privacy_compliant_export(examples, export_path, anonymization_level)