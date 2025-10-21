"""
Training Data Collection Framework for ML Model Training

This module provides comprehensive data collection capabilities for training
ML models on FIR document processing. It handles extraction result logging,
data validation, quality checks, and structured storage for training data.
"""

import json
import os
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """Represents a single training example for ML models."""
    id: str
    timestamp: str
    source_file: str
    extraction_input: str
    expected_output: Dict[str, Any]
    actual_output: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_metadata: Dict[str, Any]
    annotations: List[Dict[str, Any]]
    data_quality_score: float
    is_corrected: bool = False
    correction_notes: str = ""
    model_version: str = "1.0"
    data_hash: str = ""

@dataclass
class DataCollectionStats:
    """Statistics for data collection process."""
    total_examples: int = 0
    corrected_examples: int = 0
    average_quality_score: float = 0.0
    entity_type_distribution: Dict[str, int] = None
    collection_start_time: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if self.entity_type_distribution is None:
            self.entity_type_distribution = {}

class TrainingDataCollector:
    """
    Main class for collecting and managing training data for ML models.
    """

    def __init__(self, storage_path: str = "models/training_data", auto_save: bool = True):
        """
        Initialize the training data collector.

        Args:
            storage_path: Directory path for storing training data
            auto_save: Whether to automatically save data after collection
        """
        self.storage_path = Path(storage_path)
        self.auto_save = auto_save
        self.examples: List[TrainingExample] = []
        self.stats = DataCollectionStats()
        self._current_session_id = str(uuid.uuid4())

        # Create storage directories
        self._create_storage_structure()

        # Load existing data
        self._load_existing_data()

        logger.info(f"TrainingDataCollector initialized with storage path: {storage_path}")

    def _create_storage_structure(self) -> None:
        """Create the necessary directory structure for training data."""
        directories = [
            self.storage_path,
            self.storage_path / "raw_examples",
            self.storage_path / "annotations",
            self.storage_path / "models",
            self.storage_path / "exports",
            self.storage_path / "metadata"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_existing_data(self) -> None:
        """Load existing training data from storage."""
        examples_file = self.storage_path / "training_examples.jsonl"

        if examples_file.exists():
            try:
                with open(examples_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            example = TrainingExample(**data)
                            self.examples.append(example)

                logger.info(f"Loaded {len(self.examples)} existing training examples")
                self._update_stats()

            except Exception as e:
                logger.error(f"Error loading existing training data: {e}")

    def collect_extraction_result(
        self,
        source_file: str,
        extraction_input: str,
        expected_output: Dict[str, Any],
        actual_output: Dict[str, Any],
        confidence_scores: Dict[str, float],
        processing_metadata: Dict[str, Any],
        model_version: str = "1.0"
    ) -> str:
        """
        Collect a training example from extraction results.

        Args:
            source_file: Path to the source FIR document
            extraction_input: Input text that was processed
            expected_output: Ground truth/expected extraction results
            actual_output: Actual extraction results from the model
            confidence_scores: Confidence scores for each extracted field
            processing_metadata: Additional processing information
            model_version: Version of the model that produced the results

        Returns:
            ID of the collected training example
        """
        # Calculate data quality score
        quality_score = self._calculate_data_quality(
            expected_output,
            actual_output,
            confidence_scores
        )

        # Generate unique ID and hash
        example_id = str(uuid.uuid4())
        data_hash = self._calculate_data_hash(
            source_file + extraction_input + json.dumps(actual_output)
        )

        # Create training example
        example = TrainingExample(
            id=example_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_file=source_file,
            extraction_input=extraction_input,
            expected_output=expected_output,
            actual_output=actual_output,
            confidence_scores=confidence_scores,
            processing_metadata=processing_metadata,
            annotations=[],
            data_quality_score=quality_score,
            model_version=model_version,
            data_hash=data_hash
        )

        # Add to collection
        self.examples.append(example)

        # Update statistics
        self._update_stats()

        # Auto-save if enabled
        if self.auto_save:
            self.save_example(example)

        logger.info(f"Collected training example {example_id} with quality score {quality_score:.3f}")
        return example_id

    def _calculate_data_quality(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        confidence_scores: Dict[str, float]
    ) -> float:
        """
        Calculate quality score for the training example.

        Args:
            expected: Expected extraction results
            actual: Actual extraction results
            confidence_scores: Confidence scores for extracted fields

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not expected or not actual:
            return 0.0

        quality_score = 0.0
        total_fields = 0

        # Compare each field
        all_fields = set(expected.keys()) | set(actual.keys())

        for field in all_fields:
            total_fields += 1

            expected_value = expected.get(field, "")
            actual_value = actual.get(field, "")

            # Field-level comparison
            if expected_value == actual_value:
                field_score = 1.0
            elif not expected_value and not actual_value:
                field_score = 1.0  # Both empty is correct
            elif not expected_value or not actual_value:
                field_score = 0.0  # One empty, one not
            else:
                # Partial credit for similar values
                field_score = self._calculate_field_similarity(expected_value, actual_value)

            # Weight by confidence if available
            confidence = confidence_scores.get(field, 0.5)
            weighted_score = field_score * confidence

            quality_score += weighted_score

        # Normalize by total fields
        if total_fields > 0:
            quality_score /= total_fields

        return max(0.0, min(1.0, quality_score))

    def _calculate_field_similarity(self, expected: str, actual: str) -> float:
        """
        Calculate similarity between expected and actual field values.

        Args:
            expected: Expected field value
            actual: Actual field value

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if expected == actual:
            return 1.0

        if not expected or not actual:
            return 0.0

        # Simple string similarity (can be enhanced with more sophisticated methods)
        expected_lower = str(expected).lower()
        actual_lower = str(actual).lower()

        # Check for partial matches
        if expected_lower in actual_lower or actual_lower in expected_lower:
            return 0.7

        # Check for word-level overlap
        expected_words = set(expected_lower.split())
        actual_words = set(actual_lower.split())

        if expected_words and actual_words:
            overlap = len(expected_words & actual_words)
            total = len(expected_words | actual_words)
            return overlap / total if total > 0 else 0.0

        return 0.0

    def _calculate_data_hash(self, data: str) -> str:
        """Calculate hash for data integrity checking."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _update_stats(self) -> None:
        """Update collection statistics."""
        self.stats.total_examples = len(self.examples)
        self.stats.corrected_examples = sum(1 for ex in self.examples if ex.is_corrected)

        if self.examples:
            self.stats.average_quality_score = sum(
                ex.data_quality_score for ex in self.examples
            ) / len(self.examples)

            # Update entity type distribution
            entity_counts = {}
            for example in self.examples:
                for field in example.actual_output.keys():
                    entity_counts[field] = entity_counts.get(field, 0) + 1
            self.stats.entity_type_distribution = entity_counts

        self.stats.last_updated = datetime.now(timezone.utc).isoformat()

        if not self.stats.collection_start_time:
            self.stats.collection_start_time = datetime.now(timezone.utc).isoformat()

    def save_example(self, example: TrainingExample) -> None:
        """Save a single training example to storage."""
        try:
            examples_file = self.storage_path / "training_examples.jsonl"

            with open(examples_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(example), f, ensure_ascii=False)
                f.write('\n')

        except Exception as e:
            logger.error(f"Error saving training example {example.id}: {e}")

    def save_all_examples(self) -> None:
        """Save all training examples to storage."""
        try:
            examples_file = self.storage_path / "training_examples.jsonl"

            with open(examples_file, 'w', encoding='utf-8') as f:
                for example in self.examples:
                    json.dump(asdict(example), f, ensure_ascii=False)
                    f.write('\n')

            # Save statistics
            stats_file = self.storage_path / "metadata" / "collection_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.stats), f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(self.examples)} training examples to storage")

        except Exception as e:
            logger.error(f"Error saving training examples: {e}")

    def get_examples_by_quality_range(self, min_quality: float, max_quality: float) -> List[TrainingExample]:
        """Get examples within a quality score range."""
        return [
            ex for ex in self.examples
            if min_quality <= ex.data_quality_score <= max_quality
        ]

    def get_examples_by_entity_type(self, entity_type: str) -> List[TrainingExample]:
        """Get examples that contain a specific entity type."""
        return [
            ex for ex in self.examples
            if entity_type in ex.actual_output or entity_type in ex.expected_output
        ]

    def get_examples_needing_correction(self, threshold: float = 0.8) -> List[TrainingExample]:
        """Get examples that likely need manual correction."""
        return [
            ex for ex in self.examples
            if ex.data_quality_score < threshold and not ex.is_corrected
        ]

    def export_for_training(
        self,
        output_path: str,
        format: str = "json",
        include_corrected_only: bool = False,
        quality_threshold: float = 0.0
    ) -> str:
        """
        Export training data in various formats for model training.

        Args:
            output_path: Path for the exported file
            format: Export format ('json', 'jsonl', 'csv')
            include_corrected_only: Whether to include only corrected examples
            quality_threshold: Minimum quality score to include

        Returns:
            Path to the exported file
        """
        # Filter examples
        filtered_examples = self.examples

        if include_corrected_only:
            filtered_examples = [ex for ex in filtered_examples if ex.is_corrected]

        if quality_threshold > 0.0:
            filtered_examples = [
                ex for ex in filtered_examples
                if ex.data_quality_score >= quality_threshold
            ]

        # Prepare export data
        export_data = []
        for example in filtered_examples:
            # Create training instance
            training_instance = {
                "input": example.extraction_input,
                "expected_output": example.expected_output,
                "actual_output": example.actual_output,
                "confidence_scores": example.confidence_scores,
                "metadata": {
                    "id": example.id,
                    "quality_score": example.data_quality_score,
                    "model_version": example.model_version,
                    "timestamp": example.timestamp
                }
            }
            export_data.append(training_instance)

        # Export based on format
        output_file = Path(output_path)

        try:
            if format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

            elif format.lower() == "jsonl":
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in export_data:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')

            elif format.lower() == "csv":
                import csv

                if export_data:
                    # Flatten the data for CSV format
                    fieldnames = ["id", "input", "quality_score"]
                    fieldnames.extend(list(export_data[0]["expected_output"].keys()))
                    fieldnames.extend([f"confidence_{k}" for k in export_data[0]["confidence_scores"].keys()])

                    with open(output_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

                        for example in export_data:
                            row = {
                                "id": example["metadata"]["id"],
                                "input": example["input"],
                                "quality_score": example["metadata"]["quality_score"]
                            }

                            # Add expected outputs
                            for key, value in example["expected_output"].items():
                                row[key] = str(value) if value else ""

                            # Add confidence scores
                            for key, value in example["confidence_scores"].items():
                                row[f"confidence_{key}"] = str(value) if value else ""

                            writer.writerow(row)

            logger.info(f"Exported {len(export_data)} training examples to {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            raise

    def get_collection_summary(self) -> Dict[str, Any]:
        """Get a summary of the current data collection."""
        return {
            "total_examples": self.stats.total_examples,
            "corrected_examples": self.stats.corrected_examples,
            "average_quality_score": self.stats.average_quality_score,
            "entity_type_distribution": self.stats.entity_type_distribution,
            "collection_start_time": self.stats.collection_start_time,
            "last_updated": self.stats.last_updated,
            "session_id": self._current_session_id
        }

# Global instance for easy access
_data_collector = None

def get_data_collector(storage_path: str = "models/training_data") -> TrainingDataCollector:
    """Get or create global data collector instance."""
    global _data_collector
    if _data_collector is None:
        _data_collector = TrainingDataCollector(storage_path)
    return _data_collector

def collect_training_example(
    source_file: str,
    extraction_input: str,
    expected_output: Dict[str, Any],
    actual_output: Dict[str, Any],
    confidence_scores: Dict[str, float],
    processing_metadata: Dict[str, Any],
    model_version: str = "1.0"
) -> str:
    """
    Convenience function to collect a training example.

    Returns:
        ID of the collected training example
    """
    collector = get_data_collector()
    return collector.collect_extraction_result(
        source_file, extraction_input, expected_output,
        actual_output, confidence_scores, processing_metadata, model_version
    )