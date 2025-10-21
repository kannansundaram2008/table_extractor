"""
Data Export Functionality for Model Training

This module provides comprehensive data export capabilities for preparing
training data in various formats suitable for different ML models and frameworks.
"""

import json
import csv
import pickle
import gzip
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

# Import data collector and validator
from utils.data_collector import TrainingExample, get_data_collector
from utils.data_validator import get_data_validator

# Configure logging
logger = logging.getLogger(__name__)

class TrainingDataExporter:
    """
    Comprehensive exporter for training data in various ML formats.
    """

    def __init__(self, export_directory: str = "models/training_data/exports"):
        """
        Initialize the data exporter.

        Args:
            export_directory: Directory for storing exported files
        """
        self.export_directory = Path(export_directory)
        self.export_directory.mkdir(parents=True, exist_ok=True)

        # Data collector and validator
        self.data_collector = get_data_collector()
        self.validator = get_data_validator()

        logger.info(f"TrainingDataExporter initialized with export directory: {export_directory}")

    def export_for_ner_training(
        self,
        output_path: str,
        include_confidence: bool = True,
        quality_threshold: float = 0.7,
        format: str = "json"
    ) -> str:
        """
        Export data specifically formatted for Named Entity Recognition training.

        Args:
            output_path: Path for the exported file
            include_confidence: Whether to include confidence scores
            quality_threshold: Minimum quality score to include
            format: Export format ('json', 'csv', 'conll')

        Returns:
            Path to the exported file
        """
        # Filter examples by quality
        filtered_examples = [
            ex for ex in self.data_collector.examples
            if ex.data_quality_score >= quality_threshold
        ]

        if format.lower() == "conll":
            return self._export_conll_format(filtered_examples, output_path, include_confidence)
        elif format.lower() == "csv":
            return self._export_ner_csv(filtered_examples, output_path, include_confidence)
        else:
            return self._export_ner_json(filtered_examples, output_path, include_confidence)

    def _export_ner_json(self, examples: List[TrainingExample], output_path: str, include_confidence: bool) -> str:
        """Export NER data in JSON format."""
        export_data = []

        for example in examples:
            # Extract entities from the text
            ner_data = {
                "text": example.extraction_input,
                "entities": [],
                "metadata": {
                    "example_id": example.id,
                    "quality_score": example.data_quality_score,
                    "timestamp": example.timestamp
                }
            }

            # Convert extracted data to entity format
            entities = self._extract_entities_for_ner(example)
            ner_data["entities"] = entities

            if include_confidence:
                ner_data["confidence_scores"] = example.confidence_scores

            export_data.append(ner_data)

        # Save to file
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(export_data)} NER training examples to {output_file}")
        return str(output_file)

    def _export_ner_csv(self, examples: List[TrainingExample], output_path: str, include_confidence: bool) -> str:
        """Export NER data in CSV format."""
        output_file = Path(output_path)

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            header = ["example_id", "text", "entity_type", "entity_text", "start_pos", "end_pos"]
            if include_confidence:
                header.append("confidence")
            header.extend(["quality_score", "timestamp"])

            writer.writerow(header)

            # Write data
            for example in examples:
                entities = self._extract_entities_for_ner(example)

                for entity in entities:
                    row = [
                        example.id,
                        example.extraction_input.replace('\n', ' '),  # Flatten text
                        entity["entity_type"],
                        entity["entity_text"],
                        entity["start_pos"],
                        entity["end_pos"]
                    ]

                    if include_confidence:
                        row.append(example.confidence_scores.get(entity["entity_type"], 0.0))

                    row.extend([example.data_quality_score, example.timestamp])
                    writer.writerow(row)

        logger.info(f"Exported NER training data to {output_file}")
        return str(output_file)

    def _export_conll_format(self, examples: List[TrainingExample], output_path: str, include_confidence: bool) -> str:
        """Export NER data in CoNLL format."""
        output_file = Path(output_path)

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                # Tokenize text and assign labels
                tokens, labels = self._tokenize_and_label(example)

                # Write tokens with labels
                for token, label in zip(tokens, labels):
                    f.write(f"{token} {label}\n")
                f.write("\n")  # Empty line between sentences

        logger.info(f"Exported NER training data in CoNLL format to {output_file}")
        return str(output_file)

    def _extract_entities_for_ner(self, example: TrainingExample) -> List[Dict[str, Any]]:
        """Extract entities from training example for NER format."""
        entities = []

        # Process complainant
        complainant = example.actual_output.get('complainant', {})
        if isinstance(complainant, dict):
            if complainant.get('name'):
                entities.append({
                    "entity_type": "PERSON",
                    "entity_text": complainant['name'],
                    "start_pos": -1,  # Would need actual position in text
                    "end_pos": -1
                })

        # Process victims
        victims = example.actual_output.get('victims', [])
        if isinstance(victims, list):
            for victim in victims:
                if isinstance(victim, dict) and victim.get('name'):
                    entities.append({
                        "entity_type": "PERSON",
                        "entity_text": victim['name'],
                        "start_pos": -1,
                        "end_pos": -1
                    })

        # Process accused
        accused = example.actual_output.get('accused', [])
        if isinstance(accused, list):
            for acc in accused:
                if isinstance(acc, dict) and acc.get('name'):
                    entities.append({
                        "entity_type": "PERSON",
                        "entity_text": acc['name'],
                        "start_pos": -1,
                        "end_pos": -1
                    })

        # Process locations
        location_fields = ['police_station', 'place_of_occurrence']
        for field in location_fields:
            value = example.actual_output.get(field, "")
            if value:
                entities.append({
                    "entity_type": "LOCATION",
                    "entity_text": value,
                    "start_pos": -1,
                    "end_pos": -1
                })

        # Process dates
        date_fields = ['date_of_occurrence', 'date_of_report']
        for field in date_fields:
            value = example.actual_output.get(field, "")
            if value:
                entities.append({
                    "entity_type": "DATE",
                    "entity_text": value,
                    "start_pos": -1,
                    "end_pos": -1
                })

        return entities

    def _tokenize_and_label(self, example: TrainingExample) -> Tuple[List[str], List[str]]:
        """Tokenize text and assign NER labels for CoNLL format."""
        import re

        text = example.extraction_input
        tokens = []
        labels = []

        # Simple tokenization (can be enhanced with spaCy)
        words = re.findall(r'\b\w+\b', text)

        # For each word, determine if it's part of an entity
        current_pos = 0
        for word in words:
            word_start = text.find(word, current_pos)
            word_end = word_start + len(word)

            # Check if word is part of any entity
            entity_label = "O"  # Outside

            # Check complainant name
            complainant = example.actual_output.get('complainant', {})
            if isinstance(complainant, dict) and complainant.get('name'):
                comp_name = complainant['name']
                if comp_name in text[word_start:word_end+10]:  # Simple overlap check
                    entity_label = "B-PERSON" if word == comp_name.split()[0] else "I-PERSON"

            # Check other entities similarly
            # This is simplified - in practice, would need proper position tracking

            tokens.append(word)
            labels.append(entity_label)
            current_pos = word_end

        return tokens, labels

    def export_for_classification(
        self,
        output_path: str,
        classification_type: str = "section_classification",
        quality_threshold: float = 0.7
    ) -> str:
        """
        Export data for classification tasks.

        Args:
            output_path: Path for the exported file
            classification_type: Type of classification ('section_classification', 'priority_classification')
            quality_threshold: Minimum quality score to include

        Returns:
            Path to the exported file
        """
        # Filter examples by quality
        filtered_examples = [
            ex for ex in self.data_collector.examples
            if ex.data_quality_score >= quality_threshold
        ]

        if classification_type == "section_classification":
            return self._export_section_classification(filtered_examples, output_path)
        elif classification_type == "priority_classification":
            return self._export_priority_classification(filtered_examples, output_path)
        else:
            raise ValueError(f"Unknown classification type: {classification_type}")

    def _export_section_classification(self, examples: List[TrainingExample], output_path: str) -> str:
        """Export data for legal section classification."""
        export_data = []

        for example in examples:
            section = example.actual_output.get('section_of_law', "")

            if section:
                # Categorize sections (simplified)
                category = self._categorize_legal_section(section)

                export_data.append({
                    "text": example.extraction_input,
                    "section": section,
                    "category": category,
                    "example_id": example.id,
                    "quality_score": example.data_quality_score
                })

        # Save to file
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(export_data)} section classification examples to {output_file}")
        return str(output_file)

    def _export_priority_classification(self, examples: List[TrainingExample], output_path: str) -> str:
        """Export data for case priority classification."""
        export_data = []

        for example in examples:
            # Determine priority based on content (simplified heuristic)
            priority = self._determine_case_priority(example)

            export_data.append({
                "text": example.extraction_input,
                "priority": priority,
                "example_id": example.id,
                "quality_score": example.data_quality_score
            })

        # Save to file
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(export_data)} priority classification examples to {output_file}")
        return str(output_file)

    def _categorize_legal_section(self, section: str) -> str:
        """Categorize legal sections into broad categories."""
        section_upper = section.upper()

        if 'IPC' in section_upper or 'BNS' in section_upper:
            if any(code in section_upper for code in ['302', '307', '326']):  # Violent crimes
                return "VIOLENT_CRIME"
            elif any(code in section_upper for code in ['379', '380', '381']):  # Theft
                return "THEFT"
            elif any(code in section_upper for code in ['420', '465', '477']):  # Fraud
                return "FRAUD"
            else:
                return "OTHER_IPC"

        return "OTHER"

    def _determine_case_priority(self, example: TrainingExample) -> str:
        """Determine case priority based on content."""
        text = example.extraction_input.upper()
        output = example.actual_output

        # High priority indicators
        if any(indicator in text for indicator in ['MURDER', 'RAPE', 'KIDNAPPING', 'TERRORISM']):
            return "HIGH"
        elif any(indicator in text for indicator in ['DEATH', 'SERIOUS INJURY', 'LARGE VALUE']):
            return "MEDIUM"
        else:
            return "LOW"

    def export_for_model_training(
        self,
        output_path: str,
        model_type: str = "extraction",
        format: str = "json",
        quality_threshold: float = 0.7,
        train_test_split: float = 0.8,
        include_confidence: bool = True
    ) -> Dict[str, str]:
        """
        Export comprehensive training data for various model types.

        Args:
            output_path: Base path for exported files
            model_type: Type of model ('extraction', 'ner', 'classification')
            format: Export format
            quality_threshold: Minimum quality score
            train_test_split: Ratio for train/test split
            include_confidence: Whether to include confidence scores

        Returns:
            Dictionary with paths to exported files
        """
        # Filter examples by quality
        filtered_examples = [
            ex for ex in self.data_collector.examples
            if ex.data_quality_score >= quality_threshold
        ]

        if not filtered_examples:
            raise ValueError("No examples meet the quality threshold")

        # Split into train and test sets
        import random
        random.shuffle(filtered_examples)

        split_point = int(len(filtered_examples) * train_test_split)
        train_examples = filtered_examples[:split_point]
        test_examples = filtered_examples[split_point:]

        output_base = Path(output_path)
        exported_files = {}

        if model_type == "extraction":
            # Export for field extraction models
            train_file = output_base / "train_extraction.json"
            test_file = output_base / "test_extraction.json"

            self._export_extraction_data(train_examples, str(train_file), include_confidence)
            self._export_extraction_data(test_examples, str(test_file), include_confidence)

            exported_files["train"] = str(train_file)
            exported_files["test"] = str(test_file)

        elif model_type == "ner":
            # Export for NER models
            train_file = output_base / "train_ner.json"
            test_file = output_base / "test_ner.json"

            self._export_ner_json(train_examples, str(train_file), include_confidence)
            self._export_ner_json(test_examples, str(test_file), include_confidence)

            exported_files["train"] = str(train_file)
            exported_files["test"] = str(test_file)

        elif model_type == "classification":
            # Export for classification models
            train_file = output_base / "train_classification.json"
            test_file = output_base / "test_classification.json"

            self._export_classification_data(train_examples, str(train_file))
            self._export_classification_data(test_examples, str(test_file))

            exported_files["train"] = str(train_file)
            exported_files["test"] = str(test_file)

        # Create metadata file
        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "format": format,
            "quality_threshold": quality_threshold,
            "train_test_split": train_test_split,
            "total_examples": len(filtered_examples),
            "train_examples": len(train_examples),
            "test_examples": len(test_examples),
            "include_confidence": include_confidence
        }

        metadata_file = output_base / "export_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        exported_files["metadata"] = str(metadata_file)

        logger.info(f"Exported training data: {len(train_examples)} train, {len(test_examples)} test examples")
        return exported_files

    def _export_extraction_data(self, examples: List[TrainingExample], output_path: str, include_confidence: bool) -> None:
        """Export data for field extraction training."""
        export_data = []

        for example in examples:
            training_instance = {
                "input_text": example.extraction_input,
                "target_output": example.expected_output if example.is_corrected else example.actual_output,
                "metadata": {
                    "example_id": example.id,
                    "quality_score": example.data_quality_score,
                    "is_corrected": example.is_corrected,
                    "timestamp": example.timestamp
                }
            }

            if include_confidence:
                training_instance["confidence_scores"] = example.confidence_scores

            export_data.append(training_instance)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def _export_classification_data(self, examples: List[TrainingExample], output_path: str) -> None:
        """Export data for classification training."""
        export_data = []

        for example in examples:
            # Multi-task classification data
            classification_targets = {}

            # Section category
            section = example.actual_output.get('section_of_law', "")
            if section:
                classification_targets["section_category"] = self._categorize_legal_section(section)

            # Case priority
            classification_targets["priority"] = self._determine_case_priority(example)

            if classification_targets:
                export_data.append({
                    "input_text": example.extraction_input,
                    "targets": classification_targets,
                    "metadata": {
                        "example_id": example.id,
                        "quality_score": example.data_quality_score,
                        "timestamp": example.timestamp
                    }
                })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def create_model_ready_dataset(
        self,
        output_directory: str,
        model_framework: str = "huggingface",
        quality_threshold: float = 0.7
    ) -> Dict[str, str]:
        """
        Create a complete dataset ready for specific ML frameworks.

        Args:
            output_directory: Directory for the dataset
            model_framework: Target framework ('huggingface', 'spacy', 'tensorflow')
            quality_threshold: Minimum quality score

        Returns:
            Dictionary with paths to created dataset files
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter high-quality examples
        filtered_examples = [
            ex for ex in self.data_collector.examples
            if ex.data_quality_score >= quality_threshold
        ]

        dataset_files = {}

        if model_framework == "huggingface":
            dataset_files = self._create_huggingface_dataset(filtered_examples, output_dir)
        elif model_framework == "spacy":
            dataset_files = self._create_spacy_dataset(filtered_examples, output_dir)
        elif model_framework == "tensorflow":
            dataset_files = self._create_tensorflow_dataset(filtered_examples, output_dir)

        # Create dataset info
        info_file = output_dir / "dataset_info.json"
        dataset_info = {
            "framework": model_framework,
            "created_at": datetime.now().isoformat(),
            "total_examples": len(filtered_examples),
            "quality_threshold": quality_threshold,
            "fields": list(self.data_collector.validation_rules.keys()) if hasattr(self.data_collector, 'validation_rules') else [],
            "files": dataset_files
        }

        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        dataset_files["info"] = str(info_file)

        logger.info(f"Created {model_framework} dataset with {len(filtered_examples)} examples")
        return dataset_files

    def _create_huggingface_dataset(self, examples: List[TrainingExample], output_dir: Path) -> Dict[str, str]:
        """Create HuggingFace-style dataset."""
        # Convert to HuggingFace dataset format
        hf_data = []

        for example in examples:
            hf_data.append({
                "text": example.extraction_input,
                **example.actual_output,
                "quality_score": example.data_quality_score,
                "confidence_scores": example.confidence_scores
            })

        # Save as JSON (HuggingFace can load this)
        train_file = output_dir / "train.json"
        test_file = output_dir / "test.json"

        # Simple split
        split_point = int(len(hf_data) * 0.8)
        train_data = hf_data[:split_point]
        test_data = hf_data[split_point:]

        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        return {
            "train": str(train_file),
            "test": str(test_file)
        }

    def _create_spacy_dataset(self, examples: List[TrainingExample], output_dir: Path) -> Dict[str, str]:
        """Create spaCy-compatible dataset."""
        # Convert to spaCy training format
        spacy_data = []

        for example in examples:
            # Create spaCy training example
            entities = self._extract_entities_for_ner(example)

            # Convert entities to spaCy format (start, end, label)
            spacy_entities = []
            for entity in entities:
                # This is simplified - would need proper position finding
                spacy_entities.append((0, len(example.extraction_input), entity["entity_type"]))

            spacy_data.append((example.extraction_input, {"entities": spacy_entities}))

        # Save as pickle (spaCy's preferred format)
        data_file = output_dir / "spacy_training_data.pkl"

        with open(data_file, 'wb') as f:
            pickle.dump(spacy_data, f)

        return {"data": str(data_file)}

    def _create_tensorflow_dataset(self, examples: List[TrainingExample], output_dir: Path) -> Dict[str, str]:
        """Create TensorFlow-compatible dataset."""
        # Create TF records or simple format
        tf_data = []

        for example in examples:
            tf_data.append({
                "input_text": example.extraction_input,
                "target_fields": example.actual_output,
                "confidence_scores": example.confidence_scores,
                "quality_score": example.data_quality_score
            })

        # Save as JSON for easy TensorFlow loading
        data_file = output_dir / "tensorflow_data.json"

        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(tf_data, f, ensure_ascii=False, indent=2)

        return {"data": str(data_file)}

    def export_validation_report(self, output_path: str) -> str:
        """Export a comprehensive validation report."""
        # Generate validation report using the validator
        report_content = self.validator.generate_validation_report(self.data_collector.examples)

        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"Exported validation report to {output_file}")
        return str(output_file)

# Global exporter instance
_data_exporter = None

def get_data_exporter(export_directory: str = "models/training_data/exports") -> TrainingDataExporter:
    """Get or create global data exporter instance."""
    global _data_exporter
    if _data_exporter is None:
        _data_exporter = TrainingDataExporter(export_directory)
    return _data_exporter

def export_for_model_training(
    output_path: str,
    model_type: str = "extraction",
    **kwargs
) -> Dict[str, str]:
    """Convenience function to export training data."""
    exporter = get_data_exporter()
    return exporter.export_for_model_training(output_path, model_type, **kwargs)