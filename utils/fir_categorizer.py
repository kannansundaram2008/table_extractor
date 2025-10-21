"""
FIR Type Classification System
Main classification engine for categorizing FIR documents by type.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import time
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Result of FIR classification."""
    category: str
    confidence: float
    scores: Dict[str, float]
    reasoning: List[str]
    processing_time: float
    method: str  # 'rule_based', 'ml', 'hybrid'

@dataclass
class CategoryMatch:
    """Individual category match details."""
    category: str
    keyword_matches: int
    section_matches: int
    context_score: float
    total_score: float

class FIRCategorizer:
    """
    Main FIR categorization system using multiple classification approaches.
    """

    def __init__(self, categories_file: str = 'models/fir_categories.json', model_dir: str = 'models'):
        """
        Initialize the FIR categorizer.

        Args:
            categories_file: Path to categories configuration file
            model_dir: Directory for storing trained models
        """
        self.categories_file = Path(categories_file)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Load category definitions
        self.categories_config = self._load_categories()
        self.categories = self.categories_config.get('categories', {})

        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=self.categories_config.get('training_config', {}).get('max_features', 5000),
            ngram_range=self.categories_config.get('training_config', {}).get('ngram_range', [1, 3]),
            stop_words='english'
        )

        # Classification models
        self.classifier = RandomForestClassifier(
            n_estimators=self.categories_config.get('training_config', {}).get('n_estimators', 100),
            random_state=self.categories_config.get('training_config', {}).get('random_state', 42),
            n_jobs=-1
        )

        # Training data for incremental learning
        self.training_data = []
        self.is_trained = False

        # Load existing models if available
        self._load_models()

        # Performance tracking
        self.performance_stats = {
            'total_classifications': 0,
            'total_processing_time': 0.0,
            'method_usage': Counter(),
            'accuracy_tracking': []
        }

    def _load_categories(self) -> Dict[str, Any]:
        """Load category definitions from JSON file."""
        try:
            with open(self.categories_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Categories file not found: {self.categories_file}")
            return self._get_default_categories()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing categories file: {e}")
            return self._get_default_categories()

    def _get_default_categories(self) -> Dict[str, Any]:
        """Get default category configuration if file loading fails."""
        return {
            'categories': {
                'CRIMINAL': {
                    'name': 'Criminal Cases',
                    'keywords': ['theft', 'assault', 'murder'],
                    'ipc_sections': ['IPC 302', 'IPC 379']
                },
                'CIVIL': {
                    'name': 'Civil Cases',
                    'keywords': ['property', 'contract'],
                    'ipc_sections': []
                }
            },
            'classification_rules': {
                'min_keyword_matches': 1,
                'confidence_thresholds': {'high': 0.8, 'medium': 0.6, 'low': 0.4}
            }
        }

    def classify_fir(self, fir_text: str, fir_data: Optional[Dict] = None) -> ClassificationResult:
        """
        Classify FIR document into appropriate category.

        Args:
            fir_text: The main FIR text content
            fir_data: Optional parsed FIR data from FIR parser

        Returns:
            ClassificationResult with category, confidence, and reasoning
        """
        start_time = time.time()

        if not fir_text or not fir_text.strip():
            return ClassificationResult(
                category='UNKNOWN',
                confidence=0.0,
                scores={},
                reasoning=['Empty or invalid FIR text'],
                processing_time=0.0,
                method='none'
            )

        # Normalize text
        normalized_text = self._normalize_text(fir_text)

        # Get classification scores using multiple methods
        scores = {}
        matches = []

        # Rule-based classification
        rule_scores = self._rule_based_classification(normalized_text, fir_data)
        scores.update(rule_scores)

        # Pattern-based classification
        pattern_scores = self._pattern_based_classification(normalized_text)
        scores.update(pattern_scores)

        # ML-based classification if trained
        if self.is_trained:
            ml_scores = self._ml_based_classification(normalized_text)
            scores.update(ml_scores)

        # Combine scores and determine best category
        best_category, confidence, reasoning = self._combine_scores(scores)

        # Update performance tracking
        processing_time = time.time() - start_time
        self.performance_stats['total_classifications'] += 1
        self.performance_stats['total_processing_time'] += processing_time

        # Determine method used
        method = 'hybrid'
        if not self.is_trained:
            method = 'rule_based'
        elif all(score < 0.3 for score in scores.values()):
            method = 'rule_based'

        self.performance_stats['method_usage'][method] += 1

        return ClassificationResult(
            category=best_category,
            confidence=confidence,
            scores=scores,
            reasoning=reasoning,
            processing_time=processing_time,
            method=method
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better classification."""
        if not text:
            return ""

        # Convert to lowercase for matching
        normalized = text.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove special characters but keep important ones
        normalized = re.sub(r'[^\w\s\-/.,]', ' ', normalized)

        return normalized.strip()

    def _rule_based_classification(self, text: str, fir_data: Optional[Dict] = None) -> Dict[str, float]:
        """Rule-based classification using keywords and sections."""
        scores = {}

        for category_id, category_info in self.categories.items():
            score = 0.0
            matches = []

            # Keyword matching
            keywords = category_info.get('keywords', [])
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text)

            if keyword_matches > 0:
                # Normalize by total keywords
                keyword_score = keyword_matches / len(keywords) if keywords else 0
                score += keyword_score * self.categories_config.get('feature_weights', {}).get('keyword_matches', 0.4)
                matches.append(f"Found {keyword_matches} keyword matches")

            # IPC section matching
            ipc_sections = category_info.get('ipc_sections', [])
            section_matches = 0

            for section in ipc_sections:
                if section.lower() in text:
                    section_matches += 1

            if section_matches > 0:
                section_score = min(section_matches / 5, 1.0)  # Cap at 1.0
                score += section_score * self.categories_config.get('feature_weights', {}).get('section_matches', 0.3)
                matches.append(f"Found {section_matches} IPC section matches")

            # Context similarity (if FIR data available)
            if fir_data:
                context_score = self._calculate_context_similarity(category_id, fir_data)
                score += context_score * self.categories_config.get('feature_weights', {}).get('context_similarity', 0.2)
                if context_score > 0:
                    matches.append(f"Context similarity: {context_score:.2f}")

            # Apply category priority weight
            priority_weight = self.categories_config.get('classification_rules', {}).get('category_priority_weights', {}).get(category_id, 1.0)
            score *= priority_weight

            scores[category_id] = score

        return scores

    def _pattern_based_classification(self, text: str) -> Dict[str, float]:
        """Pattern-based classification using regex and structural analysis."""
        scores = {}

        for category_id, category_info in self.categories.items():
            pattern_score = 0.0

            # Category-specific patterns
            if category_id == 'CRIMINAL':
                # Criminal case patterns
                criminal_patterns = [
                    r'\b(theft|stole|robbed|assaulted|murdered|raped|kidnapped)\b',
                    r'\b(criminal|crime|offense|offence)\b',
                    r'\b(injury|hurt|wound|assault)\b'
                ]
                pattern_matches = sum(1 for pattern in criminal_patterns if re.search(pattern, text))
                pattern_score = min(pattern_matches / 3, 1.0)

            elif category_id == 'TRAFFIC':
                # Traffic case patterns
                traffic_patterns = [
                    r'\b(accident|collision|crash)\b',
                    r'\b(vehicle|car|truck|bus|motorcycle)\b',
                    r'\b(driving|driver|license|registration)\b'
                ]
                pattern_matches = sum(1 for pattern in traffic_patterns if re.search(pattern, text))
                pattern_score = min(pattern_matches / 3, 1.0)

            elif category_id == 'CYBER':
                # Cyber crime patterns
                cyber_patterns = [
                    r'\b(online|internet|computer|digital)\b',
                    r'\b(hacking|phishing|fraud|cyber)\b',
                    r'\b(email|website|account|password)\b'
                ]
                pattern_matches = sum(1 for pattern in cyber_patterns if re.search(pattern, text))
                pattern_score = min(pattern_matches / 3, 1.0)

            elif category_id == 'ECONOMIC':
                # Economic offense patterns
                economic_patterns = [
                    r'\b(fraud|cheating|misappropriation)\b',
                    r'\b(money|financial|bank|account)\b',
                    r'\b(transaction|payment|cheque|currency)\b'
                ]
                pattern_matches = sum(1 for pattern in economic_patterns if re.search(pattern, text))
                pattern_score = min(pattern_matches / 3, 1.0)

            elif category_id == 'NARCOTICS':
                # Narcotics patterns
                narcotics_patterns = [
                    r'\b(drug|narcotic|opium|heroin|cocaine)\b',
                    r'\b(possession|trafficking|smuggling)\b',
                    r'\b(substance|contraband|intoxicant)\b'
                ]
                pattern_matches = sum(1 for pattern in narcotics_patterns if re.search(pattern, text))
                pattern_score = min(pattern_matches / 3, 1.0)

            # Apply pattern weight
            pattern_weight = self.categories_config.get('feature_weights', {}).get('pattern_recognition', 0.1)
            scores[category_id] = pattern_score * pattern_weight

        return scores

    def _ml_based_classification(self, text: str) -> Dict[str, float]:
        """ML-based classification using trained model."""
        try:
            # Vectorize text
            text_vector = self.vectorizer.transform([text])

            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(text_vector)[0]

            # Map probabilities to categories
            category_labels = self.classifier.classes_
            ml_scores = {}

            for i, category in enumerate(category_labels):
                if category in self.categories:
                    ml_scores[category] = probabilities[i]

            return ml_scores

        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return {}

    def _calculate_context_similarity(self, category_id: str, fir_data: Dict) -> float:
        """Calculate similarity based on FIR context data."""
        if not fir_data:
            return 0.0

        similarity_score = 0.0

        # Check section of law
        section_of_law = fir_data.get('section_of_law', '')
        if section_of_law:
            category_sections = self.categories[category_id].get('ipc_sections', [])
            section_matches = sum(1 for section in category_sections if section.lower() in section_of_law.lower())
            if section_matches > 0:
                similarity_score += 0.3

        # Check gist content
        gist = fir_data.get('gist', '')
        if gist:
            keywords = self.categories[category_id].get('keywords', [])
            gist_matches = sum(1 for keyword in keywords if keyword.lower() in gist.lower())
            if gist_matches > 0:
                similarity_score += min(gist_matches / 3, 0.3)

        # Check property information for economic cases
        if category_id == 'ECONOMIC':
            property_data = fir_data.get('property_lost', []) + fir_data.get('property_recovered', [])
            if property_data:
                similarity_score += 0.2

        return min(similarity_score, 1.0)

    def _combine_scores(self, scores: Dict[str, float]) -> Tuple[str, float, List[str]]:
        """Combine all scores to determine best category."""
        if not scores:
            return 'MISCELLANEOUS', 0.0, ['No classification scores available']

        # Filter out low scores
        min_score = self.categories_config.get('classification_rules', {}).get('min_keyword_matches', 1) * 0.1
        filtered_scores = {k: v for k, v in scores.items() if v >= min_score}

        if not filtered_scores:
            return 'MISCELLANEOUS', 0.0, ['All scores below threshold']

        # Get best category
        best_category = max(filtered_scores.items(), key=lambda x: x[1])
        category_id, score = best_category

        # Generate reasoning
        reasoning = []
        if score > 0.7:
            reasoning.append(f"High confidence score: {score:.3f}")
        elif score > 0.4:
            reasoning.append(f"Medium confidence score: {score:.3f}")
        else:
            reasoning.append(f"Low confidence score: {score:.3f}")

        # Add category-specific reasoning
        category_info = self.categories.get(category_id, {})
        if category_info:
            reasoning.append(f"Classified as: {category_info.get('name', category_id)}")

        return category_id, score, reasoning

    def add_training_example(self, fir_text: str, correct_category: str, fir_data: Optional[Dict] = None):
        """
        Add training example for incremental learning.

        Args:
            fir_text: The FIR text content
            correct_category: The correct category for this FIR
            fir_data: Optional parsed FIR data
        """
        example = {
            'text': fir_text,
            'category': correct_category,
            'fir_data': fir_data or {},
            'timestamp': time.time()
        }

        self.training_data.append(example)
        logger.info(f"Added training example for category: {correct_category}")

    def train_classifier(self, test_size: float = 0.2):
        """
        Train the ML classifier using collected training data.

        Args:
            test_size: Fraction of data to use for testing
        """
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for model training")
            return

        try:
            # Prepare training data
            texts = [ex['text'] for ex in self.training_data]
            categories = [ex['category'] for ex in self.training_data]

            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            y = np.array(categories)

            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Train classifier
            self.classifier.fit(X_train, y_train)

            # Evaluate
            train_accuracy = self.classifier.score(X_train, y_train)
            test_accuracy = self.classifier.score(X_test, y_test)

            logger.info(f"Classifier trained - Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")

            # Save model
            self._save_models()
            self.is_trained = True

        except Exception as e:
            logger.error(f"Error training classifier: {e}")

    def _save_models(self):
        """Save trained models to disk."""
        try:
            model_path = self.model_dir / 'fir_categorizer'
            model_path.mkdir(exist_ok=True)

            joblib.dump(self.vectorizer, model_path / 'vectorizer.joblib')
            joblib.dump(self.classifier, model_path / 'classifier.joblib')

            # Save training data
            import pickle
            with open(model_path / 'training_data.pkl', 'wb') as f:
                pickle.dump(self.training_data, f)

            logger.info("FIR categorizer models saved successfully")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def _load_models(self):
        """Load trained models from disk."""
        try:
            model_path = self.model_dir / 'fir_categorizer'

            if not model_path.exists():
                return

            vectorizer_path = model_path / 'vectorizer.joblib'
            classifier_path = model_path / 'classifier.joblib'
            training_data_path = model_path / 'training_data.pkl'

            if vectorizer_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)

            if classifier_path.exists():
                self.classifier = joblib.load(classifier_path)

            if training_data_path.exists():
                import pickle
                with open(training_data_path, 'rb') as f:
                    self.training_data = pickle.load(f)

            if hasattr(self.classifier, 'classes_'):
                self.is_trained = True
                logger.info("FIR categorizer models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        if stats['total_classifications'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_classifications']

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_classifications': 0,
            'total_processing_time': 0.0,
            'method_usage': Counter(),
            'accuracy_tracking': []
        }

    def get_category_info(self, category_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific category."""
        return self.categories.get(category_id)

    def list_categories(self) -> List[str]:
        """Get list of available categories."""
        return list(self.categories.keys())

    def export_training_data(self, filepath: str):
        """Export training data for external analysis."""
        try:
            export_data = {
                'categories': self.categories,
                'training_examples': self.training_data,
                'config': self.categories_config
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Training data exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting training data: {e}")

# Global instance for backward compatibility
_categorizer = None

def get_fir_categorizer(categories_file: str = 'models/fir_categories.json') -> FIRCategorizer:
    """Get or create global categorizer instance."""
    global _categorizer
    if _categorizer is None:
        _categorizer = FIRCategorizer(categories_file)
    return _categorizer

def classify_fir_text(fir_text: str, fir_data: Optional[Dict] = None) -> ClassificationResult:
    """
    Convenience function to classify FIR text.

    Args:
        fir_text: The FIR text content
        fir_data: Optional parsed FIR data

    Returns:
        ClassificationResult with category and confidence
    """
    categorizer = get_fir_categorizer()
    return categorizer.classify_fir(fir_text, fir_data)