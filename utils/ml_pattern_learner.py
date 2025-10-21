"""
Machine Learning-based Pattern Learning for FIR Document Processing.
Uses supervised and unsupervised learning to improve entity extraction and pattern recognition.
"""

import re
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import textstat

# Configure logging
logger = logging.getLogger(__name__)

class MLPatternLearner:
    """
    Machine learning system for learning patterns in FIR documents.
    Combines supervised learning for entity classification and unsupervised learning for pattern discovery.
    """

    def __init__(self, model_dir: str = 'models'):
        """
        Initialize the ML pattern learner.

        Args:
            model_dir: Directory to save/load trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Initialize models
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.clusterer = KMeans(n_clusters=5, random_state=42)

        # Training data storage
        self.training_data = []
        self.entity_patterns = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'DATE': [],
            'LAW': [],
            'MONEY': []
        }

        # Load existing models if available
        self._load_models()

    def add_training_example(self, text: str, entities: Dict[str, List[str]], context: str = ''):
        """
        Add a training example to the dataset.

        Args:
            text: The original text
            entities: Dictionary of entity types and their values
            context: Additional context about the document
        """
        example = {
            'text': text,
            'entities': entities,
            'context': context,
            'features': self._extract_features(text),
            'timestamp': datetime.now().isoformat()
        }
        self.training_data.append(example)

    def _extract_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic and statistical features from text."""
        features = {}

        # Basic text statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0

        # Readability metrics
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            features['gunning_fog'] = textstat.gunning_fog(text)
        except:
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0
            features['gunning_fog'] = 0

        # Capitalization patterns
        words = text.split()
        features['capitalized_words'] = sum(1 for word in words if word.istitle())
        features['all_caps_words'] = sum(1 for word in words if word.isupper() and len(word) > 1)
        features['capitalized_ratio'] = features['capitalized_words'] / features['word_count'] if features['word_count'] > 0 else 0

        # Punctuation patterns
        features['colon_count'] = text.count(':')
        features['comma_count'] = text.count(',')
        features['period_count'] = text.count('.')
        features['slash_count'] = text.count('/')

        # Legal domain specific patterns
        features['has_cr_no'] = bool(re.search(r'\b\d{1,4}/\d{2,4}\b', text))
        features['has_ipc'] = bool(re.search(r'\bIPC\b', text, re.IGNORECASE))
        features['has_section'] = bool(re.search(r'\bsection\b', text, re.IGNORECASE))
        features['has_police_station'] = bool(re.search(r'\bpolice\s+station\b', text, re.IGNORECASE))

        # Date and time patterns
        features['date_count'] = len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text))
        features['time_count'] = len(re.findall(r'\b\d{1,2}:\d{2}\b', text))

        # Name-like patterns
        features['potential_names'] = len(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text))

        return features

    def train_entity_classifier(self, test_size: float = 0.2):
        """
        Train the entity classification model.

        Args:
            test_size: Fraction of data to use for testing
        """
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for model training")
            return

        # Prepare training data
        texts = [ex['text'] for ex in self.training_data]
        labels = []

        for example in self.training_data:
            # Create labels for each entity type
            for entity_type, entities in example['entities'].items():
                for entity in entities:
                    labels.append(entity_type)

        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        if len(np.unique(y)) < 2:
            logger.warning("Need at least 2 classes for classification")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train classifier
        self.classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Entity classifier trained with accuracy: {accuracy:.3f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")

        # Save model
        self._save_models()

    def train_pattern_discovery(self, n_clusters: int = 5):
        """
        Train unsupervised pattern discovery model.

        Args:
            n_clusters: Number of clusters for pattern discovery
        """
        if len(self.training_data) < n_clusters:
            logger.warning("Insufficient data for pattern discovery")
            return

        # Extract features for clustering
        features_list = [ex['features'] for ex in self.training_data]
        feature_df = pd.DataFrame(features_list)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df)

        # Train clusterer
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.clusterer.fit_predict(X_scaled)

        # Analyze clusters
        self._analyze_clusters(feature_df, clusters)

        # Save models
        self._save_models()

    def _analyze_clusters(self, feature_df: pd.DataFrame, clusters: np.ndarray):
        """Analyze the characteristics of discovered clusters."""
        cluster_analysis = {}

        for cluster_id in np.unique(clusters):
            cluster_data = feature_df[clusters == cluster_id]
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'avg_length': cluster_data['length'].mean(),
                'avg_word_count': cluster_data['word_count'].mean(),
                'common_patterns': self._identify_common_patterns(cluster_data)
            }

        logger.info("Pattern discovery analysis:")
        for cluster_id, analysis in cluster_analysis.items():
            logger.info(f"Cluster {cluster_id}: {analysis['size']} samples, "
                       f"avg length: {analysis['avg_length']:.1f}, "
                       f"avg words: {analysis['avg_word_count']:.1f}")

    def _identify_common_patterns(self, cluster_data: pd.DataFrame) -> List[str]:
        """Identify common patterns in a cluster."""
        patterns = []

        # Check for common structural patterns
        if cluster_data['colon_count'].mean() > 1.5:
            patterns.append("structured_data")
        if cluster_data['has_cr_no'].mean() > 0.5:
            patterns.append("fir_format")
        if cluster_data['potential_names'].mean() > 2:
            patterns.append("personal_info")
        if cluster_data['has_ipc'].mean() > 0.3:
            patterns.append("legal_content")

        return patterns

    def predict_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Predict entities in new text using trained models.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of predicted entities by type
        """
        if not hasattr(self.classifier, 'estimators_'):
            logger.warning("Classifier not trained, using fallback")
            return self._fallback_entity_prediction(text)

        # Vectorize text
        X = self.vectorizer.transform([text])

        # Get predictions
        predictions = self.classifier.predict(X)

        # Extract entities based on predictions
        entities = {}
        for pred in predictions:
            if pred not in entities:
                entities[pred] = []

            # Extract potential entities of this type
            extracted = self._extract_entities_by_type(text, pred)
            entities[pred].extend(extracted)

        return entities

    def _extract_entities_by_type(self, text: str, entity_type: str) -> List[str]:
        """Extract entities of a specific type from text."""
        extractors = {
            'PERSON': self._extract_person_entities,
            'ORG': self._extract_org_entities,
            'GPE': self._extract_gpe_entities,
            'DATE': self._extract_date_entities,
            'LAW': self._extract_law_entities,
            'MONEY': self._extract_money_entities
        }

        extractor = extractors.get(entity_type, lambda x: [])
        return extractor(text)

    def _extract_person_entities(self, text: str) -> List[str]:
        """Extract person names using learned patterns."""
        # Use regex patterns for person names
        patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Three word names
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Two word names
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b',  # Names with initials
        ]

        persons = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            persons.extend(matches)

        return list(set(persons))  # Remove duplicates

    def _extract_org_entities(self, text: str) -> List[str]:
        """Extract organization names."""
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Police\s+Station\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Court\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Department\b',
        ]

        orgs = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            orgs.extend(matches)

        return list(set(orgs))

    def _extract_gpe_entities(self, text: str) -> List[str]:
        """Extract geographical locations."""
        # Simple location extraction based on context
        locations = []

        # Words after prepositions
        loc_indicators = ['at', 'in', 'near', 'from', 'to']
        for indicator in loc_indicators:
            pattern = rf'\b{indicator}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            matches = re.findall(pattern, text, re.IGNORECASE)
            locations.extend(matches)

        return list(set(locations))

    def _extract_date_entities(self, text: str) -> List[str]:
        """Extract date entities."""
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)

        return list(set(dates))

    def _extract_law_entities(self, text: str) -> List[str]:
        """Extract legal section references."""
        law_patterns = [
            r'\bIPC\s+\d+(?:\([^)]+\))?\b',
            r'\bSection\s+\d+(?:\([^)]+\))?\s+of\s+IPC\b',
            r'\b\d+(?:\([^)]+\))?\s+IPC\b',
        ]

        laws = []
        for pattern in law_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            laws.extend(matches)

        return list(set(laws))

    def _extract_money_entities(self, text: str) -> List[str]:
        """Extract monetary values."""
        money_patterns = [
            r'\bRs\.?\s*\d+(?:,\d{3})*(?:\.\d{2})?\b',
            r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*rupees\b',
        ]

        money = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            money.extend(matches)

        return money

    def _fallback_entity_prediction(self, text: str) -> Dict[str, List[str]]:
        """Fallback entity prediction when model is not trained."""
        # Use simple regex-based extraction
        entities = {
            'PERSON': self._extract_person_entities(text),
            'ORG': self._extract_org_entities(text),
            'GPE': self._extract_gpe_entities(text),
            'DATE': self._extract_date_entities(text),
            'LAW': self._extract_law_entities(text),
            'MONEY': self._extract_money_entities(text)
        }
        return entities

    def get_pattern_clusters(self, text: str) -> Dict[str, Any]:
        """
        Get pattern cluster information for text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with cluster information and patterns
        """
        if not hasattr(self.clusterer, 'cluster_centers_'):
            return {'cluster': None, 'patterns': []}

        features = self._extract_features(text)
        feature_df = pd.DataFrame([features])

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df)

        cluster = self.clusterer.predict(X_scaled)[0]

        # Get cluster characteristics
        cluster_info = {
            'cluster': int(cluster),
            'patterns': self._identify_common_patterns(pd.DataFrame([features]))
        }

        return cluster_info

    def save_training_data(self, filepath: str = 'training_data.pkl'):
        """Save training data to file."""
        filepath = self.model_dir / filepath
        with open(filepath, 'wb') as f:
            pickle.dump(self.training_data, f)
        logger.info(f"Training data saved to {filepath}")

    def load_training_data(self, filepath: str = 'training_data.pkl'):
        """Load training data from file."""
        filepath = self.model_dir / filepath
        if filepath.exists():
            with open(filepath, 'rb') as f:
                self.training_data = pickle.load(f)
            logger.info(f"Training data loaded from {filepath}")
        else:
            logger.warning(f"Training data file {filepath} not found")

    def _save_models(self):
        """Save trained models to disk."""
        try:
            joblib.dump(self.vectorizer, self.model_dir / 'vectorizer.joblib')
            joblib.dump(self.classifier, self.model_dir / 'classifier.joblib')
            joblib.dump(self.clusterer, self.model_dir / 'clusterer.joblib')
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def _load_models(self):
        """Load trained models from disk."""
        try:
            vectorizer_path = self.model_dir / 'vectorizer.joblib'
            classifier_path = self.model_dir / 'classifier.joblib'
            clusterer_path = self.model_dir / 'clusterer.joblib'

            if vectorizer_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)
            if classifier_path.exists():
                self.classifier = joblib.load(classifier_path)
            if clusterer_path.exists():
                self.clusterer = joblib.load(clusterer_path)

            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about the trained models."""
        stats = {
            'training_samples': len(self.training_data),
            'vectorizer_trained': hasattr(self.vectorizer, 'vocabulary_'),
            'classifier_trained': hasattr(self.classifier, 'estimators_'),
            'clusterer_trained': hasattr(self.clusterer, 'cluster_centers_'),
        }

        if hasattr(self.clusterer, 'cluster_centers_'):
            stats['n_clusters'] = len(self.clusterer.cluster_centers_)

        return stats


# Integration with FIR categorization system
def integrate_fir_categorization(ml_learner_instance, fir_categorizer=None):
    """
    Integrate FIR categorization with the ML pattern learner.

    Args:
        ml_learner_instance: Instance of MLPatternLearner
        fir_categorizer: Optional FIR categorizer instance

    Returns:
        Enhanced MLPatternLearner with categorization capabilities
    """
    if fir_categorizer is None:
        from utils.fir_categorizer import get_fir_categorizer
        fir_categorizer = get_fir_categorizer()

    # Add categorization capabilities to ML learner
    ml_learner_instance.fir_categorizer = fir_categorizer
    ml_learner_instance.categorization_enabled = True

    # Add categorization methods
    ml_learner_instance.categorize_fir_text = categorize_fir_text.__get__(ml_learner_instance)
    ml_learner_instance.add_categorization_training = add_categorization_training.__get__(ml_learner_instance)
    ml_learner_instance.train_fir_categorizer = train_fir_categorizer.__get__(ml_learner_instance)

    return ml_learner_instance

def categorize_fir_text(self, fir_text: str, fir_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Categorize FIR text using integrated categorizer.

    Args:
        fir_text: The FIR text content
        fir_data: Optional parsed FIR data

    Returns:
        Dictionary with categorization results
    """
    if not hasattr(self, 'categorization_enabled') or not self.categorization_enabled:
        return {'error': 'Categorization not enabled'}

    try:
        result = self.fir_categorizer.classify_fir(fir_text, fir_data)

        return {
            'category': result.category,
            'confidence': result.confidence,
            'all_scores': result.scores,
            'reasoning': result.reasoning,
            'processing_time': result.processing_time,
            'method': result.method
        }

    except Exception as e:
        logger.error(f"Error in FIR categorization: {e}")
        return {'error': str(e)}

def add_categorization_training(self, fir_text: str, correct_category: str, source_file: str = "manual"):
    """
    Add training example for FIR categorization.

    Args:
        fir_text: The FIR text content
        correct_category: The correct category for this FIR
        source_file: Source file identifier
    """
    if not hasattr(self, 'categorization_enabled') or not self.categorization_enabled:
        logger.warning("Categorization not enabled")
        return

    try:
        self.fir_categorizer.add_training_example(fir_text, correct_category)
        logger.info(f"Added categorization training example for category: {correct_category}")

    except Exception as e:
        logger.error(f"Error adding categorization training: {e}")

def train_fir_categorizer(self, test_size: float = 0.2):
    """
    Train the integrated FIR categorizer.

    Args:
        test_size: Fraction of data to use for testing
    """
    if not hasattr(self, 'categorization_enabled') or not self.categorization_enabled:
        logger.warning("Categorization not enabled")
        return

    try:
        self.fir_categorizer.train_classifier(test_size=test_size)
        logger.info("FIR categorizer training completed")

    except Exception as e:
        logger.error(f"Error training FIR categorizer: {e}")

# Integration with existing FIR parser
def enhance_fir_parser_with_ml(fir_parser_module):
    """
    Enhance the existing FIR parser with ML capabilities.

    Args:
        fir_parser_module: The fir_parser module to enhance
    """
    # Add ML learner to the parser
    ml_learner = MLPatternLearner()

    # Monkey patch the extract_entities_with_ner function
    original_extract_entities = fir_parser_module.extract_entities_with_ner

    def enhanced_extract_entities(text):
        # Try ML-based extraction first
        ml_entities = ml_learner.predict_entities(text)

        # Fall back to original if ML fails
        if not ml_entities or all(not v for v in ml_entities.values()):
            return original_extract_entities(text)

        return ml_entities

    fir_parser_module.extract_entities_with_ner = enhanced_extract_entities
    fir_parser_module.ml_learner = ml_learner

    return fir_parser_module
