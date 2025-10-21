"""
FIR Category Training System
Handles training data management, model training, and evaluation for FIR categorization.
"""

import json
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """Training example for FIR categorization."""
    text: str
    category: str
    source_file: str
    fir_data: Optional[Dict] = None
    metadata: Optional[Dict] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class TrainingResult:
    """Result of training process."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    test_samples: int
    feature_count: int
    model_size: int
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]

class CategoryTrainer:
    """
    Training system for FIR categorization models.
    """

    def __init__(self, categories_file: str = 'models/fir_categories.json', training_dir: str = 'models/training_data'):
        """
        Initialize the category trainer.

        Args:
            categories_file: Path to categories configuration file
            training_dir: Directory for training data and models
        """
        self.categories_file = Path(categories_file)
        self.training_dir = Path(training_dir)
        self.training_dir.mkdir(parents=True, exist_ok=True)

        # Load category definitions
        self.categories_config = self._load_categories()
        self.categories = self.categories_config.get('categories', {})

        # Training data storage
        self.training_examples: List[TrainingExample] = []
        self.validation_examples: List[TrainingExample] = []

        # Available models for comparison
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': MultinomialNB()
        }

        # Vectorizer configuration
        self.vectorizer_config = self.categories_config.get('training_config', {})

        # Load existing training data
        self._load_training_data()

        # Performance tracking
        self.training_history = []

    def _load_categories(self) -> Dict[str, Any]:
        """Load category definitions from JSON file."""
        try:
            with open(self.categories_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading categories: {e}")
            return self._get_default_categories()

    def _get_default_categories(self) -> Dict[str, Any]:
        """Get default category configuration."""
        return {
            'categories': {
                'CRIMINAL': {'name': 'Criminal Cases', 'keywords': []},
                'CIVIL': {'name': 'Civil Cases', 'keywords': []}
            },
            'training_config': {
                'test_size': 0.2,
                'validation_size': 0.1,
                'random_state': 42
            }
        }

    def add_training_example(self, text: str, category: str, source_file: str = "manual",
                           fir_data: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """
        Add a training example to the dataset.

        Args:
            text: FIR text content
            category: Correct category for this FIR
            source_file: Source file or identifier
            fir_data: Optional parsed FIR data
            metadata: Optional additional metadata
        """
        if category not in self.categories:
            logger.warning(f"Unknown category: {category}")
            return

        example = TrainingExample(
            text=text,
            category=category,
            source_file=source_file,
            fir_data=fir_data,
            metadata=metadata
        )

        self.training_examples.append(example)
        logger.info(f"Added training example for category: {category}")

    def add_validation_example(self, text: str, category: str, source_file: str = "manual",
                             fir_data: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """
        Add a validation example to the dataset.

        Args:
            text: FIR text content
            category: Correct category for this FIR
            source_file: Source file or identifier
            fir_data: Optional parsed FIR data
            metadata: Optional additional metadata
        """
        if category not in self.categories:
            logger.warning(f"Unknown category: {category}")
            return

        example = TrainingExample(
            text=text,
            category=category,
            source_file=source_file,
            fir_data=fir_data,
            metadata=metadata
        )

        self.validation_examples.append(example)
        logger.info(f"Added validation example for category: {category}")

    def load_training_data_from_file(self, filepath: str, category_column: str = 'category',
                                   text_column: str = 'text', source_column: str = 'source'):
        """
        Load training data from CSV or JSON file.

        Args:
            filepath: Path to the data file
            category_column: Name of category column
            text_column: Name of text column
            source_column: Name of source column
        """
        filepath = Path(filepath)

        try:
            if filepath.suffix.lower() == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix.lower() == '.json':
                df = pd.read_json(filepath)
            else:
                logger.error(f"Unsupported file format: {filepath.suffix}")
                return

            loaded_count = 0
            for _, row in df.iterrows():
                text = str(row.get(text_column, ''))
                category = str(row.get(category_column, ''))

                if text and category and category in self.categories:
                    source = str(row.get(source_column, 'file'))
                    self.add_training_example(text, category, source)
                    loaded_count += 1

            logger.info(f"Loaded {loaded_count} training examples from {filepath}")

        except Exception as e:
            logger.error(f"Error loading training data: {e}")

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
        """
        Prepare training data for model training.

        Returns:
            Tuple of (X, y, vectorizer) ready for training
        """
        if len(self.training_examples) < 10:
            raise ValueError("Insufficient training data. Need at least 10 examples.")

        # Extract texts and labels
        texts = [ex.text for ex in self.training_examples]
        categories = [ex.category for ex in self.training_examples]

        # Create vectorizer
        vectorizer = TfidfVectorizer(
            max_features=self.vectorizer_config.get('max_features', 5000),
            ngram_range=self.vectorizer_config.get('ngram_range', [1, 3]),
            stop_words='english',
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )

        # Vectorize texts
        X = vectorizer.fit_transform(texts)
        y = np.array(categories)

        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")

        return X, y, vectorizer

    def train_model(self, model_name: str = 'random_forest', test_size: float = 0.2) -> TrainingResult:
        """
        Train a specific model and evaluate performance.

        Args:
            model_name: Name of model to train ('random_forest', 'logistic_regression', 'svm', 'naive_bayes')
            test_size: Fraction of data to use for testing

        Returns:
            TrainingResult with performance metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        start_time = time.time()

        try:
            # Prepare data
            X, y, vectorizer = self.prepare_training_data()

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Get model
            model = self.models[model_name]

            # Train model
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            class_report = classification_report(y_test, y_pred, output_dict=True)

            training_time = time.time() - start_time

            # Create result
            result = TrainingResult(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_time=training_time,
                test_samples=len(y_test),
                feature_count=X.shape[1],
                model_size=len(pickle.dumps(model)),
                confusion_matrix=conf_matrix,
                classification_report=class_report
            )

            # Save model and vectorizer
            self._save_model(model_name, model, vectorizer)

            # Record in history
            self.training_history.append(asdict(result))

            logger.info(f"Model {model_name} trained successfully - Accuracy: {accuracy".3f"}")

            return result

        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            raise

    def train_all_models(self) -> Dict[str, TrainingResult]:
        """
        Train all available models and compare performance.

        Returns:
            Dictionary of model names to TrainingResult objects
        """
        results = {}

        for model_name in self.models.keys():
            try:
                result = self.train_model(model_name)
                results[model_name] = result
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")

        # Log comparison
        self._log_model_comparison(results)

        return results

    def cross_validate_model(self, model_name: str = 'random_forest', cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for a model.

        Args:
            model_name: Name of model to validate
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation scores
        """
        try:
            X, y, vectorizer = self.prepare_training_data()

            model = self.models[model_name]

            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')

            cv_results = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'min_accuracy': cv_scores.min(),
                'max_accuracy': cv_scores.max(),
                'cv_scores': cv_scores.tolist()
            }

            logger.info(f"Cross-validation for {model_name}: {cv_results['mean_accuracy']".3f"} ± {cv_results['std_accuracy']:".3f"")

            return cv_results

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}

    def analyze_training_data(self) -> Dict[str, Any]:
        """
        Analyze the training dataset for quality and balance.

        Returns:
            Dictionary with analysis results
        """
        if not self.training_examples:
            return {'error': 'No training data available'}

        # Category distribution
        category_counts = Counter(ex.category for ex in self.training_examples)

        # Text length analysis
        text_lengths = [len(ex.text) for ex in self.training_examples]
        text_length_stats = {
            'mean': np.mean(text_lengths),
            'median': np.median(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths),
            'std': np.std(text_lengths)
        }

        # Source distribution
        source_counts = Counter(ex.source_file for ex in self.training_examples)

        # Check for potential issues
        issues = []
        if len(category_counts) < 2:
            issues.append("Only one category present - need multiple categories for classification")

        if min(category_counts.values()) < 5:
            issues.append("Some categories have very few examples (< 5)")

        if text_length_stats['mean'] < 100:
            issues.append("Average text length is very short (< 100 characters)")

        # Category balance score (1.0 = perfectly balanced)
        total_examples = len(self.training_examples)
        expected_count = total_examples / len(category_counts)
        balance_score = 1.0 - np.std([count/expected_count for count in category_counts.values()])

        analysis = {
            'total_examples': total_examples,
            'num_categories': len(category_counts),
            'category_distribution': dict(category_counts),
            'text_length_stats': text_length_stats,
            'source_distribution': dict(source_counts),
            'balance_score': balance_score,
            'potential_issues': issues
        }

        return analysis

    def augment_training_data(self, target_per_category: int = 100):
        """
        Augment training data to balance categories and increase dataset size.

        Args:
            target_per_category: Target number of examples per category
        """
        # Group by category
        category_groups = defaultdict(list)
        for example in self.training_examples:
            category_groups[example.category].append(example)

        augmented_examples = []

        for category, examples in category_groups.items():
            current_count = len(examples)

            if current_count >= target_per_category:
                augmented_examples.extend(examples)
                continue

            # Add original examples
            augmented_examples.extend(examples)

            # Generate augmented versions for remaining slots
            remaining = target_per_category - current_count

            for i in range(min(remaining, current_count)):
                original = examples[i % current_count]

                # Simple text augmentation
                augmented_text = self._augment_text(original.text)

                augmented_example = TrainingExample(
                    text=augmented_text,
                    category=category,
                    source_file=f"{original.source_file}_augmented",
                    fir_data=original.fir_data,
                    metadata={'augmented_from': original.source_file, 'augmentation_method': 'simple'}
                )

                augmented_examples.append(augmented_example)

        self.training_examples = augmented_examples
        logger.info(f"Augmented training data to {len(self.training_examples)} examples")

    def _augment_text(self, text: str) -> str:
        """Simple text augmentation for FIR documents."""
        import random

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 2:
            return text

        # Randomly reorder some sentences (keeping first and last)
        middle_sentences = sentences[1:-1]
        if middle_sentences:
            random.shuffle(middle_sentences)
            augmented_sentences = [sentences[0]] + middle_sentences + [sentences[-1]]
            return '. '.join(augmented_sentences)

        return text

    def _save_model(self, model_name: str, model: Any, vectorizer: TfidfVectorizer):
        """Save trained model and vectorizer."""
        try:
            model_path = self.training_dir / 'models' / model_name
            model_path.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, model_path / 'model.joblib')
            joblib.dump(vectorizer, model_path / 'vectorizer.joblib')

            logger.info(f"Saved model {model_name} to {model_path}")

        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")

    def _load_training_data(self):
        """Load existing training data from disk."""
        try:
            data_file = self.training_dir / 'training_examples.pkl'
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    self.training_examples = pickle.load(f)

            validation_file = self.training_dir / 'validation_examples.pkl'
            if validation_file.exists():
                with open(validation_file, 'rb') as f:
                    self.validation_examples = pickle.load(f)

            logger.info(f"Loaded {len(self.training_examples)} training examples and {len(self.validation_examples)} validation examples")

        except Exception as e:
            logger.warning(f"Error loading training data: {e}")

    def save_training_data(self):
        """Save training data to disk."""
        try:
            # Save training examples
            with open(self.training_dir / 'training_examples.pkl', 'wb') as f:
                pickle.dump(self.training_examples, f)

            # Save validation examples
            with open(self.training_dir / 'validation_examples.pkl', 'wb') as f:
                pickle.dump(self.validation_examples, f)

            # Save analysis
            analysis = self.analyze_training_data()
            with open(self.training_dir / 'data_analysis.json', 'w') as f:
                json.dump(analysis, f, indent=2)

            logger.info("Training data saved successfully")

        except Exception as e:
            logger.error(f"Error saving training data: {e}")

    def _log_model_comparison(self, results: Dict[str, TrainingResult]):
        """Log comparison of model performances."""
        if not results:
            return

        logger.info("Model Performance Comparison:")
        logger.info("-" * 50)

        for model_name, result in results.items():
            logger.info(f"{model_name"20"} | Accuracy: {result.accuracy".3f"} | F1: {result.f1_score".3f"} | Time: {result.training_time".2f"}s")

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1].f1_score)
        logger.info(f"Best performing model: {best_model[0]} (F1: {best_model[1].f1_score".3f"})")

    def export_training_report(self, filepath: str):
        """Export comprehensive training report."""
        try:
            report = {
                'data_analysis': self.analyze_training_data(),
                'training_history': self.training_history,
                'categories': self.categories,
                'config': self.categories_config
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Training report exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting training report: {e}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training data and models."""
        summary = {
            'total_training_examples': len(self.training_examples),
            'total_validation_examples': len(self.validation_examples),
            'categories': list(self.categories.keys()),
            'available_models': list(self.models.keys()),
            'data_analysis': self.analyze_training_data(),
            'training_history_count': len(self.training_history)
        }

        return summary

# Utility functions for data collection
def collect_training_data_from_firs(fir_files: List[str], output_dir: str = 'models/training_data'):
    """
    Collect training data from existing FIR files.

    Args:
        fir_files: List of FIR file paths
        output_dir: Output directory for training data
    """
    trainer = CategoryTrainer(training_dir=output_dir)

    for fir_file in fir_files:
        try:
            # This would need to be implemented based on FIR file format
            # For now, just log the file
            logger.info(f"Processing FIR file: {fir_file}")
            # TODO: Implement FIR file parsing and category extraction

        except Exception as e:
            logger.error(f"Error processing {fir_file}: {e}")

    trainer.save_training_data()
    return trainer

def create_synthetic_training_data(categories: Dict[str, Any], examples_per_category: int = 50) -> List[TrainingExample]:
    """
    Create synthetic training data for testing purposes.

    Args:
        categories: Category definitions
        examples_per_category: Number of examples to generate per category

    Returns:
        List of synthetic training examples
    """
    examples = []

    # Templates for different categories
    templates = {
        'CRIMINAL': [
            "The complainant reported that {action} occurred at {location}. The accused {details}. FIR registered under IPC {section}.",
            "Victim reported {crime} by {accused}. Incident took place on {date}. Police investigating under {section}.",
            "Complaint filed regarding {offense}. Accused {name} allegedly {action}. Case registered at {station}."
        ],
        'CIVIL': [
            "Civil suit filed for {dispute} regarding {property}. Parties seeking {relief} from court.",
            "Property dispute between {parties}. Matter pending in civil court for {years} years.",
            "Application for {relief} in civil matter. Dispute regarding {issue} valued at {amount}."
        ],
        'TRAFFIC': [
            "Motor vehicle accident reported at {location}. Vehicle {number} collided with {object}.",
            "Traffic violation by driver {name}. Case registered for {violation} under MV Act.",
            "Hit and run incident reported. Vehicle {description} fled the scene after accident."
        ],
        'CYBER': [
            "Online fraud reported through {platform}. Amount lost: {amount}. Cyber cell investigating.",
            "Hacking incident reported. Account {details} compromised. FIR under IT Act {section}.",
            "Digital evidence collected for {crime}. Online transaction fraud of {amount}."
        ],
        'ECONOMIC': [
            "Financial fraud case registered. Amount involved: {amount}. Accused {name} arrested.",
            "Cheating case under IPC {section}. Victim lost {amount} in fraudulent transaction.",
            "Economic offense reported at {bank}. Suspicious transaction of {amount} detected."
        ],
        'NARCOTICS': [
            "Drug possession case. {quantity} of {substance} recovered from accused {name}.",
            "Narcotics trafficking case under NDPS Act. Contraband seized from {location}.",
            "Drug peddling incident reported. Accused arrested with {substance} worth {amount}."
        ]
    }

    # Sample data for template filling
    sample_data = {
        'action': ['theft occurred', 'assault took place', 'burglary happened', 'robbery committed'],
        'location': ['market area', 'residential colony', 'commercial complex', 'public place'],
        'crime': ['theft', 'assault', 'burglary', 'cheating'],
        'accused': ['unknown person', 'identified suspect', 'named individual'],
        'name': ['John Doe', 'Ram Kumar', 'Priya Sharma', 'Rajesh Singh'],
        'station': ['City Police Station', 'Central Police Station', 'Local Police Station'],
        'date': ['15/03/2024', '01/01/2024', '22/12/2023'],
        'section': ['379', '420', '406', '302'],
        'dispute': ['property ownership', 'contract breach', 'partnership dispute'],
        'property': ['land parcel', 'commercial property', 'residential building'],
        'relief': ['injunction', 'damages', 'specific performance'],
        'parties': ['two families', 'business partners', 'landlord and tenant'],
        'years': ['2', '5', '3', '1'],
        'issue': ['boundary dispute', 'title claim', 'possession rights'],
        'amount': ['₹50,000', '₹1,00,000', '₹2,50,000'],
        'number': ['DL01CA1234', 'UP32AB5678', 'HR26DK9012'],
        'object': ['another vehicle', 'pedestrian', 'stationary object'],
        'violation': ['rash driving', 'drunken driving', 'signal jumping'],
        'description': ['white car', 'blue motorcycle', 'red truck'],
        'platform': ['banking app', 'social media', 'e-commerce site'],
        'details': ['credentials stolen', 'unauthorized access', 'data breach'],
        'bank': ['State Bank', 'National Bank', 'Private Bank'],
        'quantity': ['100 grams', '50 tablets', '2 kg'],
        'substance': ['heroin', 'cocaine', 'marijuana']
    }

    for category, templates_list in templates.items():
        if category not in categories:
            continue

        for i in range(examples_per_category):
            template = np.random.choice(templates_list)

            # Fill template with random data
            filled_text = template
            for key, values in sample_data.items():
                placeholder = f"{{{key}}}"
                if placeholder in filled_text:
                    filled_text = filled_text.replace(placeholder, np.random.choice(values))

            example = TrainingExample(
                text=filled_text,
                category=category,
                source_file=f"synthetic_{category}_{i+1}",
                metadata={'synthetic': True, 'template_used': template}
            )

            examples.append(example)

    logger.info(f"Generated {len(examples)} synthetic training examples")
    return examples