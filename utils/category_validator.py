"""
FIR Category Validation and Quality Assessment System
Handles model validation, performance monitoring, and quality metrics for FIR categorization.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of model validation."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    kappa_score: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    validation_time: float
    test_set_size: int
    cross_validation_scores: Optional[List[float]] = None

@dataclass
class QualityMetrics:
    """Quality metrics for categorization system."""
    overall_accuracy: float
    category_accuracies: Dict[str, float]
    confidence_distribution: Dict[str, int]
    processing_time_stats: Dict[str, float]
    error_analysis: Dict[str, Any]
    data_quality_score: float
    model_stability_score: float

@dataclass
class PredictionAnalysis:
    """Analysis of individual predictions."""
    prediction_id: str
    true_category: str
    predicted_category: str
    confidence: float
    text_length: int
    category_difficulty: float
    is_correct: bool
    error_type: Optional[str] = None

class CategoryValidator:
    """
    Validation and quality assessment system for FIR categorization.
    """

    def __init__(self, categories_file: str = 'models/fir_categories.json', models_dir: str = 'models/training_data/models'):
        """
        Initialize the category validator.

        Args:
            categories_file: Path to categories configuration file
            models_dir: Directory containing trained models
        """
        self.categories_file = Path(categories_file)
        self.models_dir = Path(models_dir)

        # Load category definitions
        self.categories_config = self._load_categories()
        self.categories = self.categories_config.get('categories', {})

        # Available models
        self.trained_models = {}
        self._load_available_models()

        # Validation results storage
        self.validation_history = []
        self.prediction_analyses = []

        # Quality tracking
        self.quality_metrics = {}

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
                'CRIMINAL': {'name': 'Criminal Cases'},
                'CIVIL': {'name': 'Civil Cases'}
            }
        }

    def _load_available_models(self):
        """Load available trained models."""
        if not self.models_dir.exists():
            return

        for model_path in self.models_dir.iterdir():
            if model_path.is_dir():
                model_name = model_path.name
                try:
                    model = joblib.load(model_path / 'model.joblib')
                    vectorizer = joblib.load(model_path / 'vectorizer.joblib')
                    self.trained_models[model_name] = {
                        'model': model,
                        'vectorizer': vectorizer,
                        'path': model_path
                    }
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")

    def validate_model(self, model_name: str, test_texts: List[str], test_categories: List[str],
                      cv_folds: int = 5) -> ValidationResult:
        """
        Validate a trained model against test data.

        Args:
            model_name: Name of model to validate
            test_texts: List of test FIR texts
            test_categories: List of true categories
            cv_folds: Number of cross-validation folds

        Returns:
            ValidationResult with performance metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")

        start_time = time.time()

        try:
            model_info = self.trained_models[model_name]
            model = model_info['model']
            vectorizer = model_info['vectorizer']

            # Vectorize test texts
            X_test = vectorizer.transform(test_texts)
            y_test = np.array(test_categories)

            # Get predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

            # Cohen's Kappa for inter-rater agreement
            kappa = cohen_kappa_score(y_test, y_pred)

            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()

            # Detailed classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Cross-validation scores
            cv_scores = cross_val_score(model, X_test, y_test, cv=cv_folds, scoring='accuracy')

            validation_time = time.time() - start_time

            result = ValidationResult(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                kappa_score=kappa,
                confusion_matrix=conf_matrix,
                classification_report=class_report,
                validation_time=validation_time,
                test_set_size=len(test_texts),
                cross_validation_scores=cv_scores.tolist()
            )

            # Store result
            self.validation_history.append(asdict(result))

            logger.info(f"Model {model_name} validation completed - Accuracy: {accuracy".3f"}")

            return result

        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            raise

    def analyze_predictions(self, model_name: str, texts: List[str], true_categories: List[str],
                          predictions: List[str], confidences: List[float]) -> List[PredictionAnalysis]:
        """
        Analyze individual predictions for error patterns and insights.

        Args:
            model_name: Name of model used for predictions
            texts: Original FIR texts
            true_categories: True categories
            predictions: Predicted categories
            confidences: Prediction confidences

        Returns:
            List of PredictionAnalysis objects
        """
        analyses = []

        for i, (text, true_cat, pred_cat, conf) in enumerate(zip(texts, true_categories, predictions, confidences)):
            analysis = PredictionAnalysis(
                prediction_id=f"{model_name}_{i}",
                true_category=true_cat,
                predicted_category=pred_cat,
                confidence=conf,
                text_length=len(text),
                category_difficulty=self._calculate_category_difficulty(true_cat),
                is_correct=true_cat == pred_cat
            )

            if not analysis.is_correct:
                analysis.error_type = self._classify_error_type(true_cat, pred_cat, text, conf)

            analyses.append(analysis)

        self.prediction_analyses.extend(analyses)
        return analyses

    def _calculate_category_difficulty(self, category: str) -> float:
        """Calculate difficulty score for a category based on historical performance."""
        if not self.prediction_analyses:
            return 0.5  # Default medium difficulty

        # Get historical accuracy for this category
        category_analyses = [a for a in self.prediction_analyses if a.true_category == category]
        if not category_analyses:
            return 0.5

        accuracy = sum(1 for a in category_analyses if a.is_correct) / len(category_analyses)
        return 1.0 - accuracy  # Higher score = more difficult

    def _classify_error_type(self, true_cat: str, pred_cat: str, text: str, confidence: float) -> str:
        """Classify the type of prediction error."""
        # Similar category error
        if self._are_similar_categories(true_cat, pred_cat):
            return 'similar_category'

        # Low confidence error
        if confidence < 0.3:
            return 'low_confidence'

        # Text length related error
        if len(text) < 100:
            return 'short_text'

        # Category imbalance error
        if self._is_rare_category(true_cat):
            return 'rare_category'

        return 'other'

    def _are_similar_categories(self, cat1: str, cat2: str) -> bool:
        """Check if two categories are similar."""
        similar_groups = [
            {'CRIMINAL', 'CYBER', 'ECONOMIC'},
            {'CIVIL', 'MISCELLANEOUS'},
            {'TRAFFIC', 'MISCELLANEOUS'}
        ]

        for group in similar_groups:
            if cat1 in group and cat2 in group:
                return True

        return False

    def _is_rare_category(self, category: str) -> bool:
        """Check if category is rare based on historical data."""
        if not self.prediction_analyses:
            return False

        category_counts = Counter(a.true_category for a in self.prediction_analyses)
        total_count = sum(category_counts.values())

        if total_count == 0:
            return False

        category_frequency = category_counts[category] / total_count
        return category_frequency < 0.1  # Less than 10% of data

    def generate_quality_report(self, model_name: str = None) -> QualityMetrics:
        """
        Generate comprehensive quality report for the categorization system.

        Args:
            model_name: Optional specific model to analyze

        Returns:
            QualityMetrics object with comprehensive quality assessment
        """
        if not self.prediction_analyses:
            logger.warning("No prediction analyses available for quality report")
            return self._get_empty_quality_metrics()

        # Filter by model if specified
        analyses = self.prediction_analyses
        if model_name:
            analyses = [a for a in analyses if model_name in a.prediction_id]

        if not analyses:
            return self._get_empty_quality_metrics()

        # Overall accuracy
        correct_predictions = sum(1 for a in analyses if a.is_correct)
        overall_accuracy = correct_predictions / len(analyses)

        # Per-category accuracy
        category_accuracies = {}
        for category in self.categories.keys():
            cat_analyses = [a for a in analyses if a.true_category == category]
            if cat_analyses:
                cat_accuracy = sum(1 for a in cat_analyses if a.is_correct) / len(cat_analyses)
                category_accuracies[category] = cat_accuracy

        # Confidence distribution
        confidence_bins = [0, 0.3, 0.6, 0.8, 1.0]
        confidence_dist = {}
        for i in range(len(confidence_bins) - 1):
            bin_start, bin_end = confidence_bins[i], confidence_bins[i + 1]
            bin_name = f"{bin_start".1f"}-{bin_end".1f"}"
            count = sum(1 for a in analyses if bin_start <= a.confidence < bin_end)
            confidence_dist[bin_name] = count

        # Processing time statistics
        processing_times = [getattr(a, 'processing_time', 0) for a in analyses if hasattr(a, 'processing_time')]
        time_stats = {}
        if processing_times:
            time_stats = {
                'mean': np.mean(processing_times),
                'median': np.median(processing_times),
                'min': np.min(processing_times),
                'max': np.max(processing_times),
                'std': np.std(processing_times)
            }

        # Error analysis
        error_analyses = [a for a in analyses if not a.is_correct]
        error_types = Counter(a.error_type for a in error_analyses if a.error_type)

        error_analysis = {
            'total_errors': len(error_analyses),
            'error_rate': len(error_analyses) / len(analyses),
            'error_types': dict(error_types),
            'most_common_errors': error_types.most_common(3)
        }

        # Data quality score (based on text length, category balance, etc.)
        data_quality_score = self._calculate_data_quality_score(analyses)

        # Model stability score (based on confidence consistency)
        model_stability_score = self._calculate_model_stability_score(analyses)

        quality_metrics = QualityMetrics(
            overall_accuracy=overall_accuracy,
            category_accuracies=category_accuracies,
            confidence_distribution=confidence_dist,
            processing_time_stats=time_stats,
            error_analysis=error_analysis,
            data_quality_score=data_quality_score,
            model_stability_score=model_stability_score
        )

        self.quality_metrics = asdict(quality_metrics)
        return quality_metrics

    def _get_empty_quality_metrics(self) -> QualityMetrics:
        """Get empty quality metrics when no data is available."""
        return QualityMetrics(
            overall_accuracy=0.0,
            category_accuracies={},
            confidence_distribution={},
            processing_time_stats={},
            error_analysis={'total_errors': 0, 'error_rate': 0.0, 'error_types': {}},
            data_quality_score=0.0,
            model_stability_score=0.0
        )

    def _calculate_data_quality_score(self, analyses: List[PredictionAnalysis]) -> float:
        """Calculate overall data quality score."""
        if not analyses:
            return 0.0

        scores = []

        # Text length quality (prefer adequate length)
        text_lengths = [a.text_length for a in analyses]
        avg_length = np.mean(text_lengths)
        length_score = min(avg_length / 500, 1.0)  # Normalize to 500 chars
        scores.append(length_score)

        # Category balance quality
        category_counts = Counter(a.true_category for a in analyses)
        if len(category_counts) > 1:
            frequencies = np.array(list(category_counts.values()))
            # Lower standard deviation = better balance
            balance_score = 1.0 - (np.std(frequencies) / np.mean(frequencies))
            scores.append(max(balance_score, 0.0))
        else:
            scores.append(0.5)  # Neutral score for single category

        # Confidence consistency
        confidences = [a.confidence for a in analyses]
        confidence_std = np.std(confidences)
        confidence_score = 1.0 - min(confidence_std, 1.0)
        scores.append(confidence_score)

        return np.mean(scores)

    def _calculate_model_stability_score(self, analyses: List[PredictionAnalysis]) -> float:
        """Calculate model stability based on confidence patterns."""
        if not analyses:
            return 0.0

        # Analyze confidence patterns for correct vs incorrect predictions
        correct_confidences = [a.confidence for a in analyses if a.is_correct]
        incorrect_confidences = [a.confidence for a in analyses if not a.is_correct]

        if not correct_confidences or not incorrect_confidences:
            return 0.5

        # Model is stable if correct predictions generally have higher confidence
        correct_mean = np.mean(correct_confidences)
        incorrect_mean = np.mean(incorrect_confidences)

        if incorrect_mean == 0:
            return 1.0 if correct_mean > 0 else 0.0

        stability_ratio = correct_mean / incorrect_mean
        return min(stability_ratio, 1.0)

    def create_confusion_matrix_plot(self, validation_result: ValidationResult, save_path: Optional[str] = None):
        """
        Create confusion matrix visualization.

        Args:
            validation_result: ValidationResult to visualize
            save_path: Optional path to save the plot
        """
        try:
            # Create confusion matrix
            cm = np.array(validation_result.confusion_matrix)
            categories = list(self.categories.keys())

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=categories, yticklabels=categories)
            plt.title(f'Confusion Matrix - {validation_result.model_name}')
            plt.xlabel('Predicted Category')
            plt.ylabel('True Category')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {e}")

    def generate_error_analysis_report(self, model_name: str = None) -> Dict[str, Any]:
        """
        Generate detailed error analysis report.

        Args:
            model_name: Optional specific model to analyze

        Returns:
            Dictionary with error analysis results
        """
        # Filter analyses
        analyses = self.prediction_analyses
        if model_name:
            analyses = [a for a in analyses if model_name in a.prediction_id]

        error_analyses = [a for a in analyses if not a.is_correct]

        if not error_analyses:
            return {'message': 'No errors to analyze'}

        # Error type distribution
        error_types = Counter(a.error_type for a in error_analyses if a.error_type)

        # Category-specific error rates
        category_errors = defaultdict(list)
        for analysis in error_analyses:
            category_errors[analysis.true_category].append(analysis)

        category_error_rates = {}
        for category, errors in category_errors.items():
            total_category = sum(1 for a in analyses if a.true_category == category)
            error_rate = len(errors) / total_category if total_category > 0 else 0
            category_error_rates[category] = error_rate

        # Most confused category pairs
        confusion_pairs = Counter()
        for analysis in error_analyses:
            pair = (analysis.true_category, analysis.predicted_category)
            confusion_pairs[pair] += 1

        # Text length analysis for errors
        error_lengths = [a.text_length for a in error_analyses]
        length_stats = {
            'mean': np.mean(error_lengths),
            'median': np.median(error_lengths),
            'min': np.min(error_lengths),
            'max': np.max(error_lengths)
        }

        report = {
            'total_errors': len(error_analyses),
            'error_rate': len(error_analyses) / len(analyses),
            'error_type_distribution': dict(error_types),
            'category_error_rates': category_error_rates,
            'most_confused_pairs': confusion_pairs.most_common(5),
            'error_text_length_stats': length_stats,
            'recommendations': self._generate_error_recommendations(error_types, category_error_rates)
        }

        return report

    def _generate_error_recommendations(self, error_types: Counter, category_error_rates: Dict[str, float]) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []

        # Low confidence errors
        if error_types.get('low_confidence', 0) > 0:
            recommendations.append(
                "Consider increasing confidence threshold or improving feature extraction for low-confidence predictions"
            )

        # Similar category errors
        if error_types.get('similar_category', 0) > 0:
            recommendations.append(
                "Consider refining category definitions or adding distinguishing features for similar categories"
            )

        # Rare category errors
        if error_types.get('rare_category', 0) > 0:
            recommendations.append(
                "Consider collecting more training data for rare categories or using data augmentation"
            )

        # Short text errors
        if error_types.get('short_text', 0) > 0:
            recommendations.append(
                "Consider preprocessing or feature engineering improvements for short texts"
            )

        # High error rate categories
        high_error_categories = [cat for cat, rate in category_error_rates.items() if rate > 0.3]
        if high_error_categories:
            recommendations.append(
                f"Focus improvement efforts on high-error categories: {', '.join(high_error_categories)}"
            )

        return recommendations

    def export_validation_report(self, filepath: str, model_name: str = None):
        """
        Export comprehensive validation report to file.

        Args:
            filepath: Path to save the report
            model_name: Optional specific model to include in report
        """
        try:
            report = {
                'quality_metrics': asdict(self.generate_quality_report(model_name)),
                'error_analysis': self.generate_error_analysis_report(model_name),
                'validation_history': self.validation_history[-10:],  # Last 10 validations
                'categories': self.categories,
                'available_models': list(self.trained_models.keys()),
                'generated_at': time.time()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Validation report exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting validation report: {e}")

    def benchmark_models(self, test_texts: List[str], test_categories: List[str]) -> Dict[str, ValidationResult]:
        """
        Benchmark all available models against test data.

        Args:
            test_texts: Test FIR texts
            test_categories: True categories

        Returns:
            Dictionary of model names to ValidationResult objects
        """
        results = {}

        for model_name in self.trained_models.keys():
            try:
                result = self.validate_model(model_name, test_texts, test_categories)
                results[model_name] = result
            except Exception as e:
                logger.warning(f"Benchmark failed for {model_name}: {e}")

        # Log comparison
        self._log_benchmark_comparison(results)

        return results

    def _log_benchmark_comparison(self, results: Dict[str, ValidationResult]):
        """Log comparison of model performances."""
        if not results:
            return

        logger.info("Model Benchmark Comparison:")
        logger.info("-" * 60)
        logger.info(f"{'Model'"<20"} {'Accuracy'"<10"} {'F1-Score'"<10"} {'Kappa'"<8"} {'Time'"<8"}")
        logger.info("-" * 60)

        for model_name, result in results.items():
            logger.info(f"{model_name"<20"} {result.accuracy".3f"}     {result.f1_score".3f"}     {result.kappa_score".3f"}  {result.validation_time".2f"}s")

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1].f1_score)
        logger.info(f"Best performing model: {best_model[0]} (F1: {best_model[1].f1_score".3f"})")

    def get_model_recommendations(self) -> List[str]:
        """Get recommendations for improving model performance."""
        recommendations = []

        if not self.quality_metrics:
            return ["Run quality assessment first to get recommendations"]

        quality = self.quality_metrics

        # Based on overall accuracy
        if quality['overall_accuracy'] < 0.7:
            recommendations.append("Overall accuracy is below 70% - consider collecting more training data")

        # Based on data quality
        if quality['data_quality_score'] < 0.6:
            recommendations.append("Data quality score is low - review training data for quality issues")

        # Based on model stability
        if quality['model_stability_score'] < 0.6:
            recommendations.append("Model stability is low - consider retraining with better data balance")

        # Based on error analysis
        error_analysis = quality.get('error_analysis', {})
        if error_analysis.get('error_rate', 0) > 0.3:
            recommendations.append("High error rate detected - focus on error-prone categories")

        # Category-specific recommendations
        category_accuracies = quality.get('category_accuracies', {})
        low_accuracy_categories = [cat for cat, acc in category_accuracies.items() if acc < 0.6]

        if low_accuracy_categories:
            recommendations.append(f"Improve accuracy for low-performing categories: {', '.join(low_accuracy_categories)}")

        return recommendations if recommendations else ["System performance is within acceptable ranges"]

    def reset_validation_data(self):
        """Reset all validation data and history."""
        self.validation_history.clear()
        self.prediction_analyses.clear()
        self.quality_metrics.clear()
        logger.info("Validation data reset")

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation status."""
        summary = {
            'available_models': list(self.trained_models.keys()),
            'validation_history_count': len(self.validation_history),
            'prediction_analyses_count': len(self.prediction_analyses),
            'quality_metrics_available': bool(self.quality_metrics),
            'categories': list(self.categories.keys())
        }

        return summary