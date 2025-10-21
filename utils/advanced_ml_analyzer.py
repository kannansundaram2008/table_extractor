"""
Advanced ML Analyzer for FIR Document Processing.
Provides sophisticated analysis using deep learning and NLP techniques.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pickle
import json
from datetime import datetime
import re

# ML and NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import textstat

# Deep learning imports (optional)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, pipeline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using sklearn-only models")

# Configure logging
logger = logging.getLogger(__name__)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Advanced text preprocessing for FIR documents.
    Handles legal terminology, Indian names, and document-specific patterns.
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Legal domain specific stopwords and terms
        self.legal_stopwords = {
            'fir', 'police', 'station', 'complainant', 'accused', 'victim',
            'section', 'ipc', 'cr', 'no', 'date', 'time', 'place', 'occurrence',
            'report', 'investigation', 'case', 'under', 'against'
        }
        self.stop_words.update(self.legal_stopwords)

        # Indian name patterns for better NER
        self.indian_name_prefixes = {
            'shri', 'smt', 'kumari', 'kumar', 'singh', 'kumar', 'sharma',
            'verma', 'gupta', 'jain', 'patel', 'yadav', 'thakur', 'rao'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transform text data with advanced preprocessing."""
        if isinstance(X, str):
            return self._preprocess_text(X)
        elif isinstance(X, list):
            return [self._preprocess_text(text) for text in X]
        else:
            return X

