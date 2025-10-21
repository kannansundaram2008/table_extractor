"""
Enhanced Named Entity Recognition (NER) system for FIR document processing.
Combines spaCy with custom rules and confidence scoring for legal domain entities.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import spacy
from spacy.language import Language
from spacy.tokens import Doc

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents a named entity with confidence score."""
    text: str
    label: str
    start: int
    end: int
    confidence: float

@dataclass
class NERResult:
    """Container for NER extraction results."""
    entities: List[Entity]
    confidence_scores: Dict[str, float]
    processing_time: float

class EnhancedNER:
    """
    Enhanced NER system combining spaCy with custom legal domain rules.
    """

    def __init__(self, model_name: str = 'en_core_web_sm', confidence_threshold: float = 0.7):
        """
        Initialize the enhanced NER system.

        Args:
            model_name: spaCy model to use
            confidence_threshold: Minimum confidence for entity acceptance
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nlp = None
        self.custom_patterns = self._load_custom_patterns()
        self.legal_entity_types = {
            'PERSON': ['complainant', 'victim', 'accused', 'witness', 'officer'],
            'ORG': ['police_station', 'court', 'department', 'organization'],
            'GPE': ['location', 'city', 'state', 'district', 'village'],
            'LAW': ['ipc_section', 'bns_section', 'crpc_section', 'act'],
            'DATE': ['occurrence_date', 'report_date', 'hearing_date'],
            'MONEY': ['fine_amount', 'bail_amount', 'property_value'],
            'QUANTITY': ['property_quantity', 'age']
        }

        self._initialize_spacy()

    def _initialize_spacy(self) -> None:
        """Initialize spaCy model with error handling."""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Successfully loaded spaCy model: {self.model_name}")
        except OSError:
            logger.warning(f"spaCy model {self.model_name} not found. Installing...")
            try:
                spacy.cli.download(self.model_name)
                self.nlp = spacy.load(self.model_name)
                logger.info(f"Successfully downloaded and loaded spaCy model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {e}")
                self.nlp = None
        except Exception as e:
            logger.error(f"Error initializing spaCy: {e}")
            self.nlp = None

    def _load_custom_patterns(self) -> Dict[str, List[str]]:
        """Load custom regex patterns for legal domain entities."""
        return {
            'indian_names': [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Multi-word names
                r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b',  # Names with initials
            ],
            'police_stations': [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Police\s+Station\b',
                r'\bPS\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            ],
            'ipc_sections': [
                r'\bIPC\s+\d+(?:\([^)]+\))?(?:\s*,\s*\d+(?:\([^)]+\))?)*\b',
                r'\bSection\s+\d+(?:\([^)]+\))?\s+of\s+IPC\b',
            ],
            'cr_numbers': [
                r'\b\d{1,4}/\d{2,4}\b',
                r'\bCR\s*No\.?\s*\d{1,4}/\d{2,4}\b',
            ],
            'dates_indian': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
            ]
        }

    def extract_entities(self, text: str) -> NERResult:
        """
        Extract named entities from text using enhanced approach.

        Args:
            text: Input text to process

        Returns:
            NERResult containing entities and confidence scores
        """
        import time
        start_time = time.time()

        if not text or not isinstance(text, str):
            return NERResult([], {}, 0.0)

        # Preprocessing
        cleaned_text = self._preprocess_text(text)

        entities = []

        # spaCy NER if available
        if self.nlp:
            spacy_entities = self._extract_with_spacy(cleaned_text)
            entities.extend(spacy_entities)

        # Custom rule-based extraction
        rule_entities = self._extract_with_rules(cleaned_text, text)
        entities.extend(rule_entities)

        # Remove duplicates and merge overlapping entities
        entities = self._deduplicate_entities(entities)

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(entities, cleaned_text)

        processing_time = time.time() - start_time

        return NERResult(entities, confidence_scores, processing_time)

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for better NER."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Normalize legal abbreviations
        text = self._normalize_legal_terms(text)

        # Handle common OCR errors in legal documents
        text = self._fix_common_ocr_errors(text)

        return text.strip()

    def _normalize_legal_terms(self, text: str) -> str:
        """Normalize common legal abbreviations and terms."""
        normalizations = {
            r'\bIPC\b': 'Indian Penal Code',
            r'\bCRPC\b': 'Criminal Procedure Code',
            r'\bBNS\b': 'Bharatiya Nyaya Sanhita',
            r'\bPS\b': 'Police Station',
            r'\bIO\b': 'Investigating Officer',
            r'\bSI\b': 'Sub Inspector',
            r'\bASI\b': 'Assistant Sub Inspector',
        }

        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors in legal documents."""
        fixes = {
            r'\b1\b': 'I',  # Common OCR error
            r'\b0\b': 'O',  # Common OCR error
            r'\b5\b': 'S',  # Common OCR error
        }

        for pattern, replacement in fixes.items():
            text = re.sub(pattern, replacement, text)

        return text

    def _extract_with_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy."""
        if not self.nlp:
            return []

        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                # Map spaCy labels to our legal domain labels
                mapped_label = self._map_spacy_label(ent.label_)

                # Calculate confidence based on entity properties
                confidence = self._calculate_spacy_confidence(ent, doc)

                if confidence >= self.confidence_threshold:
                    entity = Entity(
                        text=ent.text,
                        label=mapped_label,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=confidence
                    )
                    entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"Error in spaCy NER extraction: {e}")
            return []

    def _map_spacy_label(self, spacy_label: str) -> str:
        """Map spaCy entity labels to legal domain labels."""
        mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORG',
            'GPE': 'GPE',
            'LOC': 'GPE',
            'MONEY': 'MONEY',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'QUANTITY': 'QUANTITY',
            'CARDINAL': 'QUANTITY',
        }
        return mapping.get(spacy_label, 'MISC')

    def _calculate_spacy_confidence(self, ent, doc) -> float:
        """Calculate confidence score for spaCy entity."""
        # Base confidence from entity recognition
        base_confidence = 0.8

        # Adjust based on entity properties
        if len(ent.text.strip()) < 3:
            base_confidence -= 0.2  # Too short
        elif len(ent.text.split()) > 4:
            base_confidence -= 0.1  # Too long

        # Check if entity appears in legal context
        if self._is_legal_context(ent, doc):
            base_confidence += 0.1

        return max(0.0, min(1.0, base_confidence))

    def _is_legal_context(self, ent, doc) -> bool:
        """Check if entity appears in legal context."""
        # Look for legal keywords within 5 tokens
        legal_keywords = ['police', 'station', 'fir', 'case', 'ipc', 'section', 'accused', 'complainant']
        start = max(0, ent.start - 5)
        end = min(len(doc), ent.end + 5)

        context_tokens = [token.text.lower() for token in doc[start:end]]
        return any(keyword in context_tokens for keyword in legal_keywords)

    def _extract_with_rules(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Extract entities using custom rule-based patterns."""
        entities = []

        # Extract persons using custom patterns
        person_entities = self._extract_persons(cleaned_text, original_text)
        entities.extend(person_entities)

        # Extract locations
        location_entities = self._extract_locations(cleaned_text, original_text)
        entities.extend(location_entities)

        # Extract legal sections
        legal_entities = self._extract_legal_sections(cleaned_text, original_text)
        entities.extend(legal_entities)

        # Extract dates
        date_entities = self._extract_dates(cleaned_text, original_text)
        entities.extend(date_entities)

        return entities

    def _extract_persons(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Extract person names using custom patterns."""
        entities = []

        for pattern in self.custom_patterns['indian_names']:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                # Skip if already extracted by spaCy
                if self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    continue

                confidence = self._calculate_person_confidence(match.group(), cleaned_text)

                if confidence >= self.confidence_threshold:
                    entity = Entity(
                        text=match.group(),
                        label='PERSON',
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence
                    )
                    entities.append(entity)

        return entities

    def _extract_locations(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Extract location entities."""
        entities = []

        # Police station patterns
        for pattern in self.custom_patterns['police_stations']:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    entity = Entity(
                        text=match.group(),
                        label='ORG',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9  # High confidence for structured patterns
                    )
                    entities.append(entity)

        # General location patterns
        location_indicators = ['at', 'in', 'near', 'from', 'to']
        for indicator in location_indicators:
            pattern = rf'\b{indicator}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                location_text = match.group(1)
                if not self._is_overlapping_with_existing(match.start(1), match.end(1), entities):
                    confidence = self._calculate_location_confidence(location_text, cleaned_text)
                    if confidence >= self.confidence_threshold:
                        entity = Entity(
                            text=location_text,
                            label='GPE',
                            start=match.start(1),
                            end=match.end(1),
                            confidence=confidence
                        )
                        entities.append(entity)

        return entities

    def _extract_legal_sections(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Extract legal section references."""
        entities = []

        for pattern in self.custom_patterns['ipc_sections']:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    entity = Entity(
                        text=match.group(),
                        label='LAW',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95  # High confidence for legal sections
                    )
                    entities.append(entity)

        return entities

    def _extract_dates(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Extract date entities using custom patterns."""
        entities = []

        for pattern in self.custom_patterns['dates_indian']:
            matches = re.finditer(pattern, original_text)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    confidence = self._calculate_date_confidence(match.group())
                    if confidence >= self.confidence_threshold:
                        entity = Entity(
                            text=match.group(),
                            label='DATE',
                            start=match.start(),
                            end=match.end(),
                            confidence=confidence
                        )
                        entities.append(entity)

        return entities

    def _calculate_person_confidence(self, name: str, context: str) -> float:
        """Calculate confidence score for person name."""
        confidence = 0.7  # Base confidence

        # Increase confidence for proper capitalization
        if name.istitle():
            confidence += 0.1

        # Increase for multiple words (likely full name)
        if len(name.split()) >= 2:
            confidence += 0.1

        # Check for legal context
        legal_context_words = ['complainant', 'victim', 'accused', 'witness', 's/o', 'd/o', 'w/o']
        context_lower = context.lower()
        if any(word in context_lower for word in legal_context_words):
            confidence += 0.1

        return min(1.0, confidence)

    def _calculate_location_confidence(self, location: str, context: str) -> float:
        """Calculate confidence score for location."""
        confidence = 0.6  # Base confidence

        # Increase for proper capitalization
        if location.istitle():
            confidence += 0.1

        # Increase for multiple words
        if len(location.split()) >= 2:
            confidence += 0.1

        # Check for location indicators
        location_indicators = ['at', 'in', 'near', 'from', 'police station']
        context_lower = context.lower()
        if any(indicator in context_lower for indicator in location_indicators):
            confidence += 0.2

        return min(1.0, confidence)

    def _calculate_date_confidence(self, date_str: str) -> float:
        """Calculate confidence score for date string."""
        confidence = 0.8  # Base confidence for structured dates

        # Increase for standard formats
        if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', date_str):
            confidence += 0.1

        # Increase for month names
        if any(month in date_str for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
            confidence += 0.1

        return min(1.0, confidence)

    def _is_overlapping_with_existing(self, start: int, end: int, entities: List[Entity]) -> bool:
        """Check if new entity overlaps with existing entities."""
        for entity in entities:
            if (start < entity.end and end > entity.start):
                return True
        return False

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate and overlapping entities."""
        if not entities:
            return entities

        # Sort by confidence (highest first)
        entities.sort(key=lambda x: x.confidence, reverse=True)

        deduplicated = []
        for entity in entities:
            # Check if overlaps with any kept entity
            overlaps = False
            for kept in deduplicated:
                if (entity.start < kept.end and entity.end > kept.start):
                    overlaps = True
                    break

            if not overlaps:
                deduplicated.append(entity)

        return deduplicated

    def _calculate_confidence_scores(self, entities: List[Entity], text: str) -> Dict[str, float]:
        """Calculate overall confidence scores by entity type."""
        confidence_scores = {}

        for label in self.legal_entity_types.keys():
            label_entities = [e for e in entities if e.label == label]
            if label_entities:
                # Average confidence for this label
                avg_confidence = sum(e.confidence for e in label_entities) / len(label_entities)
                confidence_scores[label] = avg_confidence
            else:
                confidence_scores[label] = 0.0

        return confidence_scores

    def get_entities_by_type(self, result: NERResult, entity_type: str) -> List[Entity]:
        """Get entities of specific type from result."""
        return [e for e in result.entities if e.label == entity_type]

    def get_entity_texts_by_type(self, result: NERResult, entity_type: str) -> List[str]:
        """Get entity texts of specific type from result."""
        return [e.text for e in self.get_entities_by_type(result, entity_type)]


# Backward compatibility function
def extract_entities_with_ner(text: str) -> Dict[str, List[str]]:
    """
    Backward compatible function that returns entities in old format.
    """
    try:
        ner = EnhancedNER()
        result = ner.extract_entities(text)

        # Convert to old format
        entities = {
            'persons': [],
            'dates': [],
            'times': [],
            'locations': [],
            'orgs': []
        }

        for entity in result.entities:
            if entity.label == 'PERSON':
                entities['persons'].append(entity.text)
            elif entity.label == 'DATE':
                entities['dates'].append(entity.text)
            elif entity.label == 'TIME':
                entities['times'].append(entity.text)
            elif entity.label == 'GPE':
                entities['locations'].append(entity.text)
            elif entity.label == 'ORG':
                entities['orgs'].append(entity.text)

        return entities

    except Exception as e:
        logger.error(f"Error in enhanced NER: {e}")
        # Fallback to simple regex
        return _fallback_ner(text)


def _fallback_ner(text: str) -> Dict[str, List[str]]:
    """Simple fallback NER using regex patterns."""
    # Regex for dates and times, including month names and ordinal suffixes
    date_pattern = r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|June\s+\d{1,2}(?:st|nd|rd|th)?)\b'
    time_pattern = r'\b\d{1,2}:\d{2}\s*(AM|PM|am|pm)?\b'
    dates = re.findall(date_pattern, text)
    times = re.findall(time_pattern, text)

    # Simple heuristic for persons: sequences of capitalized words (2-3 words)
    person_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'
    persons = re.findall(person_pattern, text)

    # Simple heuristic for locations: words after "at" or "in"
    locations = []
    loc_indicators = ['at', 'in', 'near', 'from']
    for indicator in loc_indicators:
        pattern = rf'\b{indicator}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        locations.extend(matches)

    # For orgs, empty for now
    orgs = []

    entities = {
        'persons': persons,
        'dates': dates,
        'times': times,
        'locations': locations,
        'orgs': orgs,
    }
    return entities
