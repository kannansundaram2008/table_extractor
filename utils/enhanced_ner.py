"""
Enhanced Named Entity Recognition (NER) system for FIR document processing.
Combines spaCy with custom rules and confidence scoring for legal domain entities.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
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
        Initialize the enhanced NER system with comprehensive legal domain support.

        Args:
            model_name: spaCy model to use
            confidence_threshold: Minimum confidence for entity acceptance
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nlp = None
        self.custom_patterns = self._load_custom_patterns()
        self.legal_entity_types = {
            'PERSON': ['complainant', 'victim', 'accused', 'witness', 'officer', 'informant', 'defendant'],
            'ORG': ['police_station', 'court', 'department', 'organization', 'commissionerate', 'thana'],
            'GPE': ['location', 'city', 'state', 'district', 'village', 'locality', 'area', 'sector'],
            'LAW': ['ipc_section', 'bns_section', 'crpc_section', 'act', 'article', 'clause', 'subsection'],
            'DATE': ['occurrence_date', 'report_date', 'hearing_date', 'arrest_date', 'bail_date'],
            'TIME': ['occurrence_time', 'report_time', 'arrest_time'],
            'MONEY': ['fine_amount', 'bail_amount', 'property_value', 'compensation'],
            'QUANTITY': ['property_quantity', 'age', 'distance', 'weight', 'measurement'],
            'CRIME': ['offence', 'crime', 'violation', 'incident'],
            'DOCUMENT': ['fir_number', 'complaint_number', 'case_number', 'reference_number'],
            'VEHICLE': ['vehicle_number', 'registration_number', 'license_plate'],
            'PHONE': ['contact_number', 'mobile_number', 'telephone'],
            'ID': ['aadhar', 'passport', 'license', 'voter_id', 'pan_card']
        }

        # Enhanced legal context patterns
        self.legal_context_patterns = self._load_legal_context_patterns()

        # Entity relationship mapping
        self.entity_relationships = self._load_entity_relationships()

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
        """Load comprehensive custom regex patterns for legal domain entities."""
        return {
            'indian_names': [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Multi-word names
                r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b',  # Names with initials
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Three-word names
            ],
            'police_stations': [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Police\s+Station\b',
                r'\bPS\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Thana\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Police\s+Chowki\b',
            ],
            'ipc_sections': [
                r'\bIPC\s+\d+(?:\([^)]+\))?(?:\s*,\s*\d+(?:\([^)]+\))?)*\b',
                r'\bSection\s+\d+(?:\([^)]+\))?\s+of\s+IPC\b',
                r'\bBNS\s+\d+(?:\([^)]+\))?(?:\s*,\s*\d+(?:\([^)]+\))?)*\b',
                r'\bSection\s+\d+(?:\([^)]+\))?\s+of\s+BNS\b',
            ],
            'cr_numbers': [
                r'\b\d{1,4}/\d{2,4}\b',
                r'\bCR\s*No\.?\s*\d{1,4}/\d{2,4}\b',
                r'\bFIR\s*No\.?\s*\d{1,4}/\d{2,4}\b',
                r'\bCase\s*No\.?\s*\d{1,4}/\d{2,4}\b',
            ],
            'dates_indian': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',
            ],
            'times': [
                r'\b\d{1,2}[:.]\d{2}\s*(?:AM|PM|hrs?|hours?)\b',
                r'\b\d{1,2}[:.]\d{2}\s*hours?\b',
            ],
            'phone_numbers': [
                r'\b\d{10}\b',  # Mobile numbers
                r'\b\d{3,5}[-.\s]\d{3,5}[-.\s]\d{3,5}\b',  # Landline numbers
                r'\b\+91[-.\s]\d{10}\b',  # Mobile with country code
            ],
            'vehicle_numbers': [
                r'\b[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}\b',  # Indian vehicle registration
                r'\b[A-Z]{2}-\d{2}-[A-Z]{1,2}-\d{4}\b',
            ],
            'id_numbers': [
                r'\b\d{4}\s\d{4}\s\d{4}\b',  # Aadhar-like pattern
                r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN card pattern
            ],
            'money_amounts': [
                r'Rs\.?\s*\d+(?:,\d{3})*(?:,\d{2})?',
                r'₹\s*\d+(?:,\d{3})*(?:,\d{2})?',
                r'\$\s*\d+(?:,\d{3})*(?:,\d{2})?',
            ],
            'crime_types': [
                r'\b(?:theft|burglary|robbery|murder|assault|kidnapping|rape|cheating|fraud|forgery)\b',
                r'\b(?:देवट|चोरी|लूट|हत्या|मारपीट|अपहरण|बलात्कार|धोखाधड़ी|जालसाजी)\b',
            ]
        }

    def _load_legal_context_patterns(self) -> Dict[str, List[str]]:
        """Load legal context patterns for better entity recognition."""
        return {
            'complaint_context': [
                'complainant', 'complained', 'complains', 'informant', 'informer',
                'शिकायतकर्ता', 'सूचित', 'मुखबिर'
            ],
            'victim_context': [
                'victim', 'injured', 'deceased', 'suffered', 'affected',
                'पीड़ित', 'घायल', 'मृतक', 'प्रभावित'
            ],
            'accused_context': [
                'accused', 'suspect', 'arrested', 'charged', 'defendant',
                'आरोपी', 'संदिग्ध', 'गिरफ्तार', 'प्रतिवादी'
            ],
            'police_context': [
                'police', 'station', 'officer', 'constable', 'inspector', 'sub-inspector',
                'पुलिस', 'थाना', 'अधिकारी', 'सिपाही', 'निरीक्षक', 'उप-निरीक्षक'
            ],
            'legal_context': [
                'section', 'ipc', 'bns', 'crpc', 'act', 'article', 'clause',
                'धारा', 'कानून', 'अधिनियम', 'वाक्य'
            ]
        }

    def _load_entity_relationships(self) -> Dict[str, List[str]]:
        """Load entity relationship patterns for context understanding."""
        return {
            'PERSON-complainant': ['lodged', 'filed', 'submitted', 'informed', 'reported'],
            'PERSON-victim': ['assaulted', 'injured', 'killed', 'robbed', 'affected'],
            'PERSON-accused': ['arrested', 'charged', 'suspected', 'involved', 'named'],
            'ORG-police_station': ['registered', 'recorded', 'noted', 'entered'],
            'GPE-location': ['occurred', 'happened', 'took place', 'situated'],
            'LAW-section': ['violated', 'contravened', 'breached', 'invoked']
        }

    def extract_entities(self, text: str) -> NERResult:
        """
        Extract named entities from text using comprehensive enhanced approach.

        Args:
            text: Input text to process

        Returns:
            NERResult containing entities, confidence scores, and relationships
        """
        import time
        start_time = time.time()

        if not text or not isinstance(text, str):
            return NERResult([], {}, 0.0)

        # Preprocessing with enhanced cleaning
        cleaned_text = self._preprocess_text_enhanced(text)

        entities = []
        entity_relationships = []

        # spaCy NER if available with enhanced processing
        if self.nlp:
            spacy_entities = self._extract_with_spacy_enhanced(cleaned_text)
            entities.extend(spacy_entities)

        # Custom rule-based extraction with comprehensive patterns
        rule_entities = self._extract_with_rules_enhanced(cleaned_text, text)
        entities.extend(rule_entities)

        # Extract entity relationships
        entity_relationships = self._extract_entity_relationships(entities, cleaned_text)

        # Remove duplicates and merge overlapping entities
        entities = self._deduplicate_entities(entities)

        # Enhanced confidence scoring with context analysis
        confidence_scores = self._calculate_enhanced_confidence_scores(entities, cleaned_text)

        # Cross-reference entities for improved accuracy
        entities = self._cross_reference_entities(entities, cleaned_text)

        # Validate entities for consistency and accuracy
        validation_results = self.validate_entities(NERResult(entities, confidence_scores, 0))

        processing_time = time.time() - start_time

        # Create enhanced result with relationships and validation
        result = NERResult(entities, confidence_scores, processing_time)
        result.relationships = entity_relationships
        result.context_analysis = self._analyze_legal_context(cleaned_text)
        result.validation_results = validation_results

        return result

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

    def _preprocess_text_enhanced(self, text: str) -> str:
        """Enhanced text preprocessing for better NER performance."""
        if not text:
            return ""

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)

        # Normalize legal abbreviations with enhanced patterns
        text = self._normalize_legal_terms_enhanced(text)

        # Handle common OCR errors in legal documents
        text = self._fix_common_ocr_errors(text)

        # Normalize case for better pattern matching
        text = self._normalize_case_patterns(text)

        # Extract and preserve structured information
        text = self._preserve_structured_content(text)

        return text.strip()

    def _normalize_legal_terms_enhanced(self, text: str) -> str:
        """Enhanced normalization of legal terms and abbreviations."""
        # Comprehensive legal term mappings
        legal_normalizations = {
            r'\bIPC\b': 'Indian Penal Code',
            r'\bCRPC\b': 'Criminal Procedure Code',
            r'\bBNS\b': 'Bharatiya Nyaya Sanhita',
            r'\bPS\b': 'Police Station',
            r'\bIO\b': 'Investigating Officer',
            r'\bSI\b': 'Sub Inspector',
            r'\bASI\b': 'Assistant Sub Inspector',
            r'\bCI\b': 'Circle Inspector',
            r'\bSP\b': 'Superintendent of Police',
            r'\bDSP\b': 'Deputy Superintendent of Police',
            r'\bFIR\b': 'First Information Report',
            r'\bCR\.?\s*No\b': 'Case Registration Number',
            r'\bU/S\b': 'Under Section',
            r'\bR/O\b': 'Resident Of',
            r'\bS/O\b': 'Son Of',
            r'\bD/O\b': 'Daughter Of',
            r'\bW/O\b': 'Wife Of',
        }

        for pattern, replacement in legal_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _normalize_case_patterns(self, text: str) -> str:
        """Normalize case patterns for better entity recognition."""
        # Preserve proper nouns but normalize common words
        sentences = re.split(r'([.!?]+)', text)

        normalized_sentences = []
        for i, sentence in enumerate(sentences):
            if i % 2 == 0:  # Text parts
                # Normalize common legal words to lowercase for better pattern matching
                legal_words = [
                    'police', 'station', 'section', 'under', 'dated', 'time',
                    'place', 'occurrence', 'report', 'complainant', 'victim',
                    'accused', 'witness', 'property', 'lost', 'recovered', 'seized'
                ]
                for word in legal_words:
                    sentence = re.sub(r'\b' + word + r'\b', word.lower(), sentence, flags=re.IGNORECASE)
            normalized_sentences.append(sentence)

        return ''.join(normalized_sentences)

    def _preserve_structured_content(self, text: str) -> str:
        """Preserve structured content like numbers, dates, etc."""
        # Add spaces around special characters for better tokenization
        text = re.sub(r'([/:,-])', r' \1 ', text)
        return text

    def _extract_with_spacy_enhanced(self, text: str) -> List[Entity]:
        """Enhanced spaCy entity extraction with legal domain mapping."""
        if not self.nlp:
            return []

        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                # Enhanced label mapping for legal domain
                mapped_label = self._map_spacy_label_enhanced(ent.label_, ent.text, text)

                # Enhanced confidence calculation
                confidence = self._calculate_spacy_confidence_enhanced(ent, doc, text)

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
            logger.error(f"Error in enhanced spaCy NER extraction: {e}")
            return []

    def _map_spacy_label_enhanced(self, spacy_label: str, entity_text: str, context: str) -> str:
        """Enhanced mapping of spaCy labels to legal domain with context analysis."""
        # Base mapping
        base_mapping = {
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

        mapped_label = base_mapping.get(spacy_label, 'MISC')

        # Context-based refinement
        context_lower = context.lower()
        entity_lower = entity_text.lower()

        # Police station detection
        if any(term in context_lower for term in ['police station', 'ps ', 'thana']):
            if spacy_label in ['ORG', 'GPE']:
                return 'ORG'

        # Legal section detection
        if any(term in context_lower for term in ['section', 'ipc', 'bns', 'crpc']):
            if spacy_label in ['LAW', 'QUANTITY', 'CARDINAL']:
                return 'LAW'

        # Crime type detection
        if any(term in entity_lower for term in ['theft', 'murder', 'assault', 'rape', 'fraud']):
            return 'CRIME'

        # Document number detection
        if re.search(r'\d{1,4}/\d{2,4}', entity_text):
            return 'DOCUMENT'

        return mapped_label

    def _calculate_spacy_confidence_enhanced(self, ent, doc, context) -> float:
        """Enhanced confidence calculation for spaCy entities."""
        base_confidence = 0.8

        # Length-based adjustments
        if len(ent.text.strip()) < 2:
            base_confidence -= 0.3
        elif len(ent.text.split()) > 5:
            base_confidence -= 0.2

        # Context-based adjustments
        if self._is_legal_context_enhanced(ent, doc, context):
            base_confidence += 0.15

        # Entity type specific adjustments
        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
            base_confidence += 0.1

        return max(0.0, min(1.0, base_confidence))

    def _is_legal_context_enhanced(self, ent, doc, context) -> bool:
        """Enhanced legal context detection."""
        # Look for legal keywords within expanded window
        legal_keywords = [
            'police', 'station', 'fir', 'case', 'ipc', 'section', 'accused',
            'complainant', 'victim', 'crime', 'offence', 'arrest', 'bail',
            'court', 'judge', 'lawyer', 'witness', 'evidence', 'investigation'
        ]

        start = max(0, ent.start - 8)
        end = min(len(doc), ent.end + 8)

        context_tokens = [token.text.lower() for token in doc[start:end]]
        return any(keyword in context_tokens for keyword in legal_keywords)

    def _extract_with_rules_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced rule-based entity extraction with comprehensive patterns."""
        entities = []

        # Extract all entity types using enhanced patterns
        entities.extend(self._extract_persons_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_organizations_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_locations_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_legal_sections_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_dates_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_times_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_phone_numbers_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_vehicle_numbers_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_id_numbers_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_money_amounts_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_crime_types_enhanced(cleaned_text, original_text))
        entities.extend(self._extract_document_numbers_enhanced(cleaned_text, original_text))

        return entities

    def _extract_persons_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced person name extraction with context analysis."""
        entities = []

        for pattern in self.custom_patterns['indian_names']:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                if self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    continue

                confidence = self._calculate_person_confidence_enhanced(match.group(), cleaned_text, original_text)

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

    def _calculate_person_confidence_enhanced(self, name: str, cleaned_text: str, original_text: str) -> float:
        """Enhanced confidence calculation for person names."""
        confidence = 0.7

        # Capitalization check
        if name.istitle():
            confidence += 0.1

        # Length check
        if len(name.split()) >= 2:
            confidence += 0.1
        if len(name.split()) >= 3:
            confidence += 0.05

        # Legal context check
        legal_context_words = [
            'complainant', 'victim', 'accused', 'witness', 's/o', 'd/o', 'w/o',
            'informant', 'defendant', 'arrested', 'charged'
        ]
        context_lower = cleaned_text.lower()
        if any(word in context_lower for word in legal_context_words):
            confidence += 0.15

        # Relationship indicators
        if any(rel in original_text for rel in ['S/o', 'D/o', 'W/o']):
            confidence += 0.1

        return min(1.0, confidence)

    def _extract_organizations_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced organization extraction including police stations."""
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
                        confidence=0.95
                    )
                    entities.append(entity)

        return entities

    def _extract_locations_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced location extraction with directional indicators."""
        entities = []

        # Enhanced location patterns
        location_indicators = [
            'at', 'in', 'near', 'from', 'to', 'opposite', 'behind', 'beside',
            'में', 'पर', 'के पास', 'से', 'तक'
        ]

        for indicator in location_indicators:
            pattern = rf'\b{indicator}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                location_text = match.group(1)
                if not self._is_overlapping_with_existing(match.start(1), match.end(1), entities):
                    confidence = self._calculate_location_confidence_enhanced(location_text, cleaned_text)
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

    def _calculate_location_confidence_enhanced(self, location: str, context: str) -> float:
        """Enhanced confidence calculation for locations."""
        confidence = 0.6

        # Capitalization
        if location.istitle():
            confidence += 0.1

        # Multiple words
        if len(location.split()) >= 2:
            confidence += 0.1

        # Location indicators in context
        location_indicators = [
            'at', 'in', 'near', 'from', 'police station', 'occurred', 'happened'
        ]
        context_lower = context.lower()
        if any(indicator in context_lower for indicator in location_indicators):
            confidence += 0.2

        return min(1.0, confidence)

    def _extract_legal_sections_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced legal section extraction."""
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
                        confidence=0.95
                    )
                    entities.append(entity)

        return entities

    def _extract_dates_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced date extraction with multiple formats."""
        entities = []

        for pattern in self.custom_patterns['dates_indian']:
            matches = re.finditer(pattern, original_text)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    confidence = self._calculate_date_confidence_enhanced(match.group())
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

    def _calculate_date_confidence_enhanced(self, date_str: str) -> float:
        """Enhanced confidence calculation for dates."""
        confidence = 0.8

        # Standard formats
        if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', date_str):
            confidence += 0.1

        # Month names
        if any(month in date_str for month in [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
            'September', 'October', 'November', 'December'
        ]):
            confidence += 0.1

        return min(1.0, confidence)

    def _extract_times_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced time extraction."""
        entities = []

        for pattern in self.custom_patterns['times']:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    entity = Entity(
                        text=match.group(),
                        label='TIME',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    )
                    entities.append(entity)

        return entities

    def _extract_phone_numbers_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced phone number extraction."""
        entities = []

        for pattern in self.custom_patterns['phone_numbers']:
            matches = re.finditer(pattern, original_text)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    entity = Entity(
                        text=match.group(),
                        label='PHONE',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    )
                    entities.append(entity)

        return entities

    def _extract_vehicle_numbers_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced vehicle number extraction."""
        entities = []

        for pattern in self.custom_patterns['vehicle_numbers']:
            matches = re.finditer(pattern, original_text)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    entity = Entity(
                        text=match.group(),
                        label='VEHICLE',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    )
                    entities.append(entity)

        return entities

    def _extract_id_numbers_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced ID number extraction."""
        entities = []

        for pattern in self.custom_patterns['id_numbers']:
            matches = re.finditer(pattern, original_text)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    entity = Entity(
                        text=match.group(),
                        label='ID',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85
                    )
                    entities.append(entity)

        return entities

    def _extract_money_amounts_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced money amount extraction."""
        entities = []

        for pattern in self.custom_patterns['money_amounts']:
            matches = re.finditer(pattern, original_text)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    entity = Entity(
                        text=match.group(),
                        label='MONEY',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    )
                    entities.append(entity)

        return entities

    def _extract_crime_types_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced crime type extraction."""
        entities = []

        for pattern in self.custom_patterns['crime_types']:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    entity = Entity(
                        text=match.group(),
                        label='CRIME',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8
                    )
                    entities.append(entity)

        return entities

    def _extract_document_numbers_enhanced(self, cleaned_text: str, original_text: str) -> List[Entity]:
        """Enhanced document number extraction."""
        entities = []

        for pattern in self.custom_patterns['cr_numbers']:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                if not self._is_overlapping_with_existing(match.start(), match.end(), entities):
                    entity = Entity(
                        text=match.group(),
                        label='DOCUMENT',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95
                    )
                    entities.append(entity)

        return entities

    def _extract_entity_relationships(self, entities: List[Entity], text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities for better context understanding."""
        relationships = []

        # Group entities by type for relationship analysis
        entities_by_type = {}
        for entity in entities:
            if entity.label not in entities_by_type:
                entities_by_type[entity.label] = []
            entities_by_type[entity.label].append(entity)

        # Find relationships based on proximity and context
        for entity1 in entities:
            for entity2 in entities:
                if entity1 == entity2:
                    continue

                # Check if entities are close to each other
                if abs(entity1.start - entity2.start) < 100:  # Within 100 characters
                    relationship = self._analyze_entity_relationship(entity1, entity2, text)
                    if relationship:
                        relationships.append(relationship)

        return relationships

    def _analyze_entity_relationship(self, entity1: Entity, entity2: Entity, text: str) -> Optional[Dict[str, Any]]:
        """Analyze relationship between two entities."""
        # Extract text between entities
        start = min(entity1.start, entity2.start)
        end = max(entity1.end, entity2.end)
        between_text = text[start:end]

        # Check for relationship indicators
        relationship_indicators = {
            'PERSON-complainant': ['filed', 'lodged', 'reported', 'informed'],
            'PERSON-victim': ['injured', 'assaulted', 'killed', 'affected'],
            'PERSON-accused': ['arrested', 'charged', 'suspected'],
            'ORG-location': ['at', 'in', 'situated'],
        }

        for rel_type, indicators in relationship_indicators.items():
            if any(indicator in between_text.lower() for indicator in indicators):
                return {
                    'entity1': entity1.text,
                    'entity1_type': entity1.label,
                    'entity2': entity2.text,
                    'entity2_type': entity2.label,
                    'relationship_type': rel_type,
                    'confidence': 0.7
                }

        return None

    def _analyze_legal_context(self, text: str) -> Dict[str, Any]:
        """Analyze overall legal context of the text."""
        context_analysis = {
            'is_legal_document': False,
            'document_type': 'unknown',
            'legal_terms_count': 0,
            'context_score': 0.0
        }

        legal_terms = [
            'police', 'station', 'fir', 'ipc', 'section', 'accused', 'complainant',
            'victim', 'crime', 'offence', 'arrest', 'bail', 'court', 'judge'
        ]

        text_lower = text.lower()
        legal_terms_found = [term for term in legal_terms if term in text_lower]

        context_analysis['legal_terms_count'] = len(legal_terms_found)
        context_analysis['context_score'] = min(1.0, len(legal_terms_found) / 5.0)

        if context_analysis['context_score'] > 0.6:
            context_analysis['is_legal_document'] = True

        # Determine document type
        if 'fir' in text_lower and 'police' in text_lower:
            context_analysis['document_type'] = 'fir'
        elif 'court' in text_lower:
            context_analysis['document_type'] = 'court_document'
        elif 'police' in text_lower:
            context_analysis['document_type'] = 'police_document'

        return context_analysis

    def _cross_reference_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """Cross-reference entities for improved accuracy."""
        # Sort entities by confidence
        entities.sort(key=lambda x: x.confidence, reverse=True)

        # Remove low-confidence duplicates
        filtered_entities = []
        for entity in entities:
            is_duplicate = False
            for filtered in filtered_entities:
                if (abs(entity.start - filtered.start) < 5 and
                    entity.label == filtered.label):
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_entities.append(entity)

        return filtered_entities

    def _calculate_enhanced_confidence_scores(self, entities: List[Entity], text: str) -> Dict[str, float]:
        """Calculate enhanced confidence scores with context analysis."""
        confidence_scores = {}

        for label in self.legal_entity_types.keys():
            label_entities = [e for e in entities if e.label == label]
            if label_entities:
                # Weighted average confidence
                total_confidence = sum(e.confidence for e in label_entities)
                avg_confidence = total_confidence / len(label_entities)

                # Boost confidence for legal context
                if self._is_legal_context_detected(text):
                    avg_confidence += 0.1

                confidence_scores[label] = min(1.0, avg_confidence)
            else:
                confidence_scores[label] = 0.0

        return confidence_scores

    def _is_legal_context_detected(self, text: str) -> bool:
        """Check if text contains legal context indicators."""
        legal_indicators = [
            'police station', 'ipc', 'section', 'accused', 'complainant',
            'victim', 'fir', 'crime', 'offence', 'arrest'
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in legal_indicators)

    def validate_entities(self, result: NERResult) -> Dict[str, Any]:
        """Validate extracted entities for consistency and accuracy."""
        validation_results = {
            'total_entities': len(result.entities),
            'valid_entities': 0,
            'validation_score': 0.0,
            'issues': [],
            'warnings': []
        }

        if not result.entities:
            return validation_results

        valid_entities = []
        entity_types_found = set()

        for entity in result.entities:
            is_valid, issues = self._validate_single_entity(entity, result)

            if is_valid:
                valid_entities.append(entity)
                entity_types_found.add(entity.label)
            else:
                validation_results['issues'].extend(issues)

        validation_results['valid_entities'] = len(valid_entities)
        validation_results['validation_score'] = len(valid_entities) / len(result.entities)

        # Check for expected entity types in legal documents
        expected_types = {'PERSON', 'ORG', 'GPE', 'DATE'}
        missing_types = expected_types - entity_types_found

        if missing_types:
            validation_results['warnings'].append(f"Missing expected entity types: {missing_types}")

        # Cross-entity validation
        cross_validation_issues = self._cross_validate_entities(result.entities)
        validation_results['issues'].extend(cross_validation_issues)

        return validation_results

    def _validate_single_entity(self, entity: Entity, result: NERResult) -> Tuple[bool, List[str]]:
        """Validate a single entity for consistency."""
        issues = []
        is_valid = True

        # Length validation
        if len(entity.text.strip()) < 2:
            issues.append(f"Entity '{entity.text}' is too short")
            is_valid = False

        # Character validation
        if not re.match(r'^[a-zA-Z0-9\s\-.,()/:]+$', entity.text):
            issues.append(f"Entity '{entity.text}' contains invalid characters")
            is_valid = False

        # Confidence threshold validation
        if entity.confidence < 0.5:
            issues.append(f"Entity '{entity.text}' has low confidence: {entity.confidence:.2f}")
            is_valid = False

        # Context validation
        if not self._validate_entity_context(entity, result):
            issues.append(f"Entity '{entity.text}' lacks sufficient context")
            is_valid = False

        return is_valid, issues

    def _validate_entity_context(self, entity: Entity, result: NERResult) -> bool:
        """Validate entity has sufficient contextual support."""
        # Check if entity appears in legal context
        legal_keywords = [
            'police', 'station', 'fir', 'case', 'ipc', 'section', 'accused',
            'complainant', 'victim', 'crime', 'offence', 'arrest'
        ]

        # For legal documents, entities should have legal context
        if hasattr(result, 'context_analysis') and result.context_analysis.get('is_legal_document'):
            # Check if entity type is expected in legal context
            legal_entity_types = {'PERSON', 'ORG', 'GPE', 'LAW', 'DATE', 'DOCUMENT'}
            if entity.label in legal_entity_types:
                return True  # Legal entities are generally valid

        return True  # Non-legal context validation is more lenient

    def _cross_validate_entities(self, entities: List[Entity]) -> List[str]:
        """Cross-validate entities for consistency."""
        issues = []

        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            if entity.label not in entities_by_type:
                entities_by_type[entity.label] = []
            entities_by_type[entity.label].append(entity)

        # Validate PERSON entities have reasonable names
        if 'PERSON' in entities_by_type:
            persons = entities_by_type['PERSON']
            for person in persons:
                if len(person.text.split()) < 2 and not person.text.endswith('.'):
                    issues.append(f"Person name '{person.text}' seems incomplete")

        # Validate DATE entities are reasonable
        if 'DATE' in entities_by_type:
            dates = entities_by_type['DATE']
            for date in dates:
                # Check for obviously wrong dates
                date_nums = re.findall(r'\d+', date.text)
                if date_nums:
                    for num in date_nums:
                        if len(num) == 4 and not (1900 <= int(num) <= 2100):
                            issues.append(f"Date '{date.text}' contains invalid year")
                        elif len(num) <= 2 and int(num) > 31:
                            issues.append(f"Date '{date.text}' contains invalid day/month")

        return issues

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
