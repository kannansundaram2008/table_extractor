import re
import logging
import time
from dateutil import parser as date_parser
from datetime import datetime
from typing import Dict, List, Optional, Any

# Enhanced NER using spaCy and custom rules
from utils.enhanced_ner import extract_entities_with_ner, EnhancedNER

# ML Pattern Learner for enhanced entity extraction
from utils.ml_pattern_learner import MLPatternLearner

from utils.pattern_matcher import has_primary_pattern, has_adjacent_pattern

# Training data collection and logging
from utils.training_logger import get_training_logger

# Configure logging
logger = logging.getLogger(__name__)

def parse_fir_row(row_cells):
    """
    Parse the FIR row cells into structured fields based on column positions.
    Assumes specific columns contain specific data.
    """
    parsed = {
        'police_station': '',
        'cr_no': '',
        'section_of_law': '',
        'date_of_occurrence': '',
        'date_of_report': '',
        'do_time': '',
        'dr_time': '',
        'place_of_occurrence': '',
        'complainant': {'name': '', 'address': '', 'sex': '', 'age': ''},
        'victims_count': 0,
        'victims': [],
        'accused': [],
        'property_lost': [],
        'property_recovered': [],
        'property_seized': [],
        'gist': ''
    }

    # Determine start index based on primary and adjacent pattern match
    start_index = 0  # default start index
    primary_indices = [i for i, cell in enumerate(row_cells) if has_primary_pattern(cell)]
    if primary_indices:
        # Prefer the one with adjacent pattern
        for i in primary_indices:
            if i + 1 < len(row_cells) and has_adjacent_pattern(row_cells[i + 1]):
                start_index = i
                break
        else:
            # If no adjacent pattern matches, use the first primary index
            start_index = primary_indices[0]

    # Validate input
    if not row_cells or not isinstance(row_cells, list):
        return parsed

    if len(row_cells) > start_index:  # Police Station, CR No, Section of Law
        parse_column_5(row_cells, start_index, parsed)
    if len(row_cells) > start_index + 1:  # Dates, Times, Place of Occurrence
        parse_column_6(row_cells[start_index + 1], parsed)
    if len(row_cells) > start_index + 2:  # Complainant Details
        parse_column_7(row_cells[start_index + 2], parsed)
    if len(row_cells) > start_index + 3:  # Victim Details
        parse_column_8(row_cells[start_index + 3], parsed)
    if len(row_cells) > start_index + 4:  # Property Details
        parse_column_9(row_cells[start_index + 4], parsed)
    if len(row_cells) > start_index + 5:  # Accused Details
        parse_column_10(row_cells[start_index + 5], parsed)
    if len(row_cells) > start_index + 6:  # Gist / Remarks
        parsed['gist'] = row_cells[start_index + 6].strip()
    return parsed


class EnhancedFIRParser:
    """
    Enhanced FIR parser with ML-NER integration, confidence scoring, and performance monitoring.
    """

    def __init__(self, confidence_threshold: float = 0.7, enable_ml: bool = True, enable_logging: bool = True):
        """
        Initialize the enhanced FIR parser.

        Args:
            confidence_threshold: Minimum confidence for entity acceptance
            enable_ml: Whether to use ML pattern learner for enhanced extraction
            enable_logging: Whether to enable training data logging
        """
        self.confidence_threshold = confidence_threshold
        self.enable_ml = enable_ml
        self.enable_logging = enable_logging

        # Initialize enhanced NER system
        self.ner_system = EnhancedNER(confidence_threshold=confidence_threshold)

        # Initialize ML pattern learner if enabled
        self.ml_learner = None
        if enable_ml:
            try:
                self.ml_learner = MLPatternLearner()
                logger.info("ML pattern learner initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize ML pattern learner: {e}")
                self.enable_ml = False

        # Initialize training logger if enabled
        self.training_logger = None
        if enable_logging:
            try:
                self.training_logger = get_training_logger()
                logger.info("Training logger initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize training logger: {e}")
                self.enable_logging = False

        # Performance tracking
        self.performance_stats = {
            'total_parsing_time': 0.0,
            'total_ner_time': 0.0,
            'total_ml_time': 0.0,
            'parse_count': 0,
            'ner_call_count': 0,
            'ml_call_count': 0
        }

    def parse_fir_row_enhanced(self, row_cells: List[str], source_file: str = "unknown") -> Dict[str, Any]:
        """
        Enhanced FIR parsing with ML-NER integration and confidence scoring.

        Args:
            row_cells: List of cell texts from FIR table row
            source_file: Path to the source document for logging

        Returns:
            Dictionary with parsed data and confidence scores
        """
        start_time = time.time()

        # Initialize result structure
        parsed = {
            'police_station': '',
            'cr_no': '',
            'section_of_law': '',
            'date_of_occurrence': '',
            'date_of_report': '',
            'do_time': '',
            'dr_time': '',
            'place_of_occurrence': '',
            'complainant': {'name': '', 'address': '', 'sex': '', 'age': ''},
            'victims_count': 0,
            'victims': [],
            'accused': [],
            'property_lost': [],
            'property_recovered': [],
            'property_seized': [],
            'gist': '',
            'confidence_scores': {},
            'processing_metadata': {
                'ml_enabled': self.enable_ml,
                'ner_processing_time': 0.0,
                'ml_processing_time': 0.0,
                'fallback_used': False
            }
        }

        # Combine row cells into input text for logging
        input_text = " | ".join(str(cell) for cell in row_cells if cell)

        try:
            # Determine start index based on primary and adjacent pattern match
            start_index = self._find_start_index(row_cells)

            # Validate input
            if not row_cells or not isinstance(row_cells, list):
                if self.training_logger and self.enable_logging:
                    self.training_logger.log_fir_row_extraction(
                        source_file=source_file,
                        row_text=input_text,
                        extracted_data=parsed,
                        confidence_scores=parsed['confidence_scores'],
                        processing_time=time.time() - start_time,
                        success=False,
                        error_message="Invalid input: empty or non-list row_cells"
                    )
                return parsed

            # Parse each column with enhanced NER
            if len(row_cells) > start_index:
                self._parse_column_5_enhanced(row_cells, start_index, parsed)
            if len(row_cells) > start_index + 1:
                self._parse_column_6_enhanced(row_cells[start_index + 1], parsed)
            if len(row_cells) > start_index + 2:
                self._parse_column_7_enhanced(row_cells[start_index + 2], parsed)
            if len(row_cells) > start_index + 3:
                self._parse_column_8_enhanced(row_cells[start_index + 3], parsed)
            if len(row_cells) > start_index + 4:
                self._parse_column_9_enhanced(row_cells[start_index + 4], parsed)
            if len(row_cells) > start_index + 5:
                self._parse_column_10_enhanced(row_cells[start_index + 5], parsed)
            if len(row_cells) > start_index + 6:
                parsed['gist'] = row_cells[start_index + 6].strip()

        except Exception as e:
            logger.error(f"Error in enhanced FIR parsing: {e}")
            parsed['processing_metadata']['fallback_used'] = True

            # Log failed extraction
            if self.training_logger and self.enable_logging:
                self.training_logger.log_fir_row_extraction(
                    source_file=source_file,
                    row_text=input_text,
                    extracted_data=parsed,
                    confidence_scores=parsed['confidence_scores'],
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=str(e)
                )

        # Update performance statistics
        parsing_time = time.time() - start_time
        self.performance_stats['total_parsing_time'] += parsing_time
        self.performance_stats['parse_count'] += 1

        # Log successful extraction
        if self.training_logger and self.enable_logging:
            try:
                self.training_logger.log_fir_row_extraction(
                    source_file=source_file,
                    row_text=input_text,
                    extracted_data=parsed,
                    confidence_scores=parsed['confidence_scores'],
                    processing_time=parsing_time,
                    success=True,
                    processing_metadata={
                        'ml_enabled': self.enable_ml,
                        'ner_processing_time': parsed['processing_metadata']['ner_processing_time'],
                        'ml_processing_time': parsed['processing_metadata']['ml_processing_time'],
                        'fallback_used': parsed['processing_metadata']['fallback_used']
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log extraction result: {e}")

        return parsed

    def _find_start_index(self, row_cells: List[str]) -> int:
        """Find the optimal starting index for parsing."""
        start_index = 0
        primary_indices = [i for i, cell in enumerate(row_cells) if has_primary_pattern(cell)]

        if primary_indices:
            # Prefer the one with adjacent pattern
            for i in primary_indices:
                if i + 1 < len(row_cells) and has_adjacent_pattern(row_cells[i + 1]):
                    start_index = i
                    break
            else:
                # If no adjacent pattern matches, use the first primary index
                start_index = primary_indices[0]

        return start_index

    def _parse_column_5_enhanced(self, row_cells: List[str], index: int, parsed: Dict[str, Any]):
        """Enhanced parsing of 5th column with NER integration."""
        cell_text = row_cells[index]

        # Find CR No using pattern {1,4}/{2,4}
        cr_match = re.search(r'\b(\d{1,4}/\d{2,4})\b', cell_text)

        if not cr_match:
            # Fallback CR No Check: If CR No pattern is not found in this cell, check previous and next adjacent cells.
            if index > 0 and index-1 < len(row_cells):
                cr_match = re.search(r'\b(\d{1,4}/\d{2,4})\b', row_cells[index-1])
                if cr_match:
                    cell_text = row_cells[index-1]
            if not cr_match and index < len(row_cells)-1 and index+1 < len(row_cells):
                cr_match = re.search(r'\b(\d{1,4}/\d{2,4})\b', row_cells[index+1])
                if cr_match:
                    cell_text = row_cells[index+1]

        if cr_match:
            parsed['cr_no'] = cr_match.group(1)
            parsed['confidence_scores']['cr_no'] = 0.95  # High confidence for regex match

            # Police Station: Extract text preceding CR No with NER enhancement
            before_cr = cell_text[:cr_match.start()].strip()
            if before_cr:
                police_station = self._extract_police_station_with_ner(before_cr)
                if police_station:
                    parsed['police_station'] = police_station
                    parsed['confidence_scores']['police_station'] = 0.8

            # Section of Law: Extract text after CR No with NER enhancement
            after_cr = cell_text[cr_match.end():].strip()
            if after_cr:
                section_of_law = self._extract_law_section_with_ner(after_cr)
                if section_of_law:
                    parsed['section_of_law'] = section_of_law
                    parsed['confidence_scores']['section_of_law'] = 0.85

    def _parse_column_6_enhanced(self, cell_text: str, parsed: Dict[str, Any]):
        """Enhanced parsing of 6th column with NER integration."""
        ner_start = time.time()

        # Extract entities using enhanced NER
        entities_result = self.ner_system.extract_entities(cell_text)

        ner_time = time.time() - ner_start
        parsed['processing_metadata']['ner_processing_time'] = ner_time
        self.performance_stats['total_ner_time'] += ner_time
        self.performance_stats['ner_call_count'] += 1

        # Extract dates with confidence scoring
        dates = self._extract_dates_with_confidence(cell_text, entities_result)
        if len(dates) >= 2:
            parsed['date_of_occurrence'] = dates[0]['text']
            parsed['date_of_report'] = dates[1]['text']
            parsed['confidence_scores']['date_of_occurrence'] = dates[0]['confidence']
            parsed['confidence_scores']['date_of_report'] = dates[1]['confidence']
        elif len(dates) == 1:
            parsed['date_of_occurrence'] = dates[0]['text']
            parsed['confidence_scores']['date_of_occurrence'] = dates[0]['confidence']

        # Extract times
        time_matches = re.findall(r'\b(\d{1,2}[:.-]\d{2})hrs?\b', cell_text, re.IGNORECASE)
        if time_matches:
            parsed['do_time'] = time_matches[0].strip() + 'hrs'
            parsed['confidence_scores']['do_time'] = 0.9
            if len(time_matches) > 1:
                parsed['dr_time'] = time_matches[1].strip() + 'hrs'
                parsed['confidence_scores']['dr_time'] = 0.9

        # Extract location with NER enhancement
        location = self._extract_location_with_ner(cell_text, entities_result)
        if location:
            parsed['place_of_occurrence'] = location['text']
            parsed['confidence_scores']['place_of_occurrence'] = location['confidence']

    def _parse_column_7_enhanced(self, cell_text: str, parsed: Dict[str, Any]):
        """Enhanced parsing of 7th column (complainant) with NER integration."""
        # Use enhanced NER for person and location extraction
        entities_result = self.ner_system.extract_entities(cell_text)

        # Extract person entities
        persons = [e for e in entities_result.entities if e.label == 'PERSON']
        if persons:
            # Use highest confidence person as complainant name
            persons.sort(key=lambda x: x.confidence, reverse=True)
            parsed['complainant']['name'] = persons[0].text
            parsed['confidence_scores']['complainant_name'] = persons[0].confidence

        # Extract location entities for address
        locations = [e for e in entities_result.entities if e.label == 'GPE']
        if locations:
            parsed['complainant']['address'] = locations[0].text
            parsed['confidence_scores']['complainant_address'] = locations[0].confidence

        # Extract age if present
        age_match = re.search(r'\((\d{1,2})\)|(\d{1,2})(?:\s*/\s*(\d{1,4}))?', cell_text)
        if age_match:
            parsed['complainant']['age'] = age_match.group(1) or age_match.group(2)
            parsed['confidence_scores']['complainant_age'] = 0.9

        # Determine sex based on relationship indicators
        if 'S/o' in cell_text:
            parsed['complainant']['sex'] = 'Male'
            parsed['confidence_scores']['complainant_sex'] = 0.8
        elif 'D/o' in cell_text or 'W/o' in cell_text:
            parsed['complainant']['sex'] = 'Female'
            parsed['confidence_scores']['complainant_sex'] = 0.8

    def _parse_column_8_enhanced(self, cell_text: str, parsed: Dict[str, Any]):
        """Enhanced parsing of 8th column (victims) with NER integration."""
        entities_result = self.ner_system.extract_entities(cell_text)

        # Extract person entities for victims
        persons = [e for e in entities_result.entities if e.label == 'PERSON']

        # Count victims by numbers 1. xxxx, 2. yyyy or by S/o, D/o count
        victim_blocks = re.split(r'\b(\d+)\.\s*', cell_text)
        victims = []

        if len(victim_blocks) > 1:
            for i in range(1, len(victim_blocks), 2):
                block = victim_blocks[i+1].strip()
                victim = self._parse_victim_with_ner(block, persons)
                if victim['name']:
                    victims.append(victim)
        else:
            # Count by S/o, D/o, W/o
            so_do_count = len(re.findall(r'\b[Ss]/o\b|\b[Dd]/o\b|\b[Ww]/o\b', cell_text))
            if so_do_count > 0:
                lines = cell_text.split('\n')
                for line in lines:
                    if 'S/o' in line or 'D/o' in line or 'W/o' in line:
                        victim = self._parse_victim_with_ner(line, persons)
                        if victim['name']:
                            victims.append(victim)

        parsed['victims'] = victims
        parsed['victims_count'] = len(victims)
        parsed['confidence_scores']['victims_count'] = 0.8 if victims else 0.5

    def _parse_column_9_enhanced(self, cell_text: str, parsed: Dict[str, Any]):
        """Enhanced parsing of 9th column (property) with NER integration."""
        # Use ML pattern learner if available for better property extraction
        if self.enable_ml and self.ml_learner:
            ml_start = time.time()
            try:
                ml_entities = self.ml_learner.predict_entities(cell_text)
                ml_time = time.time() - ml_start
                parsed['processing_metadata']['ml_processing_time'] = ml_time
                self.performance_stats['total_ml_time'] += ml_time
                self.performance_stats['ml_call_count'] += 1

                # Extract property-related entities
                property_items = []
                for entity_type, entities in ml_entities.items():
                    if entity_type in ['MONEY', 'QUANTITY']:
                        property_items.extend(entities)

                if property_items:
                    # Parse property with enhanced confidence
                    for item in property_items:
                        prop = self._parse_property_with_confidence(item)
                        if prop:
                            if 'lost' in cell_text.lower():
                                parsed['property_lost'].append(prop)
                            elif 'recovered' in cell_text.lower():
                                parsed['property_recovered'].append(prop)
                            elif 'seized' in cell_text.lower():
                                parsed['property_seized'].append(prop)

            except Exception as e:
                logger.warning(f"ML property extraction failed: {e}")

        # Fallback to regex-based property extraction
        if not any([parsed['property_lost'], parsed['property_recovered'], parsed['property_seized']]):
            lines = [line.strip() for line in cell_text.replace('\n', ',').split(',') if line.strip()]
            for line in lines:
                line_lower = line.lower()
                if 'property lost' in line_lower:
                    prop = parse_property(line)
                    if prop:
                        parsed['property_lost'].append(prop)
                elif 'property recovered' in line_lower:
                    prop = parse_property(line)
                    if prop:
                        parsed['property_recovered'].append(prop)
                elif 'property seized' in line_lower:
                    prop = parse_property(line)
                    if prop:
                        parsed['property_seized'].append(prop)

    def _parse_column_10_enhanced(self, cell_text: str, parsed: Dict[str, Any]):
        """Enhanced parsing of 10th column (accused) with NER integration."""
        entities_result = self.ner_system.extract_entities(cell_text)

        # Extract person entities for accused
        persons = [e for e in entities_result.entities if e.label == 'PERSON']

        # Similar to victims parsing but for accused
        accused_blocks = re.split(r'\b(\d+)\.\s*', cell_text)
        accused = []

        if len(accused_blocks) > 1:
            for i in range(1, len(accused_blocks), 2):
                block = accused_blocks[i+1].strip()
                acc = self._parse_accused_with_ner(block, persons)
                if acc['name']:
                    accused.append(acc)
        else:
            # Count by S/o, D/o, W/o
            so_do_count = len(re.findall(r'\b[Ss]/o\b|\b[Dd]/o\b|\b[Ww]/o\b', cell_text))
            if so_do_count > 0:
                lines = cell_text.split('\n')
                for line in lines:
                    if 'S/o' in line or 'D/o' in line or 'W/o' in line:
                        acc = self._parse_accused_with_ner(line, persons)
                        if acc['name']:
                            accused.append(acc)

        parsed['accused'] = accused
        parsed['confidence_scores']['accused_count'] = 0.8 if accused else 0.5

    def _extract_police_station_with_ner(self, text: str) -> Optional[str]:
        """Extract police station using NER with fallback."""
        try:
            entities_result = self.ner_system.extract_entities(text)
            orgs = [e for e in entities_result.entities if e.label == 'ORG']

            if orgs:
                # Use highest confidence organization
                orgs.sort(key=lambda x: x.confidence, reverse=True)
                return orgs[0].text

            # Fallback to regex patterns
            ps_patterns = [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Police\s+Station\b',
                r'\bPS\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            ]

            for pattern in ps_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group()

        except Exception as e:
            logger.warning(f"Police station extraction failed: {e}")

        return None

    def _extract_law_section_with_ner(self, text: str) -> Optional[str]:
        """Extract law section using NER with fallback."""
        try:
            entities_result = self.ner_system.extract_entities(text)
            laws = [e for e in entities_result.entities if e.label == 'LAW']

            if laws:
                # Use highest confidence law entity
                laws.sort(key=lambda x: x.confidence, reverse=True)
                return laws[0].text

            # Fallback to regex patterns
            law_patterns = [
                r'\bIPC\s+\d+(?:\([^)]+\))?(?:\s*,\s*\d+(?:\([^)]+\))?)*\b',
                r'\bSection\s+\d+(?:\([^)]+\))?\s+of\s+IPC\b',
            ]

            for pattern in law_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group()

        except Exception as e:
            logger.warning(f"Law section extraction failed: {e}")

        return None

    def _extract_dates_with_confidence(self, text: str, entities_result) -> List[Dict[str, Any]]:
        """Extract dates with confidence scores."""
        dates = []

        # Use NER dates first
        ner_dates = [e for e in entities_result.entities if e.label == 'DATE']
        for date_entity in ner_dates:
            if date_entity.confidence >= self.confidence_threshold:
                dates.append({
                    'text': date_entity.text,
                    'confidence': date_entity.confidence
                })

        # Fallback to regex if no NER dates found
        if not dates:
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
            ]

            for pattern in date_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    dates.append({
                        'text': match,
                        'confidence': 0.7  # Lower confidence for regex matches
                    })

        return dates

    def _extract_location_with_ner(self, text: str, entities_result) -> Optional[Dict[str, Any]]:
        """Extract location with confidence score."""
        # Use NER locations first
        locations = [e for e in entities_result.entities if e.label == 'GPE']
        if locations:
            # Use highest confidence location
            locations.sort(key=lambda x: x.confidence, reverse=True)
            return {
                'text': locations[0].text,
                'confidence': locations[0].confidence
            }

        # Fallback to regex patterns
        directions = ['east', 'west', 'north', 'south', 'northeast', 'northwest', 'southeast', 'southwest']
        time_end = 0

        for match in re.finditer(r'\b(\d{1,2}[:.-]\d{2})hrs?\b', text, re.IGNORECASE):
            time_end = match.end()

        if time_end:
            after_time = text[time_end:].strip()
            for dir in directions:
                dir_pos = after_time.lower().find(dir)
                if dir_pos != -1:
                    location_text = after_time[:dir_pos + len(dir)].strip()
                    return {
                        'text': location_text,
                        'confidence': 0.6
                    }

        return None

    def _parse_victim_with_ner(self, text: str, persons: List) -> Dict[str, str]:
        """Parse victim details with NER enhancement."""
        victim = {'name': '', 'age': '', 'address': '', 'sex': ''}

        # Find matching person entity
        victim_persons = []
        for person in persons:
            if person.text.lower() in text.lower():
                victim_persons.append(person)

        if victim_persons:
            # Use highest confidence matching person
            victim_persons.sort(key=lambda x: x.confidence, reverse=True)
            victim['name'] = victim_persons[0].text

        # Extract age
        age_match = re.search(r'\((\d{1,2})\)|(\d{1,2})(?:\s*/\s*(\d{1,4}))?', text)
        if age_match:
            victim['age'] = age_match.group(1) or age_match.group(2)

        # Determine sex
        if 'S/o' in text:
            victim['sex'] = 'Male'
        elif 'D/o' in text or 'W/o' in text:
            victim['sex'] = 'Female'

        return victim

    def _parse_accused_with_ner(self, text: str, persons: List) -> Dict[str, str]:
        """Parse accused details with NER enhancement."""
        accused = {'name': '', 'age': '', 'address': '', 'sex': ''}

        # Find matching person entity
        accused_persons = []
        for person in persons:
            if person.text.lower() in text.lower():
                accused_persons.append(person)

        if accused_persons:
            # Use highest confidence matching person
            accused_persons.sort(key=lambda x: x.confidence, reverse=True)
            accused['name'] = accused_persons[0].text

        # Extract age
        age_match = re.search(r'\((\d{1,2})\)|(\d{1,2})(?:\s*/\s*(\d{1,4}))?', text)
        if age_match:
            accused['age'] = age_match.group(1) or age_match.group(2)

        # Determine sex
        if 'S/o' in text:
            accused['sex'] = 'Male'
        elif 'D/o' in text or 'W/o' in text:
            accused['sex'] = 'Female'

        return accused

    def _parse_property_with_confidence(self, text: str) -> Optional[Dict[str, str]]:
        """Parse property with confidence scoring."""
        prop = {'item': '', 'quantity': '', 'confidence': 0.0}

        # Enhanced property parsing with units
        units = r'(?:no\'?s?|packs?|packets?|Rs\.?|grams?|kg|bundles?|pieces?|items?)'
        pattern = rf'(.+?)\s+(\d+(?:\.\d+)?)\s*({units})'

        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prop['item'] = match.group(1).strip()
            prop['quantity'] = match.group(2) + ' ' + match.group(3)
            prop['confidence'] = 0.8
        else:
            # Fallback parsing
            item_match = re.search(r'item[:\-]\s*([^,]+)', text, re.IGNORECASE)
            if item_match:
                prop['item'] = item_match.group(1).strip()
                prop['confidence'] = 0.6

            quantity_match = re.search(r'quantity[:\-]\s*(\d+)', text, re.IGNORECASE)
            if quantity_match:
                prop['quantity'] = quantity_match.group(1).strip()
                prop['confidence'] = 0.6

        return prop if prop['item'] and prop['confidence'] > 0.5 else None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        stats = self.performance_stats.copy()

        if stats['parse_count'] > 0:
            stats['avg_parsing_time'] = stats['total_parsing_time'] / stats['parse_count']
            stats['avg_ner_time'] = stats['total_ner_time'] / stats['ner_call_count'] if stats['ner_call_count'] > 0 else 0
            stats['avg_ml_time'] = stats['total_ml_time'] / stats['ml_call_count'] if stats['ml_call_count'] > 0 else 0

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_parsing_time': 0.0,
            'total_ner_time': 0.0,
            'total_ml_time': 0.0,
            'parse_count': 0,
            'ner_call_count': 0,
            'ml_call_count': 0
        }


# Global instance for backward compatibility
_enhanced_parser = None

def get_enhanced_parser(confidence_threshold: float = 0.7, enable_ml: bool = True, enable_logging: bool = True) -> EnhancedFIRParser:
    """Get or create global enhanced parser instance."""
    global _enhanced_parser
    if _enhanced_parser is None:
        _enhanced_parser = EnhancedFIRParser(confidence_threshold, enable_ml, enable_logging)
    return _enhanced_parser


def parse_fir_row_enhanced(row_cells: List[str], source_file: str = "unknown") -> Dict[str, Any]:
    """
    Enhanced FIR parsing function with ML-NER integration.

    Args:
        row_cells: List of cell texts from FIR table row
        source_file: Path to the source document for logging

    Returns:
        Dictionary with parsed data and confidence scores
    """
    parser = get_enhanced_parser()
    return parser.parse_fir_row_enhanced(row_cells, source_file)

def parse_column_5(row_cells, index, parsed):
    """Parse 5th column: Police Station, CR No, Section of Law"""
    cell_text = row_cells[index]
    # Find CR No using pattern {1,4}/{2,4}
    cr_match = re.search(r'\b(\d{1,4}/\d{2,4})\b', cell_text)
    if not cr_match:
        # Fallback CR No Check: If CR No pattern is not found in this cell, check previous and next adjacent cells.
        if index > 0 and index-1 < len(row_cells):
            cr_match = re.search(r'\b(\d{1,4}/\d{2,4})\b', row_cells[index-1])
            if cr_match:
                cell_text = row_cells[index-1]
        if not cr_match and index < len(row_cells)-1 and index+1 < len(row_cells):
            cr_match = re.search(r'\b(\d{1,4}/\d{2,4})\b', row_cells[index+1])
            if cr_match:
                cell_text = row_cells[index+1]
    if cr_match:
        parsed['cr_no'] = cr_match.group(1)
        # Police Station: Extract text preceding CR No.
        before_cr = cell_text[:cr_match.start()].strip()
        if before_cr:
            # Check if labeled
            if 'police station' in before_cr.lower():
                # Use fallback method on before_cr
                lines = [line.strip() for line in before_cr.replace('\n', ',').split(',') if line.strip()]
                for line in lines:
                    line_lower = line.lower()
                    if 'police station' in line_lower:
                        parsed['police_station'] = extract_value(line)
                        break
            else:
                # If this text contains a break (e.g., newline, delimiter), check both fragments:
                if '\n' in before_cr or ',' in before_cr or ';' in before_cr:
                    parts = re.split(r'[\n,;]', before_cr)
                    parts = [p.strip() for p in parts if p.strip()]
                    # If both fragments have >3 characters, select the second fragment as station name.
                    if len(parts) > 1 and all(len(p) > 3 for p in parts):
                        parsed['police_station'] = parts[1]
                    else:
                        parsed['police_station'] = before_cr
                else:
                    parsed['police_station'] = before_cr
                # Clean police station: remove C.no, Cr.No, CR No, " PS", and CR No pattern case insensitive
                parsed['police_station'] = re.sub(r'\b[Cc][Rr]?\.?\s*[Nn][Oo]\b', '', parsed['police_station'], flags=re.IGNORECASE)
                parsed['police_station'] = re.sub(r'\s+PS\b', '', parsed['police_station'], flags=re.IGNORECASE)
                parsed['police_station'] = re.sub(r'\b\d{1,4}/\d{2,4}\b', '', parsed['police_station'], flags=re.IGNORECASE)
                parsed['police_station'] = parsed['police_station'].strip()
        # Section of Law: All remaining text after CR No is treated as section(s) of law.
        after_cr = cell_text[cr_match.end():].strip()
        if after_cr:
            after_cr = re.sub(r'^[,;]\s*', '', after_cr)
            if 'section of law' in after_cr.lower() or 'section' in after_cr.lower():
                parsed['section_of_law'] = extract_value(after_cr)
            else:
                parsed['section_of_law'] = after_cr
    else:
        # Fallback to old method with bounds checking
        lines = [line.strip() for line in cell_text.replace('\n', ',').split(',') if line.strip()]
        for line in lines:
            line_lower = line.lower()
            if 'police station' in line_lower:
                parsed['police_station'] = extract_value(line)
            elif 'cr no' in line_lower or 'cr. no' in line_lower:
                parsed['cr_no'] = extract_value(line)
            elif 'section of law' in line_lower or 'section' in line_lower:
                parsed['section_of_law'] = extract_value(line)

def parse_column_6(cell_text, parsed):
    """Parse 6th column: Dates, Times, Place of Occurrence"""
    # Dates: Use datetime.strptime() for standard formats.
    dates = extract_and_parse_dates(cell_text)
    if len(dates) >= 2:
        parsed['date_of_occurrence'] = dates[0]
        parsed['date_of_report'] = dates[1]
    elif len(dates) == 3:
        parsed['date_of_occurrence'] = f"{dates[0]} to {dates[1]}"  # first two = occurrence
        parsed['date_of_report'] = dates[2]
    elif len(dates) == 1:
        parsed['date_of_occurrence'] = dates[0]

    # Fallback to NER if dates are missing or ambiguous
    if not parsed['date_of_occurrence'] or not parsed['date_of_report']:
        entities = extract_entities_with_ner(cell_text)
        ner_dates = entities.get('dates', [])
        if not parsed['date_of_occurrence'] and ner_dates:
            parsed['date_of_occurrence'] = ner_dates[0]
        if not parsed['date_of_report'] and len(ner_dates) > 1:
            parsed['date_of_report'] = ner_dates[1]

    # Time: Look for patterns like 00.00hrs, 00-00hrs.
    time_matches = re.findall(r'\b(\d{1,2}[:.-]\d{2})hrs?\b', cell_text, re.IGNORECASE)
    if time_matches:
        parsed['do_time'] = time_matches[0].strip() + 'hrs'
        if len(time_matches) > 1:
            parsed['dr_time'] = time_matches[1].strip() + 'hrs'

    # Place of Occurrence: Extract text following time until directional keyword or sentence end.
    directions = ['east', 'west', 'north', 'south', 'northeast', 'northwest', 'southeast', 'southwest']
    time_end = 0
    for match in re.finditer(r'\b(\d{1,2}[:.-]\d{2})hrs?\b', cell_text, re.IGNORECASE):
        time_end = match.end()
    if time_end:
        after_time = cell_text[time_end:].strip()
        for dir in directions:
            dir_pos = after_time.lower().find(dir)
            if dir_pos != -1:
                parsed['place_of_occurrence'] = after_time[:dir_pos + len(dir)].strip()
                break
        else:
            # If no direction, take till end
            parsed['place_of_occurrence'] = after_time

    # Use NER if location is missing
    if not parsed['place_of_occurrence']:
        entities = extract_entities_with_ner(cell_text)
        if entities.get('locations'):
            parsed['place_of_occurrence'] = entities['locations'][0]

    # Fallback to old method if needed
    lines = [line.strip() for line in cell_text.replace('\n', ',').split(',') if line.strip()]
    for line in lines:
        line_lower = line.lower()
        if 'date of occurrence' in line_lower or 'do date' in line_lower:
            if not parsed['date_of_occurrence']:
                parsed['date_of_occurrence'] = extract_date(line)
        elif 'date of report' in line_lower or 'dr date' in line_lower:
            if not parsed['date_of_report']:
                parsed['date_of_report'] = extract_date(line)
        elif 'do time' in line_lower:
            if not parsed['do_time']:
                parsed['do_time'] = extract_value(line)
        elif 'dr time' in line_lower:
            if not parsed['dr_time']:
                parsed['dr_time'] = extract_value(line)
        elif 'place of occurrence' in line_lower or 'place' in line_lower:
            if not parsed['place_of_occurrence']:
                parsed['place_of_occurrence'] = extract_value(line)

def parse_column_7(cell_text, parsed):
    """Parse 7th column: Complainant details"""
    # If cell has break (\n), first text as name
    lines = cell_text.split('\n')
    if len(lines) > 1:
        parsed['complainant']['name'] = lines[0].strip()
        # Extract age from name
        age_match = re.search(r'\((\d{1,2})\)|(\d{1,2})(?:\s*/\s*(\d{1,4}))?', parsed['complainant']['name'])
        if age_match:
            parsed['complainant']['age'] = age_match.group(1) or age_match.group(2)
            # Remove age from name
            parsed['complainant']['name'] = re.sub(r'\(?\d{1,2}\)?(?:\s*/\s*\d{1,4})?', '', parsed['complainant']['name']).strip()
        # Address: further lines
        parsed['complainant']['address'] = ' '.join(lines[1:]).strip()
    else:
        # Fallback to old method
        lines = [line.strip() for line in cell_text.replace('\n', ',').split(',') if line.strip()]
        for line in lines:
            line_lower = line.lower()
            if 'name' in line_lower:
                parsed['complainant']['name'] = extract_value(line)
            elif 'address' in line_lower:
                parsed['complainant']['address'] = extract_value(line)
            elif 'sex' in line_lower:
                parsed['complainant']['sex'] = extract_value(line)
            elif 'age' in line_lower:
                parsed['complainant']['age'] = extract_value(line)

    # Sex by S/o, D/o, W/o
    if 'S/o' in cell_text:
        parsed['complainant']['sex'] = 'Male'
    elif 'D/o' in cell_text or 'W/o' in cell_text:
        parsed['complainant']['sex'] = 'Female'

    # Use NER to fill missing name
    entities = extract_entities_with_ner(cell_text)
    if not parsed['complainant']['name'] and entities.get('persons'):
        parsed['complainant']['name'] = entities['persons'][0]
    if not parsed['complainant']['address'] and entities.get('locations'):
        parsed['complainant']['address'] = entities['locations'][0]

def parse_column_8(cell_text, parsed):
    """Parse 8th column: Victim details"""
    # Count victims by numbers 1. xxxx, 2. yyyy or by S/o, D/o count
    victim_blocks = re.split(r'\b(\d+)\.\s*', cell_text)
    victims = []
    if len(victim_blocks) > 1:
        for i in range(1, len(victim_blocks), 2):
            block = victim_blocks[i+1].strip()
            victim = parse_victim_details(block)
            if victim['name']:
                victims.append(victim)
    else:
        # Count by S/o, D/o, W/o
        so_do_count = len(re.findall(r'\b[Ss]/o\b|\b[Dd]/o\b|\b[Ww]/o\b', cell_text))
        if so_do_count > 0:
            # Assume each S/o D/o W/o indicates a victim
            lines = cell_text.split('\n')
            for line in lines:
                if 'S/o' in line or 'D/o' in line or 'W/o' in line:
                    victim = parse_victim_details(line)
                    if victim['name']:
                        victims.append(victim)
        else:
            # Fallback to old method
            lines = [line.strip() for line in cell_text.replace('\n', ',').split(',') if line.strip()]
            current_victim = {'name': '', 'age': '', 'address': '', 'sex': ''}
            for line in lines:
                line_lower = line.lower()
                if 'name' in line_lower:
                    if current_victim['name']:  # Save previous
                        victims.append(current_victim)
                        current_victim = {'name': '', 'age': '', 'address': '', 'sex': ''}
                    value = extract_value(line)
                    if ':' in value:
                        value = extract_value(value)
                    current_victim['name'] = value
                elif 'age' in line_lower:
                    current_victim['age'] = extract_value(line)
                elif 'address' in line_lower:
                    current_victim['address'] = extract_value(line)
                elif 'sex' in line_lower:
                    current_victim['sex'] = extract_value(line)
            if current_victim['name']:
                victims.append(current_victim)
    parsed['victims'] = victims
    parsed['victims_count'] = len(victims)
    parsed['victims_names'] = ', '.join([v['name'] for v in victims if v['name']])
    parsed['victims_ages'] = ', '.join([v['age'] for v in victims if v['age']])

def parse_victim_details(text):
    """Parse individual victim details similar to complainant."""
    victim = {'name': '', 'age': '', 'address': '', 'sex': ''}
    lines = text.split('\n')
    if lines:
        victim['name'] = lines[0].strip()
        # Extract age from name
        age_match = re.search(r'\((\d{1,2})\)|(\d{1,2})(?:\s*/\s*(\d{1,4}))?', victim['name'])
        if age_match:
            victim['age'] = age_match.group(1) or age_match.group(2)
            victim['name'] = re.sub(r'\(?\d{1,2}\)?(?:\s*/\s*\d{1,4})?', '', victim['name']).strip()
        # Address: further lines
        if len(lines) > 1:
            victim['address'] = ' '.join(lines[1:]).strip()
    # Sex by S/o, D/o, W/o
    if 'S/o' in text:
        victim['sex'] = 'Male'
    elif 'D/o' in text or 'W/o' in text:
        victim['sex'] = 'Female'
    return victim

def parse_column_9(cell_text, parsed):
    """Parse 9th column: Property details"""
    lines = [line.strip() for line in cell_text.replace('\n', ',').split(',') if line.strip()]
    for line in lines:
        line_lower = line.lower()
        if 'property lost' in line_lower:
            prop = parse_property(line)
            if prop:
                parsed['property_lost'].append(prop)
        elif 'property recovered' in line_lower:
            prop = parse_property(line)
            if prop:
                parsed['property_recovered'].append(prop)
        elif 'property seized' in line_lower:
            prop = parse_property(line)
            if prop:
                parsed['property_seized'].append(prop)

def parse_column_10(cell_text, parsed):
    """Parse 10th column: Accused details"""
    # Similar to victims: count by numbers 1. xxxx, 2. yyyy or by S/o, D/o count
    accused_blocks = re.split(r'\b(\d+)\.\s*', cell_text)
    accused = []
    if len(accused_blocks) > 1:
        for i in range(1, len(accused_blocks), 2):
            block = accused_blocks[i+1].strip()
            acc = parse_accused_details(block)
            if acc['name']:
                accused.append(acc)
    else:
        # Count by S/o, D/o, W/o
        so_do_count = len(re.findall(r'\b[Ss]/o\b|\b[Dd]/o\b|\b[Ww]/o\b', cell_text))
        if so_do_count > 0:
            # Assume each S/o D/o W/o indicates an accused
            lines = cell_text.split('\n')
            for line in lines:
                if 'S/o' in line or 'D/o' in line or 'W/o' in line:
                    acc = parse_accused_details(line)
                    if acc['name']:
                        accused.append(acc)
        else:
            # Fallback to old method
            lines = [line.strip() for line in cell_text.replace('\n', ',').split(',') if line.strip()]
            current_accused = {'name': '', 'age': '', 'address': '', 'sex': ''}
            for line in lines:
                line_lower = line.lower()
                if 'name' in line_lower:
                    if current_accused['name']:  # Save previous
                        accused.append(current_accused)
                        current_accused = {'name': '', 'age': '', 'address': '', 'sex': ''}
                    value = extract_value(line)
                    if ':' in value:
                        value = extract_value(value)
                    current_accused['name'] = value
                elif 'age' in line_lower:
                    current_accused['age'] = extract_value(line)
                elif 'address' in line_lower:
                    current_accused['address'] = extract_value(line)
                elif 'sex' in line_lower:
                    current_accused['sex'] = extract_value(line)
            if current_accused['name']:
                accused.append(current_accused)
    parsed['accused'] = accused

def parse_accused_details(text):
    """Parse individual accused details similar to complainant."""
    accused = {'name': '', 'age': '', 'address': '', 'sex': ''}
    lines = text.split('\n')
    if lines:
        accused['name'] = lines[0].strip()
        # Extract age from name
        age_match = re.search(r'\((\d{1,2})\)|(\d{1,2})(?:\s*/\s*(\d{1,4}))?', accused['name'])
        if age_match:
            accused['age'] = age_match.group(1) or age_match.group(2)
            accused['name'] = re.sub(r'\(?\d{1,2}\)?(?:\s*/\s*\d{1,4})?', '', accused['name']).strip()
        # Address: further lines
        if len(lines) > 1:
            accused['address'] = ' '.join(lines[1:]).strip()
    # Sex by S/o or D/o
    if 'S/o' in text:
        accused['sex'] = 'Male'
    elif 'D/o' in text or 'W/o' in text:
        accused['sex'] = 'Female'
    return accused

def extract_value(line):
    """Extract value after the first colon."""
    parts = re.split(r':', line, 1)
    if len(parts) > 1:
        return parts[1].strip()
    return line.strip()

def extract_date(line):
    """Extract and parse date with proper error handling."""
    value = extract_value(line)
    if not value:
        return value

    try:
        dt = datetime.strptime(value, '%Y-%m-%d')
        return str(dt.date())
    except (ValueError, TypeError):
        return value

def parse_property(line):
    """Parse property details like item, quantity with units."""
    prop = {'item': '', 'quantity': ''}
    # Units: no's, packs, packets, Rs, grams, kg, bundle, etc.
    units = r'(?:no\'?s?|packs?|packets?|Rs\.?|grams?|kg|bundles?|pieces?|items?)'
    # Find item before quantity
    pattern = rf'(.+?)\s+(\d+(?:\.\d+)?)\s*({units})'
    match = re.search(pattern, line, re.IGNORECASE)
    if match:
        prop['item'] = match.group(1).strip()
        prop['quantity'] = match.group(2) + ' ' + match.group(3)
    else:
        # Fallback to old method
        item_match = re.search(r'item[:\-]\s*([^,]+)', line, re.IGNORECASE)
        if item_match:
            prop['item'] = item_match.group(1).strip()
        quantity_match = re.search(r'quantity[:\-]\s*(\d+)', line, re.IGNORECASE)
        if quantity_match:
            prop['quantity'] = quantity_match.group(1).strip()
    return prop if prop['item'] else None

def extract_and_parse_dates(text):
    """Extract and parse dates using datetime.strptime with common formats."""
    # Common date formats to try
    formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d',
        '%d/%m/%y', '%d-%m-%y', '%y/%m/%d', '%y-%m-%d',
        '%d %b %Y', '%d %B %Y', '%b %d %Y', '%B %d %Y',
        '%d %b %y', '%d %B %y', '%b %d %y', '%B %d %y'
    ]
    # Find potential date strings using regex
    date_strings = re.findall(r'\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b|\b\d{1,2}\s+\w{3,}\s+\d{2,4}\b|\b\w{3,}\s+\d{1,2}\s+\d{2,4}\b', text)
    parsed_dates = []
    for ds in date_strings:
        for fmt in formats:
            try:
                dt = datetime.strptime(ds, fmt)
                parsed_dates.append(str(dt.date()))
                break
            except ValueError:
                continue
    return parsed_dates

