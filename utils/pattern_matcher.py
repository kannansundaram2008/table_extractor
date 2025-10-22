import re
from dateutil import parser as date_parser

# Enhanced primary patterns for FIR detection
PRIMARY_REGEX = re.compile(r'\d{1,4}\s*/\s*\d{1,4}')
PRIMARY_KEYWORDS = re.compile(r'\b(ipc|bns|bnss|crpc|cr\.pc|act|u?/?s|section|police|station|fir|complaint)\b', re.IGNORECASE)

# Enhanced direction patterns including Hindi
DIRECTIONS = [
    'east', 'west', 'north', 'south', 'ne', 'se', 'nw', 'sw',
    'northeast', 'southeast', 'northwest', 'southwest',
    'पूर्व', 'पश्चिम', 'उत्तर', 'दक्षिण', 'उत्तरी', 'दक्षिणी', 'पूर्वी', 'पश्चिमी'
]

# Enhanced time patterns
TIME_PATTERNS = [
    r'\b\d{1,2}[:.]\d{2}\s*(?:AM|PM|hrs?|hours?|मिनट|घंटे)\b',
    r'\b\d{1,2}[:.]\d{2}\s*hours?\b',
    r'\b\d{1,2}\s*(?:AM|PM|hrs?|hours?)\b'
]

# Enhanced date patterns
DATE_PATTERNS = [
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
    r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',
]

# Enhanced police station patterns
POLICE_STATION_PATTERNS = [
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Police\s+Station\b',
    r'\bPS\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Thana\b',
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Police\s+Chowki\b',
]

# Enhanced legal section patterns
LEGAL_SECTION_PATTERNS = [
    r'\bIPC\s+\d+(?:\([^)]+\))?(?:\s*,\s*\d+(?:\([^)]+\))?)*\b',
    r'\bSection\s+\d+(?:\([^)]+\))?\s+of\s+IPC\b',
    r'\bBNS\s+\d+(?:\([^)]+\))?(?:\s*,\s*\d+(?:\([^)]+\))?)*\b',
    r'\bSection\s+\d+(?:\([^)]+\))?\s+of\s+BNS\b',
    r'\bU/S\s+\d+(?:\([^)]+\))?(?:\s*,\s*\d+(?:\([^)]+\))?)*\b',
]

# Enhanced person name patterns
PERSON_NAME_PATTERNS = [
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Multi-word names
    r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b',  # Names with initials
    r'\b[A-Z][a-z]+\s+S/o\s+[A-Z][a-z]+\b',  # Son of patterns
    r'\b[A-Z][a-z]+\s+D/o\s+[A-Z][a-z]+\b',  # Daughter of patterns
    r'\b[A-Z][a-z]+\s+W/o\s+[A-Z][a-z]+\b',  # Wife of patterns
]

def has_primary_pattern(cell):
    """Enhanced primary pattern detection for FIR documents."""
    if not cell or not isinstance(cell, str):
        return False

    # Check for CR number pattern
    has_cr_number = bool(PRIMARY_REGEX.search(cell))

    # Check for legal keywords
    has_legal_keywords = bool(PRIMARY_KEYWORDS.search(cell))

    # Check for police station patterns
    has_police_station = any(re.search(pattern, cell, re.IGNORECASE) for pattern in POLICE_STATION_PATTERNS)

    # Check for legal section patterns
    has_legal_section = any(re.search(pattern, cell, re.IGNORECASE) for pattern in LEGAL_SECTION_PATTERNS)

    return has_cr_number and (has_legal_keywords or has_police_station or has_legal_section)

def has_adjacent_pattern(cell):
    """Enhanced adjacent pattern detection with comprehensive patterns."""
    if not cell or not isinstance(cell, str):
        return False

    score = 0

    # Check for time patterns
    has_time = any(re.search(pattern, cell, re.IGNORECASE) for pattern in TIME_PATTERNS)
    if has_time:
        score += 1

    # Check for date patterns
    has_date = any(re.search(pattern, cell) for pattern in DATE_PATTERNS) or is_date(cell)
    if has_date:
        score += 1

    # Check for direction patterns
    has_direction = any(dir in cell.lower() for dir in DIRECTIONS)
    if has_direction:
        score += 1

    # Check for officer/investigator patterns
    officer_keywords = ['i/o', 'ins', 'si', 'ssi', 'io', 'investigating officer', 'sub inspector', 'inspector']
    has_officer = any(kw in cell.lower() for kw in officer_keywords)
    if has_officer:
        score += 1

    # Check for person name patterns
    has_person = any(re.search(pattern, cell, re.IGNORECASE) for pattern in PERSON_NAME_PATTERNS)
    if has_person:
        score += 1

    # Check for location indicators
    location_indicators = ['at', 'in', 'near', 'from', 'to', 'opposite', 'behind', 'beside', 'में', 'पर']
    has_location_indicator = any(indicator in cell.lower() for indicator in location_indicators)
    if has_location_indicator:
        score += 1

    return score >= 2

def has_time_pattern(cell):
    """Check if cell contains time patterns."""
    return any(re.search(pattern, cell, re.IGNORECASE) for pattern in TIME_PATTERNS)

def has_date_pattern(cell):
    """Check if cell contains date patterns."""
    return any(re.search(pattern, cell) for pattern in DATE_PATTERNS) or is_date(cell)

def has_police_station_pattern(cell):
    """Check if cell contains police station patterns."""
    return any(re.search(pattern, cell, re.IGNORECASE) for pattern in POLICE_STATION_PATTERNS)

def has_legal_section_pattern(cell):
    """Check if cell contains legal section patterns."""
    return any(re.search(pattern, cell, re.IGNORECASE) for pattern in LEGAL_SECTION_PATTERNS)

def has_person_pattern(cell):
    """Check if cell contains person name patterns."""
    return any(re.search(pattern, cell, re.IGNORECASE) for pattern in PERSON_NAME_PATTERNS)

def is_date(text):
    """Check if text is a valid date with proper error handling."""
    if not text or not isinstance(text, str):
        return False

    try:
        date_parser.parse(text, fuzzy=True)
        return True
    except (ValueError, TypeError, OverflowError):
        return False

def match_row(row):
    """Match FIR rows with proper bounds checking."""
    if not row or not isinstance(row, list):
        return False

    # Check if any cell has primary pattern
    primary_cells = [i for i, cell in enumerate(row) if has_primary_pattern(cell)]
    if not primary_cells:
        return False

    # Check the adjacent cell for pattern with bounds checking
    for i in primary_cells:
        if i + 1 < len(row) and has_adjacent_pattern(row[i + 1]):
            return True
    return False

def append_merged_cell_if_applicable(table, current_row_index):
    if current_row_index + 1 < len(table):
        next_row = table[current_row_index + 1]
        if len(next_row) == 1:
            cell = next_row[0]
            colspan = cell.get('colspan', 1) if isinstance(cell, dict) else 1
            if colspan > 1:
                return cell
    return None
