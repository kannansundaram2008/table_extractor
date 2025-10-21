import re
from dateutil import parser as date_parser

PRIMARY_REGEX = re.compile(r'\d{1,4}\s*/\s*\d{1,4}')
PRIMARY_KEYWORDS = re.compile(r'\b(ipc|bns|bnss|crpc|cr\.pc|act|u?/?s)\b', re.IGNORECASE)

DIRECTIONS = ['east', 'west', 'north', 'south', 'ne', 'se', 'nw', 'sw', 'northeast', 'southeast', 'northwest', 'southwest']

def has_primary_pattern(cell):
    return PRIMARY_REGEX.search(cell) and PRIMARY_KEYWORDS.search(cell)

def has_adjacent_pattern(cell):
    has_time_or_date = 'hrs' in cell.lower() or is_date(cell)
    has_direction = any(dir in cell.lower() for dir in DIRECTIONS)
    has_officer = any(kw in cell.lower() for kw in ['i/o', 'ins', 'si', 'ssi'])
    return (has_time_or_date + has_direction + has_officer) >= 2

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
