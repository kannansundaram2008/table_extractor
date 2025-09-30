import re
from dateutil import parser as date_parser

PRIMARY_REGEX = re.compile(r'\d{1,4}\s*/\s*\d{1,4}')
PRIMARY_KEYWORDS = re.compile(r'\b(ipc|bns|act|u/?s)\b', re.IGNORECASE)

DIRECTIONS = ['east', 'west', 'north', 'south', 'ne', 'se', 'nw', 'sw', 'northeast', 'southeast', 'northwest', 'southwest']

def has_primary_pattern(cell):
    return PRIMARY_REGEX.search(cell) and PRIMARY_KEYWORDS.search(cell)

def has_adjacent_pattern(cell):
    has_time_or_date = 'hrs' in cell.lower() or is_date(cell)
    has_direction = any(dir in cell.lower() for dir in DIRECTIONS)
    return has_time_or_date and has_direction

def is_date(text):
    try:
        date_parser.parse(text, fuzzy=True)
        return True
    except:
        return False

def match_row(row):
    # Check if any cell has primary pattern
    primary_cells = [i for i, cell in enumerate(row) if has_primary_pattern(cell)]
    if not primary_cells:
        return False
    # Check the adjacent cell for pattern
    for i in primary_cells:
        if i + 1 < len(row) and has_adjacent_pattern(row[i + 1]):
            return True
    return False

def append_merged_cell_if_applicable(table, current_row_index):
    if current_row_index + 1 < len(table) and len(table[current_row_index + 1]) == 1:
        return table[current_row_index + 1][0]
    return None
