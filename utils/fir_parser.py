import re
from dateutil import parser as date_parser
from datetime import datetime

# Simple NER using regex and heuristics

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

    if len(row_cells) > 4:  # 5th column (index 4)
        parse_column_5(row_cells[4], parsed)
    if len(row_cells) > 5:  # 6th column (index 5)
        parse_column_6(row_cells[5], parsed)
    if len(row_cells) > 6:  # 7th column (index 6)
        parse_column_7(row_cells[6], parsed)
    if len(row_cells) > 7:  # 8th column (index 7)
        parse_column_8(row_cells[7], parsed)
    if len(row_cells) > 8:  # 9th column (index 8)
        parse_column_9(row_cells[8], parsed)
    if len(row_cells) > 9:  # 10th column (index 9)
        parse_column_10(row_cells[9], parsed)
    if len(row_cells) > 10:  # 11th column (index 10)
        parsed['gist'] = row_cells[10].strip()

    return parsed

def parse_column_5(cell_text, parsed):
    """Parse 5th column: Police Station, CR No, Section of Law"""
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
    lines = [line.strip() for line in cell_text.replace('\n', ',').split(',') if line.strip()]
    for line in lines:
        line_lower = line.lower()
        if 'date of occurrence' in line_lower or 'do date' in line_lower:
            parsed['date_of_occurrence'] = extract_date(line)
        elif 'date of report' in line_lower or 'dr date' in line_lower:
            parsed['date_of_report'] = extract_date(line)
        elif 'do time' in line_lower:
            parsed['do_time'] = extract_value(line)
        elif 'dr time' in line_lower:
            parsed['dr_time'] = extract_value(line)
        elif 'place of occurrence' in line_lower or 'place' in line_lower:
            parsed['place_of_occurrence'] = extract_value(line)
    # Use NER to fill missing fields
    entities = extract_entities_with_ner(cell_text)
    if not parsed['date_of_occurrence'] and entities.get('dates'):
        parsed['date_of_occurrence'] = entities['dates'][0]
    if not parsed['date_of_report'] and entities.get('dates') and len(entities['dates']) > 1:
        parsed['date_of_report'] = entities['dates'][1]
    if not parsed['place_of_occurrence'] and entities.get('locations'):
        parsed['place_of_occurrence'] = entities['locations'][0]

def parse_column_7(cell_text, parsed):
    """Parse 7th column: Complainant details"""
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
    # Use NER to fill missing name
    entities = extract_entities_with_ner(cell_text)
    if not parsed['complainant']['name'] and entities.get('persons'):
        parsed['complainant']['name'] = entities['persons'][0]
    if not parsed['complainant']['address'] and entities.get('locations'):
        parsed['complainant']['address'] = entities['locations'][0]

def parse_column_8(cell_text, parsed):
    """Parse 8th column: Victim details"""
    lines = [line.strip() for line in cell_text.replace('\n', ',').split(',') if line.strip()]
    victims = []
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
    lines = [line.strip() for line in cell_text.replace('\n', ',').split(',') if line.strip()]
    accused = []
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

def extract_value(line):
    """Extract value after the first colon."""
    parts = re.split(r':', line, 1)
    if len(parts) > 1:
        return parts[1].strip()
    return line.strip()

def extract_date(line):
    """Extract and parse date."""
    value = extract_value(line)
    try:
        dt = datetime.strptime(value, '%Y-%m-%d')
        return str(dt.date())
    except:
        return value

def parse_property(line):
    """Parse property details like item, quantity."""
    prop = {'item': '', 'quantity': ''}
    item_match = re.search(r'item[:\-]\s*([^,]+)', line, re.IGNORECASE)
    if item_match:
        prop['item'] = item_match.group(1).strip()
    quantity_match = re.search(r'quantity[:\-]\s*(\d+)', line, re.IGNORECASE)
    if quantity_match:
        prop['quantity'] = quantity_match.group(1).strip()
    return prop if prop['item'] else None

def extract_entities_with_ner(text):
    """Extract named entities using simple regex and heuristics."""
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
