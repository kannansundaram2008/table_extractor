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

    if len(row_cells) > 2:  # 3rd column (index 2) -> Police Station, CR No, Section of Law
        parse_column_5(row_cells[2], parsed)
    if len(row_cells) > 3:  # 4th column (index 3) -> Dates, Times, Place of Occurrence
        parse_column_6(row_cells[3], parsed)
    if len(row_cells) > 4:  # 5th column (index 4) -> Complainant Details
        parse_column_7(row_cells[4], parsed)
    if len(row_cells) > 5:  # 6th column (index 5) -> Victim Details
        parse_column_8(row_cells[5], parsed)
    if len(row_cells) > 5:  # 6th column (index 5) -> Property Details
        parse_column_9(row_cells[5], parsed)
    if len(row_cells) > 6:  # 7th column (index 6) -> Accused Details
        parse_column_10(row_cells[6], parsed)
    if len(row_cells) > 7:  # 8th column (index 7) -> Gist / Remarks
        parsed['gist'] = row_cells[7].strip()+row_cells[8].strip()
    return parsed

def parse_column_5(cell_text, parsed):
    """Parse 5th column: Police Station, CR No, Section of Law"""
    # Find CR No using pattern {1,4}/{2,4}
    cr_match = re.search(r'\b(\d{1,4}/\d{2,4})\b', cell_text)
    if cr_match:
        parsed['cr_no'] = cr_match.group(1)
        # Police station before CR No
        before_cr = cell_text[:cr_match.start()].strip()
        if before_cr:
            parts = before_cr.split()
            if len(parts) > 1 and all(len(p) > 3 for p in parts):
                parsed['police_station'] = parts[1]  # Second part if break
            else:
                parsed['police_station'] = before_cr
            # Clean police station: remove C.no, Cr.No, " PS", and CR No pattern case insensitive
            parsed['police_station'] = re.sub(r'\b[Cc]\.?\s*[Nn][Oo]\b', '', parsed['police_station'], flags=re.IGNORECASE)
            parsed['police_station'] = re.sub(r'\s+PS\b', '', parsed['police_station'], flags=re.IGNORECASE)
            parsed['police_station'] = re.sub(r'\b\d{1,4}/\d{2,4}\b', '', parsed['police_station'], flags=re.IGNORECASE)
            parsed['police_station'] = parsed['police_station'].strip()
        # Section of law: remaining after CR No
        after_cr = cell_text[cr_match.end():].strip()
        if after_cr:
            parsed['section_of_law'] = after_cr
    else:
        # Fallback to old method
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
    # Extract all dates
    entities = extract_entities_with_ner(cell_text)
    dates = entities.get('dates', [])
    if len(dates) >= 2:
        parsed['date_of_occurrence'] = dates[0].strip()
        parsed['date_of_report'] = dates[1].strip()
    elif len(dates) == 3:
        parsed['date_of_occurrence'] = f"{dates[0].strip()} to {dates[1].strip()}"  # Range for occurrence
        parsed['date_of_report'] = dates[2].strip()
    elif len(dates) == 1:
        parsed['date_of_occurrence'] = dates[0].strip()

    # Extract times with hrs pattern
    time_matches = re.findall(r'\b(\d{1,2}[:.-]\d{2})hrs?\b', cell_text, re.IGNORECASE)
    if time_matches:
        parsed['do_time'] = time_matches[0].strip() + 'hrs'
        if len(time_matches) > 1:
            parsed['dr_time'] = time_matches[1].strip() + 'hrs'

    # Extract place after time till direction
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
            # If no direction, take till end or next keyword
            parsed['place_of_occurrence'] = after_time

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
    """Extract and parse date."""
    value = extract_value(line)
    try:
        dt = datetime.strptime(value, '%Y-%m-%d')
        return str(dt.date())
    except:
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
