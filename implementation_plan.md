# Implementation Plan

The goal is to implement a new feature that adds a "Further Extraction" option on the results page after table extraction. This feature will parse each cell in the matched table rows into structured data fields related to police FIR (First Information Report) details, including police station, CR number, section of law, dates, times, complainant details, victim details, accused details, property details, and the gist.

The implementation will involve creating a new route and template for the further extraction page, developing a parser function to split cell text into the specified fields, and updating the results page to include a link to this new functionality. The parser will use regex patterns to extract key-value pairs and handle lists for multiple victims, accused, and property items.

[Types]
No new type definitions are required as the existing data structures (dictionaries for matched_rows) will be extended with parsed fields. The parsed data will be stored as nested dictionaries and lists, e.g., {'police_station': str, 'cr_no': str, 'victims': [{'name': str, 'age': int, 'address': str, 'sex': str}], ...}.

[Files]
New files to be created:
- utils/fir_parser.py: Contains the parse_fir_cell function that takes a cell string and returns a dictionary with all extracted fields.
- templates/further_extraction.html: Template for displaying the parsed FIR data in a structured format, with sections for complainant, victims, accused, property, and gist.

Existing files to be modified:
- app.py: Add a new route '/further_extraction' that loads the result data, parses each cell using the new parser, and renders the further_extraction template.
- templates/results.html: Add a "Further Extraction" button/link next to the export buttons that navigates to the new '/further_extraction' route.

[Functions]
New functions:
- parse_fir_cell(cell_text: str) -> dict: In utils/fir_parser.py, parses the cell text using regex to extract fields like police_station, cr_no, section_of_law, date_of_occurrence, date_of_report, do_time, dr_time, complainant (dict with name, address, sex, age), victims_count, victims (list of dicts), accused (list of dicts), property_lost (list of dicts), property_recovered (list of dicts), property_seized (list of dicts), gist.

Modified functions:
- results() in app.py: No changes, but the new route will reuse similar logic for loading result data.

[Classes]
No new or modified classes are required for this implementation.

[Dependencies]
No new dependencies are needed. The existing regex and string processing capabilities in Python are sufficient.

[Testing]
Add unit tests in tests/test_fir_parser.py for the parse_fir_cell function, covering various FIR text formats and edge cases. Update the end-to-end test in tests/test_end_to_end.py to include navigation to the further extraction page and verification of parsed data display.

[Implementation Order]
1. Create utils/fir_parser.py with the parse_fir_cell function.
2. Add the '/further_extraction' route to app.py.
3. Create templates/further_extraction.html.
4. Update templates/results.html to add the "Further Extraction" link.
5. Add tests for the new functionality.
