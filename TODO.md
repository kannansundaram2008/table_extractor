# FIR Parsing Implementation TODO

## Step 1: Update parse_column_5 (Police Station, CR No, Section of Law)
- Implement CR No pattern matching: {1,4}/{2,4} (e.g., 123/23, 1/2022)
- Fallback: Check previous and next adjacent cells if CR No not found in current cell
- Police Station: Extract text preceding CR No, handle breaks (newline/delimiter), select second fragment if both >3 chars
- Section of Law: All remaining text after CR No

## Step 2: Update parse_column_6 (Dates, Times, Place of Occurrence)
- Dates: Use datetime.strptime for standard formats, handle 2 dates (occurrence, report), 3 dates (occurrence range, report)
- Fallback: Use NER if dates missing or ambiguous
- Time: Look for patterns like 00.00hrs, 00-00hrs, typically precedes directional keywords
- Place of Occurrence: Extract text following time until directional keyword or sentence end, use NER if missing

## Step 3: Update parse_column_7 (Complainant Details)
- Name: If cell contains break, take first string as name
- Age: Look for patterns like (dd), dd, dd/dd, dd/dddd
- Sex: Infer from presence of S/o (male) or D/o (female)
- Address: Remaining text after name and age
- Fallback: Use NER to extract name/address if missing

## Step 4: Update parse_column_8 (Victim Details - Multiple Supported)
- Segmentation: Detect multiple victims using numbered format (1. xxxx, 2. yyyy) or count of S/o, D/o occurrences
- Attributes: Apply same logic as Complainant for name, age, sex, address

## Step 5: Update parse_column_9 (Property Details)
- Item/Quantity Parsing: Use regex to extract units (nos, packs, packets, Rs, grams, kg, bundle, etc.)
- Normalize into structured format: {item: quantity}

## Step 6: Update parse_column_10 (Accused Details - Multiple Supported)
- Segmentation & Attributes: Same logic as Victim parsing (numbered format or S/o, D/o count)
- Extract name, age, sex, address

## Step 7: Update parse_column_11 (Gist / Remarks)
- Aggregation: Combine all remaining unparsed or overflow data
- Preserve sentence structure and context

## Step 8: Test and Validate
- Run the app with sample data
- Check further_extraction page for correct parsing
- Verify CSV/Excel/PDF exports
