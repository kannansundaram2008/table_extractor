import unittest
from utils.fir_parser import parse_fir_row, extract_entities_with_ner

class TestFirParser(unittest.TestCase):

    def test_parse_basic_fields(self):
        row_cells = [
            "",  # 0
            "",  # 1
            "",  # 2
            "",  # 3
            "Police Station: Central, CR No: 123/456, Section of Law: IPC 420",  # 4
            "Date of Occurrence: 2023-06-01, Date of Report: 2023-06-02, DO Time: 10:00 AM, DR Time: 11:00 AM, Place of Occurrence: Downtown",  # 5
            "Complainant Name: John Doe, Complainant Address: 123 Main St, Complainant Sex: M, Complainant Age: 35",  # 6
            "Victim 1: Name: Jane Doe, Age: 30, Address: 456 Elm St, Sex: F",  # 7
            "Property Lost: Item: Laptop, Quantity: 1, Property Recovered: Item: Laptop, Quantity: 1, Property Seized: Item: Phone, Quantity: 2",  # 8
            "Accused 1: Name: Richard Roe, Age: 40, Address: 789 Oak St, Sex: M",  # 9
            "Theft reported"  # 10
        ]
        parsed = parse_fir_row(row_cells)
        self.assertEqual(parsed['police_station'], 'Central')
        self.assertEqual(parsed['cr_no'], '123/456')
        self.assertEqual(parsed['section_of_law'], 'IPC 420')
        self.assertEqual(parsed['date_of_occurrence'], '2023-06-01')
        self.assertEqual(parsed['date_of_report'], '2023-06-02')
        self.assertEqual(parsed['do_time'], '10:00 AM')
        self.assertEqual(parsed['dr_time'], '11:00 AM')
        self.assertEqual(parsed['place_of_occurrence'], 'Downtown')
        self.assertEqual(parsed['complainant']['name'], 'John Doe')
        self.assertEqual(parsed['complainant']['address'], '123 Main St')
        self.assertEqual(parsed['complainant']['sex'], 'M')
        self.assertEqual(parsed['complainant']['age'], '35')
        self.assertEqual(parsed['victims_count'], 1)
        self.assertEqual(len(parsed['victims']), 1)
        self.assertEqual(parsed['victims'][0]['name'], 'Jane Doe')
        self.assertEqual(parsed['accused'][0]['name'], 'Richard Roe')
        self.assertEqual(parsed['property_lost'][0]['item'], 'Laptop')
        self.assertEqual(parsed['property_recovered'][0]['item'], 'Laptop')
        self.assertEqual(parsed['property_seized'][0]['item'], 'Phone')
        self.assertEqual(parsed['gist'], 'Theft reported')

    def test_parse_empty(self):
        parsed = parse_fir_row([])
        self.assertEqual(parsed['police_station'], '')
        self.assertEqual(parsed['victims_count'], 0)
        self.assertEqual(parsed['victims'], [])
        self.assertEqual(parsed['accused'], [])
        self.assertEqual(parsed['property_lost'], [])
        self.assertEqual(parsed['gist'], '')

    def test_extract_entities_with_ner(self):
        text = "John Doe reported the incident on June 5th at Downtown."
        entities = extract_entities_with_ner(text)
        # If spaCy is not installed, entities will be empty
        if entities:
            self.assertIn('John Doe', entities.get('persons', []))
            self.assertTrue(any('June' in date for date in entities.get('dates', [])))
            self.assertIn('Downtown', entities.get('locations', []))
        else:
            self.assertEqual(entities, {})

    def test_parse_partial_data(self):
        # Test with partial data, missing some fields
        row_cells = [
            "", "", "", "",
            "Police Station: Central",  # Only police station
            "Date of Occurrence: 2023-06-01",  # Only one date
            "Complainant Name: John Doe",  # Only name
            "",  # Empty victim
            "",  # Empty property
            "",  # Empty accused
            "Theft"  # Gist
        ]
        parsed = parse_fir_row(row_cells)
        self.assertEqual(parsed['police_station'], 'Central')
        self.assertEqual(parsed['date_of_occurrence'], '2023-06-01')
        self.assertEqual(parsed['complainant']['name'], 'John Doe')
        self.assertEqual(parsed['victims_count'], 0)
        self.assertEqual(parsed['gist'], 'Theft')

    def test_parse_malformed_lines(self):
        # Test with malformed lines, extra colons, etc.
        row_cells = [
            "", "", "", "",
            "Police Station:: Central",  # Extra colon
            "Date of Occurrence: 2023-06-01, Invalid: ",  # Invalid line
            "Complainant Name:: John Doe",  # Extra colon
            "Victim 1: Name:: Jane Doe",  # Extra colon
            "Property Lost: Item:: Laptop",  # Extra colon
            "Accused 1: Name:: Richard Roe",  # Extra colon
            "Theft reported"
        ]
        parsed = parse_fir_row(row_cells)
        self.assertEqual(parsed['police_station'], ': Central')  # Extra colon
        self.assertEqual(parsed['date_of_occurrence'], '2023-06-01')
        self.assertEqual(parsed['complainant']['name'], ': John Doe')  # Extra colon
        self.assertEqual(parsed['victims'][0]['name'], ': Jane Doe')  # Extra colon
        self.assertEqual(parsed['property_lost'][0]['item'], ': Laptop')  # Extra colon
        self.assertEqual(parsed['accused'][0]['name'], ': Richard Roe')  # Extra colon

if __name__ == '__main__':
    unittest.main()
