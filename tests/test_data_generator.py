"""
Test Data Generator for ML Integration Testing

Generates comprehensive test data for edge cases, scenarios, and various
FIR document formats to ensure robust testing of all ML components.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from pathlib import Path
import os


class TestDataGenerator:
    """Generator for comprehensive test data."""

    def __init__(self, output_dir: str = "tests/test_data"):
        """Initialize the test data generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Base data templates
        self.indian_names = [
            "Rajesh Kumar", "Priya Sharma", "Amit Singh", "Sneha Patel",
            "Vikram Reddy", "Kavita Gupta", "Rohit Mehta", "Anjali Desai",
            "Suresh Tiwari", "Meera Joshi", "Deepak Agarwal", "Ritu Saxena",
            "Manoj Kumar", "Pooja Sharma", "Arun Singh", "Divya Patel"
        ]

        self.police_stations = [
            "Mumbai Central Police Station", "Andheri Police Station",
            "Bandra Police Station", "Dadar Police Station",
            "Worli Police Station", "Malad Police Station",
            "Goregaon Police Station", "Kandivali Police Station",
            "Borivali Police Station", "Thane Police Station"
        ]

        self.locations = [
            "Andheri East", "Bandra West", "Dadar TT", "Worli Sea Face",
            "Malad West", "Goregaon East", "Kandivali East", "Borivali West",
            "Thane West", "Navi Mumbai", "Panvel", "Kalyan"
        ]

        self.law_sections = [
            "Section 379 IPC", "Section 323 IPC", "Section 420 IPC",
            "Section 406 IPC", "Section 498A IPC", "Section 376 IPC",
            "Section 302 IPC", "Section 307 IPC", "Section 394 IPC"
        ]

        self.property_items = [
            "Mobile Phone", "Laptop Computer", "Gold Chain", "Cash",
            "Motorcycle", "Car", "Television", "Jewelry Box",
            "Watch", "Camera", "Tablet", "Documents"
        ]

    def generate_fir_row_data(self, case_type: str = "standard") -> List[str]:
        """Generate a complete FIR row with various case types."""
        police_station = random.choice(self.police_stations)
        cr_number = f"{random.randint(100, 999)}/{random.randint(2020, 2023)}"

        # Generate dates
        base_date = datetime(2023, random.randint(1, 12), random.randint(1, 28))
        occurrence_date = base_date.strftime("%d/%m/%Y")
        report_date = (base_date + timedelta(days=random.randint(0, 3))).strftime("%d/%m/%Y")
        occurrence_time = f"{random.randint(8, 22):02d}:{random.randint(0, 59):02d}hrs"

        # Generate complainant
        complainant_name = random.choice(self.indian_names)
        complainant_relation = random.choice(["S/o", "D/o", "W/o"])
        complainant_father = random.choice(self.indian_names)
        complainant_age = random.randint(20, 70)
        complainant_address = f"{random.randint(100, 999)} {random.choice(self.locations)}"

        # Generate victim (may be same as complainant or different)
        if random.random() < 0.7:  # 70% chance victim is complainant
            victim_name = complainant_name
            victim_relation = complainant_relation
            victim_father = complainant_father
            victim_age = complainant_age
            victim_address = complainant_address
        else:
            victim_name = random.choice(self.indian_names)
            victim_relation = random.choice(["S/o", "D/o", "W/o"])
            victim_father = random.choice(self.indian_names)
            victim_age = random.randint(20, 70)
            victim_address = f"{random.randint(100, 999)} {random.choice(self.locations)}"

        # Generate property based on case type
        property_info = self._generate_property_info(case_type)

        # Generate accused
        accused_name = random.choice(self.indian_names)
        accused_relation = random.choice(["S/o", "D/o"])
        accused_father = random.choice(self.indian_names)
        accused_age = random.randint(18, 60)
        accused_address = f"{random.randint(100, 999)} {random.choice(self.locations)}"

        # Generate gist based on case type
        gist = self._generate_gist(case_type, complainant_name, accused_name)

        # Construct FIR row
        fir_row = [
            f"{police_station}, CR No {cr_number}, {random.choice(self.law_sections)}",
            f"{occurrence_date}, {report_date}, {occurrence_time}, {random.choice(self.locations)}",
            f"{complainant_name} {complainant_relation} {complainant_father}, Age {complainant_age}, Address: {complainant_address}",
            f"{victim_name} {victim_relation} {victim_father}, Age {victim_age}, Address: {victim_address}",
            property_info,
            f"{accused_name} {accused_relation} {accused_father}, Age {accused_age}, Address: {accused_address}",
            gist
        ]

        return fir_row

    def _generate_property_info(self, case_type: str) -> str:
        """Generate property information based on case type."""
        if case_type == "theft":
            items = random.sample(self.property_items, random.randint(1, 3))
            property_str = "Property Lost: "
            for i, item in enumerate(items):
                value = random.randint(5000, 100000)
                if i > 0:
                    property_str += ", "
                property_str += f"{item} worth Rs. {value:,}"
        elif case_type == "vehicle_theft":
            vehicle = random.choice(["Motorcycle", "Car", "Scooter", "Truck"])
            value = random.randint(50000, 500000)
            property_str = f"Property Lost: {vehicle} worth Rs. {value:,}"
        elif case_type == "cash_theft":
            amount = random.randint(10000, 500000)
            property_str = f"Property Lost: Cash Rs. {amount:,}"
        else:
            property_str = ""

        return property_str

    def _generate_gist(self, case_type: str, complainant: str, accused: str) -> str:
        """Generate case gist based on type."""
        if case_type == "theft":
            return f"{complainant} reported theft of property by {accused}"
        elif case_type == "assault":
            return f"{complainant} was assaulted by {accused} causing injuries"
        elif case_type == "fraud":
            return f"{complainant} was defrauded by {accused} in financial transaction"
        elif case_type == "vehicle_theft":
            return f"{complainant}'s vehicle was stolen by {accused}"
        elif case_type == "burglary":
            return f"{complainant}'s residence was burgled by {accused}"
        else:
            return f"Case reported by {complainant} against {accused}"

    def generate_edge_case_data(self) -> Dict[str, List[str]]:
        """Generate edge case test data."""
        edge_cases = {
            "empty_fields": [
                ", CR No 123/2023, ",
                ", , , ",
                ", , , ",
                "",
                "",
                "",
                "Case with empty fields"
            ],

            "unicode_names": [
                "केंद्र पुलिस स्टेशन, CR No १२३/२०२३, Section ३७९ IPC",
                "१५/०३/२०२३, १६/०३/२०२३, १४:३०hrs, आंधेरी ईस्ट",
                "जॉन डो S/o रॉबर्ट डो, Age ३५, Address: बांद्रा वेस्ट",
                "प्रिया शर्मा D/o अनिल शर्मा, Age २८, Address: आंधेरी ईस्ट",
                "मोबाइल फोन worth Rs. १५,०००",
                "राजेश कुमार S/o विजय कुमार, Age २५, Address: जोगेश्वरी",
                "यूनिकोड टेस्ट केस"
            ],

            "special_characters": [
                "Test@#$% Police Station, CR No 123/456, Section 379 IPC",
                "15/03/2023, Test&Location#",
                "O'Connor-Smith Jr., Age 35, Address: 123 Main@#$%",
                "",
                "Property with $peci@l characters & symbols",
                "Accused@#$% Name, Age 25",
                "Test case with special characters @#$%^&*()"
            ],

            "very_long_content": [
                "A" * 1000 + " Police Station, CR No 999/2023",
                "01/01/2023, " + "A" * 500,
                "Very Long Name " * 50 + ", Age 35",
                "",
                "Very long property description " * 20,
                "Very long accused name " * 30 + ", Age 25",
                "Very long gist description " * 25
            ],

            "minimal_data": [
                "PS, CR 1/23",
                "1/1/23",
                "Name",
                "",
                "",
                "",
                "Minimal"
            ],

            "malformed_dates": [
                "Police Station, CR No 123/2023",
                "32/13/2023, 15/25/2023, 25:70hrs, Invalid Location",
                "Invalid Date Format: 2023/13/45, Age -5",
                "",
                "",
                "",
                "Malformed date formats"
            ],

            "multiple_sections": [
                "Mumbai Police Station, CR No 123/2023, Section 379 IPC, Section 323 IPC, Section 420 IPC, Section 406 IPC",
                "15/03/2023, 16/03/2023, 14:30hrs, Andheri East",
                "John Doe S/o Robert Doe, Age 35",
                "Multiple victims case",
                "Multiple properties lost",
                "Multiple accused persons",
                "Complex case with multiple sections and parties"
            ]
        }

        return edge_cases

    def generate_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Generate comprehensive test suite data."""
        test_suite = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_test_cases": 0,
                "case_types": []
            },
            "standard_cases": [],
            "edge_cases": {},
            "integration_cases": [],
            "performance_cases": []
        }

        # Generate standard test cases
        case_types = ["theft", "assault", "fraud", "vehicle_theft", "burglary"]
        test_suite["metadata"]["case_types"] = case_types

        for case_type in case_types:
            for i in range(10):  # 10 cases per type
                fir_row = self.generate_fir_row_data(case_type)
                test_case = {
                    "id": f"{case_type}_{i+1:03d}",
                    "case_type": case_type,
                    "fir_row": fir_row,
                    "expected_entities": self._extract_expected_entities(fir_row, case_type)
                }
                test_suite["standard_cases"].append(test_case)

        # Generate edge cases
        test_suite["edge_cases"] = self.generate_edge_case_data()

        # Generate integration test cases
        integration_cases = []
        for i in range(20):
            # Mix of different case types for integration testing
            case_type = random.choice(case_types)
            fir_row = self.generate_fir_row_data(case_type)

            integration_case = {
                "id": f"integration_{i+1:03d}",
                "description": f"Integration test case {i+1} - {case_type}",
                "fir_row": fir_row,
                "expected_pipeline_behavior": {
                    "should_parse": True,
                    "should_extract_entities": True,
                    "should_anonymize": True,
                    "should_validate": True
                }
            }
            integration_cases.append(integration_case)

        test_suite["integration_cases"] = integration_cases

        # Generate performance test cases
        performance_cases = []
        for i in range(50):  # Larger set for performance testing
            case_type = random.choice(case_types)
            fir_row = self.generate_fir_row_data(case_type)

            performance_case = {
                "id": f"performance_{i+1:03d}",
                "fir_row": fir_row,
                "expected_performance": {
                    "parsing_time": "< 2.0 seconds",
                    "memory_increase": "< 50MB",
                    "should_not_crash": True
                }
            }
            performance_cases.append(performance_case)

        test_suite["performance_cases"] = performance_cases

        # Update metadata
        test_suite["metadata"]["total_test_cases"] = (
            len(test_suite["standard_cases"]) +
            len(test_suite["edge_cases"]) +
            len(test_suite["integration_cases"]) +
            len(test_suite["performance_cases"])
        )

        return test_suite

    def _extract_expected_entities(self, fir_row: List[str], case_type: str) -> Dict[str, List[str]]:
        """Extract expected entities from FIR row for validation."""
        expected = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": [],
            "LAW": [],
            "MONEY": []
        }

        # Extract from complainant field
        if len(fir_row) > 2 and fir_row[2]:
            complainant_text = fir_row[2]
            # Extract names
            import re
            name_patterns = [
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Regular names
                r'([A-Z][a-z]+\s+S/o\s+[A-Z][a-z]+)',  # Names with S/o
                r'([A-Z][a-z]+\s+D/o\s+[A-Z][a-z]+)',  # Names with D/o
            ]

            for pattern in name_patterns:
                matches = re.findall(pattern, complainant_text)
                expected["PERSON"].extend(matches)

        # Extract police station
        if len(fir_row) > 0 and fir_row[0]:
            ps_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Police\s+Station)', fir_row[0])
            if ps_match:
                expected["ORG"].append(ps_match.group(1))

        # Extract dates
        if len(fir_row) > 1 and fir_row[1]:
            date_patterns = [
                r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
            ]
            for pattern in date_patterns:
                matches = re.findall(pattern, fir_row[1])
                expected["DATE"].extend(matches)

        # Extract law sections
        if len(fir_row) > 0 and fir_row[0]:
            law_patterns = [
                r'(Section\s+\d+\s+IPC)',
            ]
            for pattern in law_patterns:
                matches = re.findall(pattern, fir_row[0])
                expected["LAW"].extend(matches)

        # Extract monetary values
        for field in fir_row:
            if field and "Rs." in field:
                money_patterns = [
                    r'Rs\.\s*(\d+,?\d+)',
                ]
                for pattern in money_patterns:
                    matches = re.findall(pattern, field)
                    expected["MONEY"].extend(matches)

        # Remove duplicates and empty strings
        for entity_type in expected:
            expected[entity_type] = list(set(expected[entity_type]))
            expected[entity_type] = [e for e in expected[entity_type] if e.strip()]

        return expected

    def save_test_suite(self, test_suite: Dict[str, Any], filename: str = "comprehensive_test_suite.json"):
        """Save test suite to file."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_suite, f, ensure_ascii=False, indent=2)

        print(f"Test suite saved to {output_path}")
        print(f"Total test cases: {test_suite['metadata']['total_test_cases']}")

    def generate_fir_text_documents(self, count: int = 20) -> List[str]:
        """Generate sample FIR text documents for testing."""
        documents = []

        for i in range(count):
            # Generate varied FIR content
            police_station = random.choice(self.police_stations)
            complainant = random.choice(self.indian_names)
            accused = random.choice(self.indian_names)
            location = random.choice(self.locations)

            # Create varied document structures
            if i % 4 == 0:
                # Standard format
                doc = f"""
                FIR Details:

                Police Station: {police_station}
                CR No: {random.randint(100, 999)}/{random.randint(2020, 2023)}
                Date of Incident: {random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2023
                Time: {random.randint(8, 22):02d}:{random.randint(0, 59):02d}hrs

                Complainant: {complainant}, Age: {random.randint(20, 70)}
                Address: {random.randint(100, 999)} {location}

                Details: Theft of property by accused {accused}.
                Section: {random.choice(self.law_sections)}
                """
            elif i % 4 == 1:
                # Detailed format
                doc = f"""
                FIRST INFORMATION REPORT

                Station: {police_station}
                Crime No: {random.randint(100, 999)}/{random.randint(2020, 2023)}

                Date and Time of Occurrence: {random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2023 at {random.randint(8, 22):02d}:{random.randint(0, 59):02d}hrs
                Place of Occurrence: {location}

                Complainant/Informant:
                Name: {complainant}
                Father's Name: {random.choice(self.indian_names)}
                Age: {random.randint(20, 70)}
                Address: {random.randint(100, 999)} {location}, Mumbai

                Details of Incident: The complainant reported that the accused {accused} committed theft.

                Legal Sections: {random.choice(self.law_sections)}

                Property Details: {random.choice(self.property_items)} worth Rs. {random.randint(10000, 100000):,}
                """
            elif i % 4 == 2:
                # Tabular format
                doc = f"""
                | Police Station | {police_station} |
                | CR Number | {random.randint(100, 999)}/{random.randint(2020, 2023)} |
                | Date | {random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2023 |
                | Complainant | {complainant} |
                | Accused | {accused} |
                | Location | {location} |
                | Sections | {random.choice(self.law_sections)} |
                """
            else:
                # Narrative format
                doc = f"""
                This is to report that on {random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2023,
                I, {complainant}, aged {random.randint(20, 70)} years, residing at {random.randint(100, 999)} {location},
                wish to report that {accused}, aged {random.randint(18, 60)} years, residing at {random.randint(100, 999)} {random.choice(self.locations)},
                has committed theft of my {random.choice(self.property_items)} worth Rs. {random.randint(10000, 100000):,}.
                The incident occurred at {location} at around {random.randint(8, 22):02d}:{random.randint(0, 59):02d}hrs.
                I request appropriate legal action under {random.choice(self.law_sections)}.
                """

            documents.append(doc.strip())

        return documents

    def generate_malformed_documents(self) -> List[str]:
        """Generate malformed documents for error testing."""
        malformed_docs = [
            "",  # Empty document
            "Random text with no structure",
            "Police Station CR No Section Date Complainant",  # Headers only
            "::: ::: ::: ::: :::",  # Only separators
            "Police Station: [INVALID] CR No: [INVALID] Section: [INVALID]",
            "Very " * 1000 + "long document with repetitive content",  # Very long
            "Document with special chars: !@#$%^&*()[]{}|\\:;\"'<>?/.,`~",
            "Document\nwith\ninconsistent\nline\nbreaks\nand\tmixed\twhitespace",
            "Document with SQL-like content: '; DROP TABLE users; --",
            "Document with HTML-like content: <script>alert('test')</script>",
            "Document with JSON-like content: {'invalid': 'json'}",
            "Document with XML-like content: <invalid><tag>content</tag></invalid>",
        ]

        return malformed_docs

    def generate_unicode_documents(self) -> List[str]:
        """Generate documents with Unicode content."""
        unicode_docs = [
            """
            केंद्र पुलिस स्टेशन, CR No १२३/२०२३, Section ३७९ IPC
            तारीख: १५/०३/२०२३, समय: १४:३०hrs, स्थान: आंधेरी ईस्ट
            शिकायतकर्ता: जॉन डो S/o रॉबर्ट डो, आयु: ३५
            पता: बांद्रा वेस्ट, मुंबई
            """,

            """
            मराठीतील केस: फिर्यादी प्रिया शर्मा यांनी दिलेल्या माहितीनुसार
            आरोपी राजेश कुमार याने चोरी केली आहे.
            सेक्शन ३७९ IPC अंतर्गत कारवाई व्हावी.
            """,

            """
            हिंदी मिक्स: Police Station Mumbai में complainant द्वारा
            theft की रिपोर्ट दर्ज की गई है। Section 420 IPC लागू।
            """
        ]

        return unicode_docs


def main():
    """Generate comprehensive test data."""
    generator = TestDataGenerator()

    print("Generating comprehensive test data...")

    # Generate main test suite
    test_suite = generator.generate_comprehensive_test_suite()
    generator.save_test_suite(test_suite)

    # Generate additional test documents
    print("\nGenerating FIR text documents...")
    fir_docs = generator.generate_fir_text_documents(20)
    for i, doc in enumerate(fir_docs):
        doc_path = generator.output_dir / f"fir_document_{i+1:02d}.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc)

    # Generate malformed documents
    print("Generating malformed documents...")
    malformed_docs = generator.generate_malformed_documents()
    for i, doc in enumerate(malformed_docs):
        doc_path = generator.output_dir / f"malformed_document_{i+1:02d}.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc)

    # Generate Unicode documents
    print("Generating Unicode documents...")
    unicode_docs = generator.generate_unicode_documents()
    for i, doc in enumerate(unicode_docs):
        doc_path = generator.output_dir / f"unicode_document_{i+1:02d}.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc)

    print(f"\nTest data generation completed!")
    print(f"Output directory: {generator.output_dir}")
    print(f"Files generated: {len(list(generator.output_dir.glob('*')))}")


if __name__ == "__main__":
    main()