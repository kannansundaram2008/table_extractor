#!/usr/bin/env python3
"""
Test Data Generation Script for FIR Document Testing

This script programmatically generates various test files for FIR document processing tests.
It creates .doc files with different structures, edge cases, and error scenarios.
"""

import os
import random
import string
from datetime import datetime, timedelta

class FIRTestDataGenerator:
    def __init__(self, output_dir="tests/test_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def random_string(self, length=10):
        """Generate a random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def random_date(self):
        """Generate a random date string."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2025, 12, 31)
        random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        return random_date.strftime("%Y-%m-%d")

    def random_time(self):
        """Generate a random time string."""
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        return f"{hour:02d}:{minute:02d}"

    def generate_basic_fir(self, case_number=1):
        """Generate a basic FIR document."""
        police_stations = [
            "Central Police Station",
            "East Police Station",
            "West Police Station",
            "North Police Station",
            "South Police Station"
        ]

        sections = [
            "IPC 420",
            "IPC 379",
            "IPC 302",
            "IPC 395",
            "BNS 101",
            "IPC 420, 406"
        ]

        names = [
            "Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sunita Singh",
            "Vikram Reddy", "Anita Desai", "Rohit Gupta", "Kavita Jain"
        ]

        content = f"""FIR Test Case {case_number}

Police Station: {random.choice(police_stations)}
CR No: {random.randint(100, 999)}/{random.randint(2020, 2025)}
Section of Law: {random.choice(sections)}

Date of Occurrence: {self.random_date()}
Date of Report: {self.random_date()}
Time of Occurrence: {self.random_time()}
Time of Report: {self.random_time()}
Place of Occurrence: {self.random_string(15)} Area, Bangalore

Complainant Details:
Name: {random.choice(names)}
Address: {random.randint(100, 999)} {self.random_string(10)} Road, Bangalore
Age: {random.randint(20, 70)}
Sex: {random.choice(['Male', 'Female'])}

Victim Details:
Name: {random.choice(names)}
Address: {random.randint(100, 999)} {self.random_string(10)} Street, Bangalore
Age: {random.randint(15, 80)}
Sex: {random.choice(['Male', 'Female'])}

Property Lost:
Item: {random.choice(['Mobile Phone', 'Gold Chain', 'Laptop', 'Cash', 'Watch'])}
Value: ₹{random.randint(1000, 100000)}

Accused Details:
Name: {random.choice(names)}
Address: {random.randint(100, 999)} {self.random_string(10)} Nagar, Bangalore
Age: {random.randint(20, 60)}
Sex: {random.choice(['Male', 'Female'])}

Gist of Complaint:
{random.choice([
    'Theft of valuable property reported by complainant.',
    'Burglary case registered with stolen items recovered partially.',
    'Cheating case involving financial transaction.',
    'Assault case with injury to victim reported.'
])}

"""
        filename = f"tests/test_data/generated_fir_case_{case_number}.doc"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return filename

    def generate_empty_fir(self):
        """Generate an FIR document with empty fields."""
        content = """Empty FIR Test Case

Police Station:
CR No:
Section of Law:

Date of Occurrence:
Date of Report:
Time of Occurrence:
Time of Report:
Place of Occurrence:

Complainant Details:
Name:
Address:
Age:
Sex:

Victim Details:
Name:
Address:
Age:
Sex:

Property Lost:
Item:
Value:

Accused Details:
Name:
Address:
Age:
Sex:

Gist of Complaint:


"""
        filename = "tests/test_data/empty_fir_test.doc"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return filename

    def generate_malformed_fir(self):
        """Generate an FIR document with malformed data."""
        content = """Malformed FIR Test Case

Police Station:: {random.choice(['Central', 'East', 'West'])} Police Station
CR No:: {random.randint(100, 999)}::/::{random.randint(2020, 2025)}
Section of Law:: IPC:: {random.randint(100, 500)}::

Date of Occurrence:: {self.random_date()}
Date of Report:: {self.random_date()}
Place of Occurrence:: {self.random_string(20)}:::: Area

Complainant Details::
Name:: {random.choice(['John', 'Jane'])}:::: {random.choice(['Doe', 'Smith'])}
Address:: {random.randint(100, 999)}:::: {self.random_string(10)}:::: Street
Age:: {random.randint(20, 60)}::
Sex:: {random.choice(['M', 'F'])}::

Victim Details::
Name:: {random.choice(['John', 'Jane'])}:::: {random.choice(['Doe', 'Smith'])}
Address:: {random.randint(100, 999)}:::: {self.random_string(10)}:::: Street
Age:: {random.randint(15, 70)}::
Sex:: {random.choice(['M', 'F'])}::

Property Lost::
Item:: {random.choice(['Phone', 'Laptop', 'Cash'])}:::: {random.choice(['Samsung', 'Dell', 'Cash'])}
Value:: ₹{random.randint(1000, 50000)}::::

Accused Details::
Name:: {random.choice(['Richard', 'Robert'])}:::: {random.choice(['Roe', 'Smith'])}
Address:: {random.randint(100, 999)}:::: {self.random_string(10)}:::: Nagar
Age:: {random.randint(20, 60)}::
Sex:: {random.choice(['M', 'F'])}::

Gist of Complaint::
{random.choice(['Theft', 'Burglary', 'Cheating'])}:::: reported:::: with:::: extra:::: colons::::

"""
        filename = "tests/test_data/malformed_fir_test.doc"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return filename

    def generate_unicode_fir(self):
        """Generate an FIR document with Unicode characters."""
        unicode_names = [
            "Râjêsh Kûmär", "Prîyâ Shärmâ", "Àmît Pätél", "Sûnîtâ Sîngh",
            "Vîkrâm Rëddy", "Ânîtâ Désâî", "Röhît Gûptâ", "Kâvîtâ Jâîn"
        ]

        content = f"""Unicode FIR Test Case

Police Station: Çëntrâl Pölîçë Stâtïön
CR No: {random.randint(100, 999)}/2023
Section of Law: ÍPC 420, 406

Date of Occurrence: {self.random_date()}
Date of Report: {self.random_date()}
Time of Occurrence: {self.random_time()}
Time of Report: {self.random_time()}
Place of Occurrence: Çømmërcíâl Strëët Âréâ

Complainant Details:
Name: {random.choice(unicode_names)}
Address: 123 Màîñ Røâd, Bângâløré
Age: {random.randint(20, 70)}
Sex: Málé

Victim Details:
Name: {random.choice(unicode_names)}
Address: 456 Élm Strëët, Bângâløré
Age: {random.randint(15, 80)}
Sex: Fémálé

Property Lost:
Item: Läptöp Çømpütër
Value: ₹{random.randint(10000, 100000)}

Accused Details:
Name: {random.choice(unicode_names)}
Address: 789 Øåk Nëgår, Bângâløré
Age: {random.randint(20, 60)}
Sex: Málé

Gist of Complaint:
Ûñíçødé tëxt tëst çåsé wíth ñøñ-ÂSCÍÍ çhârâçtërs för tëstîñg èñçødîñg håñdlîñg.

"""
        filename = "tests/test_data/unicode_fir_test.doc"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return filename

    def generate_large_fir(self):
        """Generate an FIR document with many entries."""
        content = """Large FIR Test Case

Police Station: Metropolitan Police Station
CR No: 777/2023
Section of Law: IPC 395, 397, 398, 399, 400, 401, 402

Date of Occurrence: 2023-06-15
Date of Report: 2023-06-15
Place of Occurrence: National Highway Junction

Complainant Details:
Name: Transport Company Owner
Address: Industrial Area, Bangalore
Age: 55
Sex: Male

Victim Details:
"""

        # Add multiple victims
        for i in range(10):
            content += f"""Victim {i+1}: Name: Victim Name {i+1}, Age: {random.randint(20, 60)}, Address: {random.randint(100, 999)} {self.random_string(8)} Street, Bangalore, Sex: {random.choice(['Male', 'Female'])}

"""

        content += """
Property Lost:
"""
        # Add multiple lost items
        items = ['Truck', 'Goods Container', 'Cash Box', 'Documents', 'Mobile Phones', 'Laptops']
        for i, item in enumerate(items):
            content += f"""Item {i+1}: {item}, Value: ₹{random.randint(10000, 500000)}
"""

        content += """
Accused Details:
"""
        # Add multiple accused
        for i in range(8):
            content += f"""Accused {i+1}: Name: Accused Name {i+1}, Age: {random.randint(20, 50)}, Address: {random.randint(100, 999)} {self.random_string(8)} Nagar, Bangalore, Sex: {random.choice(['Male', 'Female'])}
"""

        content += """
Gist of Complaint:
Large scale robbery involving multiple victims, significant property loss, and multiple accused persons. The case involves organized crime with sophisticated planning and execution. Investigation requires coordination with multiple agencies and extensive forensic analysis.

"""
        filename = "tests/test_data/large_fir_test.doc"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return filename

    def generate_all_test_files(self, num_basic_files=5):
        """Generate all types of test files."""
        files_created = []

        # Generate basic FIR files
        for i in range(num_basic_files):
            files_created.append(self.generate_basic_fir(i + 1))

        # Generate special test cases
        files_created.append(self.generate_empty_fir())
        files_created.append(self.generate_malformed_fir())
        files_created.append(self.generate_unicode_fir())
        files_created.append(self.generate_large_fir())

        return files_created

def main():
    """Main function to generate all test data."""
    generator = FIRTestDataGenerator()

    print("Generating FIR test data files...")
    files_created = generator.generate_all_test_files()

    print(f"Created {len(files_created)} test files:")
    for file in files_created:
        print(f"  - {file}")

    print("\nTest data generation completed!")
    print("Files are ready for use in FIR document processing tests.")

if __name__ == "__main__":
    main()