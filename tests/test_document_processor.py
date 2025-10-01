import unittest
import os
from utils.document_processor import convert_doc_to_docx, extract_tables_from_docx, process_folder, discard_adjacent_duplicates

class TestDocumentProcessor(unittest.TestCase):

    def test_convert_doc_to_docx(self):
        # This test requires a sample .doc file in test_data folder
        doc_path = 'tests/test_data/sample.doc'
        if os.path.exists(doc_path):
            docx_path = convert_doc_to_docx(doc_path)
            self.assertTrue(docx_path.endswith('.docx'))
            self.assertTrue(os.path.exists(docx_path))
            os.remove(docx_path)
        else:
            self.skipTest("Sample .doc file not found")

    def test_extract_tables_from_docx(self):
        docx_path = 'tests/test_data/sample.docx'
        if os.path.exists(docx_path):
            tables = extract_tables_from_docx(docx_path)
            self.assertIsInstance(tables, list)
            self.assertTrue(all(isinstance(table, list) for table in tables))
        else:
            self.skipTest("Sample .docx file not found")

    def test_process_folder(self):
        folder_path = 'tests/test_data'
        if os.path.exists(folder_path):
            result = process_folder(folder_path)
            self.assertIn('matched_rows', result)
            self.assertIn('processed_files', result)
            self.assertIn('errors', result)
        else:
            self.skipTest("Test data folder not found")

    def test_discard_adjacent_duplicates(self):
        # Test cases for discard_adjacent_duplicates
        self.assertEqual(discard_adjacent_duplicates([]), [])
        self.assertEqual(discard_adjacent_duplicates(['a']), ['a'])
        self.assertEqual(discard_adjacent_duplicates(['a', 'a']), ['a'])
        self.assertEqual(discard_adjacent_duplicates(['a', 'a', 'b']), ['a', 'b'])
        self.assertEqual(discard_adjacent_duplicates(['a', 'b', 'b', 'c']), ['a', 'b', 'c'])
        self.assertEqual(discard_adjacent_duplicates(['a', 'a', 'a']), ['a'])
        self.assertEqual(discard_adjacent_duplicates(['x', 'y', 'z']), ['x', 'y', 'z'])

    def test_merged_row_appending(self):
        # Test to verify merged row appending logic
        table = [
            ['A', 'B', 'C'],
            ['A', 'B', 'C'],
            ['MergedValue'],
            ['D', 'E', 'F']
        ]
        matched_rows = []
        row_index = 0
        while row_index < len(table):
            row = discard_adjacent_duplicates(table[row_index])
            if len(row) > 0 and row != ['MergedValue']:
                matched_row = {'row': row[:]}
                if row_index + 1 < len(table):
                    next_row = discard_adjacent_duplicates(table[row_index + 1])
                    if len(next_row) == 1:
                        matched_row['row'].append(next_row[0])
                        row_index += 1
                matched_rows.append(matched_row)
            row_index += 1
        self.assertEqual(len(matched_rows), 3)
        self.assertEqual(matched_rows[1]['row'][-1], 'MergedValue')

if __name__ == '__main__':
    unittest.main()
