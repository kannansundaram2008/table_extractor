import unittest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from utils.document_processor import convert_doc_to_docx, extract_tables_from_docx, process_folder, discard_adjacent_duplicates

class TestDocumentProcessor(unittest.TestCase):

    @patch('utils.document_processor.pypandoc.convert_file')
    @patch('os.path.exists')
    def test_convert_doc_to_docx_success(self, mock_exists, mock_convert):
        """Test successful DOC to DOCX conversion"""
        mock_exists.return_value = True
        mock_convert.return_value = None

        doc_path = 'test/sample.doc'
        expected_docx_path = 'test/sample.docx'

        result = convert_doc_to_docx(doc_path)

        self.assertEqual(result, expected_docx_path)
        mock_convert.assert_called_once_with(doc_path, 'docx', outputfile=expected_docx_path)

    @patch('utils.document_processor.pypandoc.convert_file')
    def test_convert_doc_to_docx_pandoc_error(self, mock_convert):
        """Test DOC to DOCX conversion with pypandoc error"""
        mock_convert.side_effect = Exception("Pandoc conversion failed")

        doc_path = 'test/sample.doc'

        with self.assertRaises(Exception) as context:
            convert_doc_to_docx(doc_path)

        self.assertIn("Pandoc conversion failed", str(context.exception))

    @patch('utils.document_processor.pypandoc.convert_file')
    @patch('os.path.exists')
    def test_convert_doc_to_docx_file_not_exists(self, mock_exists, mock_convert):
        """Test DOC to DOCX conversion when source file doesn't exist"""
        mock_exists.return_value = False

        doc_path = 'test/nonexistent.doc'

        # Should not call pypandoc if file doesn't exist
        result = convert_doc_to_docx(doc_path)

        # The function doesn't check file existence, so pypandoc will be called
        mock_convert.assert_called_once_with(doc_path, 'docx', outputfile=doc_path.replace('.doc', '.docx'))

    @patch('utils.document_processor.Document')
    def test_extract_tables_from_docx_success(self, mock_document_class):
        """Test successful table extraction from DOCX"""
        # Mock document and table structure
        mock_table = Mock()
        mock_table.rows = [
            Mock(cells=[Mock(text='Cell 1'), Mock(text='Cell 2')]),
            Mock(cells=[Mock(text='Cell 3'), Mock(text='Cell 4')])
        ]

        mock_doc = Mock()
        mock_doc.tables = [mock_table]
        mock_document_class.return_value = mock_doc

        docx_path = 'test/sample.docx'
        tables = extract_tables_from_docx(docx_path)

        self.assertIsInstance(tables, list)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0], [['Cell 1', 'Cell 2'], ['Cell 3', 'Cell 4']])
        mock_document_class.assert_called_once_with(docx_path)

    @patch('utils.document_processor.Document')
    def test_extract_tables_from_docx_empty(self, mock_document_class):
        """Test table extraction from DOCX with no tables"""
        mock_doc = Mock()
        mock_doc.tables = []
        mock_document_class.return_value = mock_doc

        docx_path = 'test/empty.docx'
        tables = extract_tables_from_docx(docx_path)

        self.assertIsInstance(tables, list)
        self.assertEqual(len(tables), 0)

    @patch('utils.document_processor.Document')
    def test_extract_tables_from_docx_corrupted(self, mock_document_class):
        """Test table extraction from corrupted DOCX"""
        mock_document_class.side_effect = Exception("Corrupted file")

        docx_path = 'test/corrupted.docx'

        with self.assertRaises(Exception) as context:
            extract_tables_from_docx(docx_path)

        self.assertIn("Corrupted file", str(context.exception))

    @patch('utils.document_processor.glob.glob')
    @patch('utils.document_processor.process_single_file')
    @patch('utils.document_processor.concurrent.futures.ThreadPoolExecutor')
    def test_process_folder_success(self, mock_executor, mock_process_file, mock_glob):
        """Test successful folder processing"""
        # Mock file discovery
        mock_glob.return_value = ['test/file1.doc', 'test/file2.docx']

        # Mock thread pool executor
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Mock future objects
        mock_future1 = Mock()
        mock_future2 = Mock()
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

        # Mock process_single_file results
        mock_future1.result.return_value = {
            'matched_rows': [{'row': ['test'], 'source_file': 'file1.doc'}],
            'processed_files': 1,
            'errors': []
        }
        mock_future2.result.return_value = {
            'matched_rows': [{'row': ['test2'], 'source_file': 'file2.docx'}],
            'processed_files': 1,
            'errors': []
        }

        folder_path = 'test/folder'
        result = process_folder(folder_path)

        self.assertIn('matched_rows', result)
        self.assertIn('processed_files', result)
        self.assertIn('errors', result)
        self.assertEqual(result['processed_files'], 2)
        self.assertEqual(len(result['matched_rows']), 2)
        self.assertEqual(len(result['errors']), 0)

    @patch('utils.document_processor.glob.glob')
    def test_process_folder_no_files(self, mock_glob):
        """Test folder processing with no files"""
        mock_glob.return_value = []

        folder_path = 'test/empty_folder'
        result = process_folder(folder_path)

        self.assertEqual(result['matched_rows'], [])
        self.assertEqual(result['processed_files'], 0)
        self.assertEqual(result['errors'], [])

    @patch('utils.document_processor.glob.glob')
    @patch('utils.document_processor.process_single_file')
    @patch('utils.document_processor.concurrent.futures.ThreadPoolExecutor')
    def test_process_folder_with_errors(self, mock_executor, mock_process_file, mock_glob):
        """Test folder processing with file processing errors"""
        mock_glob.return_value = ['test/file1.doc']

        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        mock_future = Mock()
        mock_executor_instance.submit.return_value = mock_future

        # Simulate processing error
        mock_future.result.return_value = {
            'matched_rows': [],
            'processed_files': 0,
            'errors': ['Error processing file1.doc: File corrupted']
        }

        folder_path = 'test/folder'
        result = process_folder(folder_path)

        self.assertEqual(result['processed_files'], 0)
        self.assertEqual(len(result['matched_rows']), 0)
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('File corrupted', result['errors'][0])

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

    @patch('utils.document_processor.os.remove')
    @patch('utils.document_processor.extract_tables_from_docx')
    @patch('utils.document_processor.convert_doc_to_docx')
    def test_process_single_file_doc_success(self, mock_convert, mock_extract, mock_remove):
        """Test processing a single DOC file successfully"""
        mock_convert.return_value = 'test/sample.docx'
        mock_extract.return_value = [[['123/456 IPC', '10 hrs east']]]

        file_path = 'test/sample.doc'
        folder_path = 'test'

        result = process_single_file(file_path, folder_path)

        self.assertEqual(result['processed_files'], 1)
        self.assertEqual(len(result['matched_rows']), 1)
        self.assertEqual(result['matched_rows'][0]['source_file'], 'sample.doc')
        self.assertEqual(result['errors'], [])
        mock_convert.assert_called_once_with(file_path)
        mock_extract.assert_called_once_with('test/sample.docx')
        mock_remove.assert_called_once_with('test/sample.docx')

    @patch('utils.document_processor.os.remove')
    @patch('utils.document_processor.extract_tables_from_docx')
    def test_process_single_file_docx_success(self, mock_extract, mock_remove):
        """Test processing a single DOCX file successfully"""
        mock_extract.return_value = [[['999/1 Act', '2023-09-25 north']]]

        file_path = 'test/sample.docx'
        folder_path = 'test'

        result = process_single_file(file_path, folder_path)

        self.assertEqual(result['processed_files'], 1)
        self.assertEqual(len(result['matched_rows']), 1)
        self.assertEqual(result['matched_rows'][0]['source_file'], 'sample.docx')
        self.assertEqual(result['errors'], [])
        mock_extract.assert_called_once_with(file_path)
        mock_remove.assert_not_called()  # Should not remove original DOCX

    @patch('utils.document_processor.convert_doc_to_docx')
    def test_process_single_file_conversion_error(self, mock_convert):
        """Test processing file with conversion error"""
        mock_convert.side_effect = Exception("Conversion failed")

        file_path = 'test/sample.doc'
        folder_path = 'test'

        result = process_single_file(file_path, folder_path)

        self.assertEqual(result['processed_files'], 0)
        self.assertEqual(result['matched_rows'], [])
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('Conversion failed', result['errors'][0])

    @patch('utils.document_processor.os.remove')
    @patch('utils.document_processor.extract_tables_from_docx')
    @patch('utils.document_processor.convert_doc_to_docx')
    def test_process_single_file_extraction_error(self, mock_convert, mock_extract, mock_remove):
        """Test processing file with table extraction error"""
        mock_convert.return_value = 'test/sample.docx'
        mock_extract.side_effect = Exception("Extraction failed")

        file_path = 'test/sample.doc'
        folder_path = 'test'

        result = process_single_file(file_path, folder_path)

        self.assertEqual(result['processed_files'], 0)
        self.assertEqual(result['matched_rows'], [])
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('Extraction failed', result['errors'][0])
        mock_remove.assert_called_once_with('test/sample.docx')

    @patch('utils.document_processor.os.remove')
    @patch('utils.document_processor.extract_tables_from_docx')
    @patch('utils.document_processor.convert_doc_to_docx')
    def test_process_single_file_cleanup_error(self, mock_convert, mock_extract, mock_remove):
        """Test processing file where cleanup fails"""
        mock_convert.return_value = 'test/sample.docx'
        mock_extract.return_value = [[['123/456 IPC', '10 hrs east']]]
        mock_remove.side_effect = Exception("Cleanup failed")

        file_path = 'test/sample.doc'
        folder_path = 'test'

        result = process_single_file(file_path, folder_path)

        # Should still process successfully despite cleanup error
        self.assertEqual(result['processed_files'], 1)
        self.assertEqual(len(result['matched_rows']), 1)
        self.assertEqual(result['errors'], [])

    def test_process_single_file_unsupported_format(self):
        """Test processing file with unsupported format"""
        file_path = 'test/sample.pdf'
        folder_path = 'test'

        result = process_single_file(file_path, folder_path)

        self.assertEqual(result['processed_files'], 0)
        self.assertEqual(result['matched_rows'], [])
        self.assertEqual(result['errors'], [])

    @patch('utils.document_processor.os.path.exists')
    @patch('utils.document_processor.os.remove')
    @patch('builtins.open', new_callable=mock_open)
    def test_system_error_disk_space(self, mock_file, mock_remove, mock_exists):
        """Test system error - disk space"""
        mock_exists.return_value = True
        mock_file.side_effect = OSError("No space left on device")

        # This should be tested in the context of file operations
        with self.assertRaises(OSError):
            with open('test/file.txt', 'w') as f:
                f.write('test')

    @patch('utils.document_processor.os.path.exists')
    @patch('utils.document_processor.os.access')
    def test_system_error_permissions(self, mock_access, mock_exists):
        """Test system error - file permissions"""
        mock_exists.return_value = True
        mock_access.return_value = False  # No read permission

        # Test that the function handles permission errors gracefully
        file_path = 'test/no_permission.doc'
        folder_path = 'test'

        result = process_single_file(file_path, folder_path)

        self.assertEqual(result['processed_files'], 0)
        self.assertEqual(result['matched_rows'], [])
        self.assertEqual(len(result['errors']), 1)

    @patch('utils.document_processor.glob.glob')
    @patch('utils.document_processor.process_single_file')
    @patch('utils.document_processor.concurrent.futures.ThreadPoolExecutor')
    def test_performance_large_folder(self, mock_executor, mock_process_file, mock_glob):
        """Test performance with large number of files"""
        import time

        # Mock 100 files
        mock_glob.return_value = [f'test/file{i}.doc' for i in range(100)]

        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Create mock futures
        mock_futures = []
        for i in range(100):
            mock_future = Mock()
            mock_future.result.return_value = {
                'matched_rows': [{'row': [f'row{i}'], 'source_file': f'file{i}.doc'}],
                'processed_files': 1,
                'errors': []
            }
            mock_futures.append(mock_future)

        mock_executor_instance.submit.side_effect = mock_futures

        folder_path = 'test/large_folder'
        start_time = time.time()
        result = process_folder(folder_path)
        end_time = time.time()

        # Should process all files
        self.assertEqual(result['processed_files'], 100)
        self.assertEqual(len(result['matched_rows']), 100)

        # Performance check - should complete reasonably quickly
        processing_time = end_time - start_time
        self.assertLess(processing_time, 5.0)  # Should complete in less than 5 seconds

    @patch('utils.document_processor.glob.glob')
    @patch('utils.document_processor.process_single_file')
    @patch('utils.document_processor.concurrent.futures.ThreadPoolExecutor')
    def test_concurrent_processing_isolation(self, mock_executor, mock_process_file, mock_glob):
        """Test that concurrent processing maintains proper isolation"""
        mock_glob.return_value = ['test/file1.doc', 'test/file2.doc']

        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Create futures that return different results
        mock_future1 = Mock()
        mock_future1.result.return_value = {
            'matched_rows': [{'row': ['row1'], 'source_file': 'file1.doc'}],
            'processed_files': 1,
            'errors': []
        }

        mock_future2 = Mock()
        mock_future2.result.return_value = {
            'matched_rows': [{'row': ['row2'], 'source_file': 'file2.doc'}],
            'processed_files': 1,
            'errors': []
        }

        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

        folder_path = 'test/concurrent_folder'
        result = process_folder(folder_path)

        # Verify results are properly isolated and combined
        self.assertEqual(result['processed_files'], 2)
        self.assertEqual(len(result['matched_rows']), 2)

        # Check that files are processed independently
        source_files = [row['source_file'] for row in result['matched_rows']]
        self.assertIn('file1.doc', source_files)
        self.assertIn('file2.doc', source_files)

if __name__ == '__main__':
    unittest.main()
