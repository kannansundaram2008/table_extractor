import unittest
import os
import io
import time
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from app import app

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_file_upload_and_processing(self):
        # Prepare sample file for upload
        sample_path = 'tests/test_data/sample.docx'
        if not os.path.exists(sample_path):
            self.skipTest("Sample .docx file not found for end-to-end test")

        with open(sample_path, 'rb') as f:
            data = {
                'files': (io.BytesIO(f.read()), 'sample.docx')
            }
            response = self.client.post('/', data=data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Processing', response.data)

        # Check progress endpoint
        response = self.client.get('/progress')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'status', response.data)

        # Since processing is in background thread, wait a bit
        import time
        time.sleep(5)

        # Check results endpoint
        response = self.client.get('/results')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Extraction Results', response.data or b'')

        # Check export CSV endpoint (only if data exists)
        response = self.client.get('/export_csv')
        if b'No data to export' not in response.data:
            self.assertEqual(response.status_code, 200)
            self.assertIn('text/csv', response.content_type)
        else:
            self.assertEqual(response.status_code, 200)  # Still returns 200 with message

        # Check export Excel endpoint (only if data exists)
        response = self.client.get('/export_excel')
        if b'No data to export' not in response.data:
            self.assertEqual(response.status_code, 200)
            self.assertIn('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', response.content_type)
        else:
            self.assertEqual(response.status_code, 200)

        # Check export PDF endpoint (only if data exists)
        response = self.client.get('/export_pdf')
        if b'No data to export' not in response.data:
            self.assertEqual(response.status_code, 200)
            self.assertIn('application/pdf', response.content_type)
        else:
            self.assertEqual(response.status_code, 200)

    @patch('app.process_folder')
    def test_file_upload_processing_error(self, mock_process):
        """Test file upload with processing error"""
        mock_process.side_effect = Exception("Processing failed")

        # Create a mock file
        mock_file = io.BytesIO(b"mock file content")
        mock_file.filename = "test.doc"

        data = {
            'files': (mock_file, 'test.doc')
        }

        response = self.client.post('/', data=data, content_type='multipart/form-data')

        # Should handle error gracefully
        self.assertEqual(response.status_code, 200)
        # Should show error message or redirect appropriately

    @patch('app.process_folder')
    def test_file_upload_empty_file(self, mock_process):
        """Test file upload with empty file"""
        mock_file = io.BytesIO(b"")
        mock_file.filename = "empty.doc"

        data = {
            'files': (mock_file, 'empty.doc')
        }

        response = self.client.post('/', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)

    def test_file_upload_no_file_selected(self):
        """Test file upload when no file is selected"""
        data = {
            'files': ('', '')
        }

        response = self.client.post('/', data=data, content_type='multipart/form-data')

        # Should handle gracefully
        self.assertEqual(response.status_code, 200)

    @patch('app.process_folder')
    def test_file_upload_unsupported_format(self, mock_process):
        """Test file upload with unsupported file format"""
        mock_file = io.BytesIO(b"mock content")
        mock_file.filename = "test.txt"

        data = {
            'files': (mock_file, 'test.txt')
        }

        response = self.client.post('/', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)

    @patch('app.process_folder')
    def test_concurrent_file_uploads(self, mock_process):
        """Test concurrent file upload handling"""
        import threading
        import queue

        mock_process.return_value = {
            'matched_rows': [{'row': ['test'], 'source_file': 'test.doc'}],
            'processed_files': 1,
            'errors': []
        }

        results = queue.Queue()
        errors = queue.Queue()

        def upload_file(file_id):
            try:
                mock_file = io.BytesIO(f"content {file_id}".encode())
                mock_file.filename = f"test{file_id}.doc"

                data = {
                    'files': (mock_file, f'test{file_id}.doc')
                }

                response = self.client.post('/', data=data, content_type='multipart/form-data')
                results.put((file_id, response.status_code))
            except Exception as e:
                errors.put((file_id, str(e)))

        # Start concurrent uploads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=upload_file, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        self.assertTrue(errors.empty(), f"Errors in concurrent uploads: {list(errors.queue)}")
        self.assertEqual(results.qsize(), 10)

        # All uploads should succeed
        for _ in range(10):
            file_id, status_code = results.get()
            self.assertEqual(status_code, 200)

    def test_large_file_upload(self):
        """Test uploading a large file"""
        # Create a large file (1MB)
        large_content = b"A" * (1024 * 1024)  # 1MB of data
        mock_file = io.BytesIO(large_content)
        mock_file.filename = "large_file.doc"

        data = {
            'files': (mock_file, 'large_file.doc')
        }

        start_time = time.time()
        response = self.client.post('/', data=data, content_type='multipart/form-data')
        end_time = time.time()

        # Should handle large files within reasonable time
        self.assertEqual(response.status_code, 200)
        processing_time = end_time - start_time
        self.assertLess(processing_time, 10.0)  # Should complete in less than 10 seconds

    @patch('app.os.path.exists')
    @patch('app.open', new_callable=mock_open)
    def test_export_csv_permission_error(self, mock_file, mock_exists):
        """Test CSV export with permission error"""
        mock_exists.return_value = True
        mock_file.side_effect = PermissionError("Permission denied")

        # First need to have some data
        with patch('app.matched_rows', [{'row': ['test'], 'source_file': 'test.doc'}]):
            response = self.client.get('/export_csv')

        # Should handle permission error gracefully
        self.assertEqual(response.status_code, 200)

    @patch('app.os.path.exists')
    @patch('app.open', new_callable=mock_open)
    def test_export_excel_disk_space_error(self, mock_file, mock_exists):
        """Test Excel export with disk space error"""
        mock_exists.return_value = True
        mock_file.side_effect = OSError("No space left on device")

        # First need to have some data
        with patch('app.matched_rows', [{'row': ['test'], 'source_file': 'test.doc'}]):
            response = self.client.get('/export_excel')

        # Should handle disk space error gracefully
        self.assertEqual(response.status_code, 200)

    def test_malformed_file_upload(self):
        """Test uploading a malformed file"""
        # Create a file with invalid content
        malformed_content = b"This is not a valid document format"
        mock_file = io.BytesIO(malformed_content)
        mock_file.filename = "malformed.doc"

        data = {
            'files': (mock_file, 'malformed.doc')
        }

        response = self.client.post('/', data=data, content_type='multipart/form-data')

        # Should handle malformed files gracefully
        self.assertEqual(response.status_code, 200)

    def test_unicode_filename_upload(self):
        """Test uploading file with Unicode filename"""
        unicode_filename = "测试文档.doc"  # Chinese characters
        mock_file = io.BytesIO(b"test content")
        mock_file.filename = unicode_filename

        data = {
            'files': (mock_file, unicode_filename)
        }

        response = self.client.post('/', data=data, content_type='multipart/form-data')

        # Should handle Unicode filenames
        self.assertEqual(response.status_code, 200)

    def test_very_long_filename_upload(self):
        """Test uploading file with very long filename"""
        long_filename = "A" * 255 + ".doc"  # Very long filename
        mock_file = io.BytesIO(b"test content")
        mock_file.filename = long_filename

        data = {
            'files': (mock_file, long_filename)
        }

        response = self.client.post('/', data=data, content_type='multipart/form-data')

        # Should handle long filenames
        self.assertEqual(response.status_code, 200)

    def test_special_characters_filename_upload(self):
        """Test uploading file with special characters in filename"""
        special_filename = "test@#$%^&()_file.doc"
        mock_file = io.BytesIO(b"test content")
        mock_file.filename = special_filename

        data = {
            'files': (mock_file, special_filename)
        }

        response = self.client.post('/', data=data, content_type='multipart/form-data')

        # Should handle special characters in filename
        self.assertEqual(response.status_code, 200)

    @patch('app.process_folder')
    def test_memory_usage_with_multiple_files(self, mock_process):
        """Test memory usage when processing multiple files"""
        import gc

        # Mock processing of multiple files
        mock_process.return_value = {
            'matched_rows': [{'row': ['test'], 'source_file': f'file{i}.doc'} for i in range(100)],
            'processed_files': 100,
            'errors': []
        }

        initial_objects = len(gc.get_objects())

        # Simulate multiple file uploads
        for i in range(50):
            mock_file = io.BytesIO(f"content {i}".encode())
            mock_file.filename = f"test{i}.doc"

            data = {
                'files': (mock_file, f'test{i}.doc')
            }

            response = self.client.post('/', data=data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory usage should not grow excessively
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 1000)  # Reasonable memory growth

    def test_session_handling(self):
        """Test session handling during file processing"""
        with self.client:
            # Test multiple requests in same session
            mock_file1 = io.BytesIO(b"content1")
            mock_file1.filename = "test1.doc"

            data1 = {
                'files': (mock_file1, 'test1.doc')
            }

            response1 = self.client.post('/', data=data1, content_type='multipart/form-data')
            self.assertEqual(response1.status_code, 200)

            # Second upload in same session
            mock_file2 = io.BytesIO(b"content2")
            mock_file2.filename = "test2.doc"

            data2 = {
                'files': (mock_file2, 'test2.doc')
            }

            response2 = self.client.post('/', data=data2, content_type='multipart/form-data')
            self.assertEqual(response2.status_code, 200)

    def test_progress_endpoint_error_handling(self):
        """Test progress endpoint error handling"""
        # Test progress endpoint without active processing
        response = self.client.get('/progress')

        # Should handle gracefully
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'status', response.data)

    def test_results_endpoint_error_handling(self):
        """Test results endpoint error handling"""
        # Test results endpoint without data
        response = self.client.get('/results')

        # Should handle gracefully
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
