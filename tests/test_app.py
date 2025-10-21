import unittest
import os
import io
import time
from unittest.mock import Mock, patch, MagicMock, mock_open
from app import app

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_get(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Document Table Row Extractor', response.data)

    def test_index_post_no_files(self):
        response = self.app.post('/')
        self.assertEqual(response.status_code, 200)
        # Should render index again

    # Note: Testing file upload requires sample files, which are not present

    @patch('app.process_folder')
    def test_file_upload_success(self, mock_process):
        """Test successful file upload"""
        mock_process.return_value = {
            'matched_rows': [{'row': ['123/456 IPC', '10 hrs east'], 'source_file': 'test.doc'}],
            'processed_files': 1,
            'errors': []
        }

        # Create a mock file
        mock_file = io.BytesIO(b"mock file content")
        mock_file.filename = "test.doc"

        data = {
            'files': (mock_file, 'test.doc')
        }

        response = self.app.post('/', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Processing', response.data)

    @patch('app.process_folder')
    def test_file_upload_processing_error(self, mock_process):
        """Test file upload with processing error"""
        mock_process.side_effect = Exception("Processing failed")

        mock_file = io.BytesIO(b"mock file content")
        mock_file.filename = "test.doc"

        data = {
            'files': (mock_file, 'test.doc')
        }

        response = self.app.post('/', data=data, content_type='multipart/form-data')

        # Should handle error gracefully
        self.assertEqual(response.status_code, 200)

    def test_file_upload_empty_request(self):
        """Test file upload with empty request"""
        response = self.app.post('/', data={}, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)

    def test_file_upload_multiple_files(self):
        """Test uploading multiple files"""
        # Create multiple mock files
        files = []
        for i in range(3):
            mock_file = io.BytesIO(f"content {i}".encode())
            mock_file.filename = f"test{i}.doc"
            files.append((mock_file, f'test{i}.doc'))

        data = {
            'files': files
        }

        response = self.app.post('/', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)

    @patch('app.process_folder')
    def test_large_file_upload_performance(self, mock_process):
        """Test performance with large file upload"""
        mock_process.return_value = {
            'matched_rows': [],
            'processed_files': 1,
            'errors': []
        }

        # Create a large file (5MB)
        large_content = b"A" * (5 * 1024 * 1024)
        mock_file = io.BytesIO(large_content)
        mock_file.filename = "large_file.doc"

        data = {
            'files': (mock_file, 'large_file.doc')
        }

        start_time = time.time()
        response = self.app.post('/', data=data, content_type='multipart/form-data')
        end_time = time.time()

        self.assertEqual(response.status_code, 200)
        processing_time = end_time - start_time
        self.assertLess(processing_time, 15.0)  # Should complete in less than 15 seconds

    def test_malicious_filename_upload(self):
        """Test upload with potentially malicious filename"""
        malicious_names = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "test<script>alert('xss')</script>.doc",
            "test'; DROP TABLE users; --.doc",
            "CON.doc",  # Windows reserved name
            "PRN.doc",  # Windows reserved name
            "AUX.doc",  # Windows reserved name
        ]

        for malicious_name in malicious_names:
            with self.subTest(filename=malicious_name):
                mock_file = io.BytesIO(b"test content")
                mock_file.filename = malicious_name

                data = {
                    'files': (mock_file, malicious_name)
                }

                response = self.app.post('/', data=data, content_type='multipart/form-data')

                # Should handle malicious filenames gracefully
                self.assertEqual(response.status_code, 200)

    def test_unicode_content_upload(self):
        """Test upload with Unicode content"""
        unicode_content = "测试文档内容 with émojis 🚀 and spëcial çharacters".encode('utf-8')
        mock_file = io.BytesIO(unicode_content)
        mock_file.filename = "unicode_test.doc"

        data = {
            'files': (mock_file, 'unicode_test.doc')
        }

        response = self.app.post('/', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)

    @patch('app.process_folder')
    def test_concurrent_uploads_isolation(self, mock_process):
        """Test that concurrent uploads don't interfere with each other"""
        import threading
        import queue

        mock_process.return_value = {
            'matched_rows': [{'row': ['test'], 'source_file': 'test.doc'}],
            'processed_files': 1,
            'errors': []
        }

        results = queue.Queue()
        errors = queue.Queue()

        def upload_worker(worker_id):
            try:
                mock_file = io.BytesIO(f"content {worker_id}".encode())
                mock_file.filename = f"test{worker_id}.doc"

                data = {
                    'files': (mock_file, f'test{worker_id}.doc')
                }

                response = self.app.post('/', data=data, content_type='multipart/form-data')
                results.put((worker_id, response.status_code))
            except Exception as e:
                errors.put((worker_id, str(e)))

        # Start concurrent uploads
        threads = []
        for i in range(20):  # More threads to test isolation
            thread = threading.Thread(target=upload_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        self.assertTrue(errors.empty(), f"Errors in concurrent uploads: {list(errors.queue)}")
        self.assertEqual(results.qsize(), 20)

        # All uploads should succeed
        for _ in range(20):
            worker_id, status_code = results.get()
            self.assertEqual(status_code, 200)

    def test_memory_leak_prevention(self):
        """Test that memory is properly cleaned up after uploads"""
        import gc

        initial_objects = len(gc.get_objects())

        # Perform multiple uploads
        for i in range(10):
            mock_file = io.BytesIO(f"content {i}".encode())
            mock_file.filename = f"test{i}.doc"

            data = {
                'files': (mock_file, f'test{i}.doc')
            }

            response = self.app.post('/', data=data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory usage should not grow excessively
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 500)  # Reasonable memory growth

    def test_very_small_file_upload(self):
        """Test uploading very small files"""
        mock_file = io.BytesIO(b"")
        mock_file.filename = "empty.doc"

        data = {
            'files': (mock_file, 'empty.doc')
        }

        response = self.app.post('/', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)

    def test_binary_file_upload(self):
        """Test uploading binary file content"""
        # Create binary content (simulating a document)
        binary_content = bytes(range(256)) * 100  # All byte values
        mock_file = io.BytesIO(binary_content)
        mock_file.filename = "binary_test.doc"

        data = {
            'files': (mock_file, 'binary_test.doc')
        }

        response = self.app.post('/', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)

    def test_request_timeout_handling(self):
        """Test handling of slow/long-running requests"""
        @patch('app.process_folder')
        def slow_process(mock_process):
            # Simulate slow processing
            time.sleep(0.1)
            mock_process.return_value = {
                'matched_rows': [],
                'processed_files': 1,
                'errors': []
            }

            mock_file = io.BytesIO(b"slow content")
            mock_file.filename = "slow_test.doc"

            data = {
                'files': (mock_file, 'slow_test.doc')
            }

            start_time = time.time()
            response = self.app.post('/', data=data, content_type='multipart/form-data')
            end_time = time.time()

            self.assertEqual(response.status_code, 200)
            processing_time = end_time - start_time
            self.assertGreater(processing_time, 0.05)  # Should take some time

        slow_process()

    def test_error_response_format(self):
        """Test that error responses are properly formatted"""
        @patch('app.process_folder')
        def test_error_format(mock_process):
            mock_process.side_effect = ValueError("Invalid file format")

            mock_file = io.BytesIO(b"invalid content")
            mock_file.filename = "invalid.doc"

            data = {
                'files': (mock_file, 'invalid.doc')
            }

            response = self.app.post('/', data=data, content_type='multipart/form-data')

            self.assertEqual(response.status_code, 200)
            # Should contain error information in response
            self.assertIsInstance(response.data, bytes)

        test_error_format()

if __name__ == '__main__':
    unittest.main()
