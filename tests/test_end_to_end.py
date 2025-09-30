import unittest
import os
import io
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

        # Check export CSV endpoint
        response = self.client.get('/export_csv')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/csv', response.content_type)

        # Check export Excel endpoint
        response = self.client.get('/export_excel')
        self.assertEqual(response.status_code, 200)
        self.assertIn('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', response.content_type)

        # Check export PDF endpoint
        response = self.client.get('/export_pdf')
        self.assertEqual(response.status_code, 200)
        self.assertIn('application/pdf', response.content_type)

if __name__ == '__main__':
    unittest.main()
