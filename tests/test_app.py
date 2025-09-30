import unittest
import os
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

if __name__ == '__main__':
    unittest.main()
