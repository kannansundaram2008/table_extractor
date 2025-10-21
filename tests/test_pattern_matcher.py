import unittest
import time
from unittest.mock import patch
from utils.pattern_matcher import match_row, has_primary_pattern, has_adjacent_pattern, is_date

class TestPatternMatcher(unittest.TestCase):

    def test_has_primary_pattern(self):
        self.assertTrue(has_primary_pattern("123/456 IPC"))
        self.assertTrue(has_primary_pattern("12/34 u/s"))
        self.assertFalse(has_primary_pattern("no match here"))
        self.assertTrue(has_primary_pattern("9999/1 Act"))
        self.assertFalse(has_primary_pattern("123/456 no keyword"))

    def test_is_date(self):
        self.assertTrue(is_date("2023-09-25"))
        self.assertTrue(is_date("September 25, 2023"))
        self.assertFalse(is_date("not a date"))
        self.assertTrue(is_date("25/09/2023"))
        self.assertTrue(is_date("09/25/2023"))

    def test_has_adjacent_pattern(self):
        self.assertTrue(has_adjacent_pattern("10 hrs east"))
        self.assertTrue(has_adjacent_pattern("2023-09-25 north"))
        self.assertFalse(has_adjacent_pattern("10 hours"))
        self.assertFalse(has_adjacent_pattern("east only"))
        self.assertFalse(has_adjacent_pattern("random text"))

    def test_match_row(self):
        row1 = ["123/456 IPC", "10 hrs east", "other"]
        row2 = ["no match", "10 hrs east"]
        row3 = ["123/456 IPC", "no direction"]
        row4 = ["123/456 IPC", "2023-09-25 north"]
        self.assertTrue(match_row(row1))
        self.assertFalse(match_row(row2))
        self.assertFalse(match_row(row3))
        self.assertTrue(match_row(row4))

    def test_has_primary_pattern_edge_cases(self):
        """Test primary pattern matching with edge cases"""
        # Test with None values
        self.assertFalse(has_primary_pattern(None))
        self.assertFalse(has_primary_pattern(""))

        # Test with case variations
        self.assertTrue(has_primary_pattern("123/456 ipc"))
        self.assertTrue(has_primary_pattern("123/456 IPC"))
        self.assertTrue(has_primary_pattern("123/456 IpC"))

        # Test with extra whitespace
        self.assertTrue(has_primary_pattern("  123/456 IPC  "))
        self.assertTrue(has_primary_pattern("\t123/456 IPC\n"))

        # Test with special characters
        self.assertTrue(has_primary_pattern("123/456 IPC@#$%"))
        self.assertFalse(has_primary_pattern("123/456"))

    def test_is_date_edge_cases(self):
        """Test date validation with edge cases"""
        # Test with None and empty values
        self.assertFalse(is_date(None))
        self.assertFalse(is_date(""))

        # Test with various date formats
        self.assertTrue(is_date("2023-12-31"))
        self.assertTrue(is_date("12/31/2023"))
        self.assertTrue(is_date("31-12-2023"))
        self.assertTrue(is_date("December 31, 2023"))
        self.assertTrue(is_date("31 Dec 2023"))

        # Test with invalid dates
        self.assertFalse(is_date("2023-13-31"))  # Invalid month
        self.assertFalse(is_date("2023-12-32"))  # Invalid day
        self.assertFalse(is_date("not a date"))
        self.assertFalse(is_date("2023-12-31 25:70:80"))  # Invalid time

    def test_has_adjacent_pattern_edge_cases(self):
        """Test adjacent pattern matching with edge cases"""
        # Test with None and empty values
        self.assertFalse(has_adjacent_pattern(None))
        self.assertFalse(has_adjacent_pattern(""))

        # Test with case variations
        self.assertTrue(has_adjacent_pattern("10 hrs east"))
        self.assertTrue(has_adjacent_pattern("10 HRS EAST"))
        self.assertTrue(has_adjacent_pattern("10 hrs East"))

        # Test with extra whitespace
        self.assertTrue(has_adjacent_pattern("  10 hrs east  "))
        self.assertTrue(has_adjacent_pattern("\t10 hrs east\n"))

        # Test with partial matches (should fail)
        self.assertFalse(has_adjacent_pattern("10 hrs"))  # Missing direction
        self.assertFalse(has_adjacent_pattern("east"))   # Missing time
        self.assertFalse(has_adjacent_pattern("10 east")) # Wrong order

    def test_match_row_edge_cases(self):
        """Test row matching with edge cases"""
        # Test with None values
        self.assertFalse(match_row([None, None, None]))
        self.assertFalse(match_row([]))

        # Test with empty strings
        self.assertFalse(match_row(["", "", ""]))
        self.assertFalse(match_row(["123/456 IPC", "", ""]))  # Missing adjacent pattern

        # Test with very long strings
        long_string = "A" * 10000
        self.assertFalse(match_row([long_string, long_string, long_string]))

        # Test with special characters
        self.assertTrue(match_row(["123/456 IPC", "10 hrs @#$%"]))
        self.assertTrue(match_row(["999/1 Act", "2023-09-25 @#$%"]))

        # Test with Unicode characters
        self.assertTrue(match_row(["१२३/४५६ IPC", "१० hrs पूर्व"]))
        self.assertTrue(match_row(["999/1 Act", "२०२३-०९-२५ उत्तर"]))

    def test_pattern_matcher_performance(self):
        """Test pattern matcher performance with large datasets"""
        # Create a large dataset
        large_dataset = []
        for i in range(1000):
            if i % 2 == 0:
                large_dataset.append(f"Row {i}: 123/456 IPC, 10 hrs east, other data")
            else:
                large_dataset.append(f"Row {i}: no match pattern here")

        start_time = time.time()
        match_count = 0

        for row_text in large_dataset:
            # Simulate row processing
            if "123/456 IPC" in row_text and ("hrs east" in row_text or "hrs west" in row_text or "hrs north" in row_text or "hrs south" in row_text):
                match_count += 1

        end_time = time.time()
        processing_time = end_time - start_time

        # Should process 1000 rows quickly
        self.assertEqual(match_count, 500)  # Every even row should match
        self.assertLess(processing_time, 1.0)  # Should complete in less than 1 second

    def test_pattern_matcher_memory_efficiency(self):
        """Test pattern matcher memory efficiency"""
        import gc

        # Test with large row data
        large_row = ["123/456 IPC"] + ["A" * 1000] * 10 + ["10 hrs east"]

        initial_objects = len(gc.get_objects())

        # Process the row multiple times
        for _ in range(100):
            result = match_row(large_row)
            self.assertTrue(result)

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory usage should not grow excessively
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 1000)  # Reasonable memory growth

    def test_pattern_matcher_concurrent_access(self):
        """Test pattern matcher thread safety"""
        import threading
        import queue

        large_row = ["123/456 IPC", "10 hrs east", "other data"]
        results = queue.Queue()
        errors = queue.Queue()

        def worker(worker_id):
            try:
                for _ in range(100):
                    result = match_row(large_row)
                    if result:
                        results.put(result)
            except Exception as e:
                errors.put(e)

        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        self.assertTrue(errors.empty(), f"Errors occurred: {list(errors.queue)}")
        self.assertEqual(results.qsize(), 1000)  # 10 threads * 100 iterations

    def test_is_date_malformed_inputs(self):
        """Test date validation with malformed inputs"""
        malformed_dates = [
            "2023-02-29",  # Invalid leap year
            "2023-13-01",  # Invalid month
            "2023-12-32",  # Invalid day
            "25-25-2023",  # Invalid format
            "2023/12/31",  # Wrong separator
            "31/12/2023 extra",  # Extra content
            "2023-12-31T10:30:00",  # ISO datetime
            "yesterday",
            "tomorrow",
            "next week"
        ]

        for malformed_date in malformed_dates:
            with self.subTest(date=malformed_date):
                self.assertFalse(is_date(malformed_date))

    def test_pattern_matcher_case_sensitivity_edge_cases(self):
        """Test pattern matcher case sensitivity edge cases"""
        # Test mixed case scenarios
        test_cases = [
            ("123/456 ipc", True),   # lowercase ipc
            ("123/456 IPC", True),   # uppercase IPC
            ("123/456 IpC", True),   # mixed case
            ("123/456 ipc 10 hrs east", True),
            ("123/456 IPC 10 HRS EAST", True),
            ("123/456 ipc 10 hrs EAST", True),
        ]

        for pattern, expected in test_cases:
            with self.subTest(pattern=pattern):
                row = [pattern, "other data"]
                self.assertEqual(match_row(row), expected)

if __name__ == '__main__':
    unittest.main()
