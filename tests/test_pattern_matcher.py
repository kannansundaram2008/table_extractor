import unittest
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

if __name__ == '__main__':
    unittest.main()
