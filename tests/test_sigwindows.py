#!/usr/bin/python3
import os
import unittest

from gamutrf.sigwindows import freq_excluded
from gamutrf.sigwindows import parse_freq_excluded


TESTDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


class WindowsTestCase(unittest.TestCase):
    def test_freq_excluded(self):
        self.assertTrue(freq_excluded(100, ((100, 200),)))
        self.assertFalse(freq_excluded(99, ((100, 200),)))
        self.assertTrue(freq_excluded(1e9, ((1e6, None),)))
        self.assertFalse(freq_excluded(1e6, ((1e9, None),)))
        self.assertFalse(freq_excluded(1e9, ((None, 1e6),)))

    def test_parse_excluded(self):
        self.assertEqual(
            ((100, 200), (200, None), (None, 100)),
            parse_freq_excluded(["100-200", "200-", "-100"]),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
