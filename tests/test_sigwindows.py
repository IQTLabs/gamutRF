#!/usr/bin/python3
import os
import unittest

import pandas as pd

from gamutrf.sigwindows import find_sig_windows

TESTDIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'data')


class WindowsTestCase(unittest.TestCase):

    def test_find_wifi(self):
        df = pd.read_csv(os.path.join(TESTDIR, 'wifi24.csv'),
                         delim_whitespace=True)
        self.assertEqual(
            [(2420.422912, 2437.485056, -32.40808890037803, -44.99829069187393)], find_sig_windows(df))


if __name__ == '__main__':
    unittest.main()
