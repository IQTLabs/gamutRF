#!/usr/bin/python3

import unittest
import numpy as np
from gamutrf.utils import parse_filename


class UtilsTestCase(unittest.TestCase):

    def test_parse_filename(self):
        self.assertEqual(
            (None, None, None, None, None, None),
            parse_filename('test.raw'))
        self.assertEqual(
            (760000000, 20000000, np.dtype([('i', '<f4'), ('q', '<f4')]), 8, 'float', 32),
            parse_filename('gamutrf_recording1645570069_760000000Hz_20000000sps.raw'))
        self.assertEqual(
            (140000000, 20000000, np.dtype([('i', '<i2'), ('q', '<i2')]), 4, 'signed-integer', 16),
            parse_filename('gamutrf_recording1645540092_140000000Hz_20000000sps.s16.gz'))
        self.assertEqual(
            (140000000, 20000000, np.dtype([('i', '<f4'), ('q', '<f4')]), 8, 'float', 32),
            parse_filename('fft_gamutrf_recording1645540092_140000000Hz_20000000sps.s16.gz'))



if __name__ == '__main__':
    unittest.main()
