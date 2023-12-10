#!/usr/bin/python3
import unittest

import numpy as np

from gamutrf.utils import endianstr
from gamutrf.sample_reader import parse_filename


class UtilsTestCase(unittest.TestCase):
    def test_parse_filename(self):
        self.assertEqual(
            {
                "filename": "test.raw",
                "center_frequency": None,
                "sample_rate": None,
                "sample_dtype": None,
                "sample_len": None,
                "sample_type": None,
                "sample_bits": None,
                "nfft": None,
                "timestamp": None,
            },
            parse_filename("test.raw"),
        )
        self.assertEqual(
            {
                "timestamp": 1645570069,
                "filename": "gamutrf_recording1645570069_760000000Hz_20000000sps.raw",
                "nfft": None,
                "center_frequency": 760000000,
                "sample_rate": 20000000,
                "sample_dtype": np.dtype([("i", "<f4"), ("q", "<f4")]),
                "sample_len": 8,
                "sample_type": "float",
                "sample_bits": 32,
            },
            parse_filename("gamutrf_recording1645570069_760000000Hz_20000000sps.raw"),
        )
        self.assertEqual(
            {
                "filename": "gamutrf_recording1645540092_140000000Hz_20000000sps.ci16_%s.gz"
                % endianstr(),
                "nfft": None,
                "timestamp": 1645540092,
                "center_frequency": 140000000,
                "sample_rate": 20000000,
                "sample_dtype": np.dtype([("i", "<i2"), ("q", "<i2")]),
                "sample_len": 4,
                "sample_type": "signed-integer",
                "sample_bits": 16,
            },
            parse_filename(
                "gamutrf_recording1645540092_140000000Hz_20000000sps.ci16_%s.gz"
                % endianstr()
            ),
        )
        self.assertEqual(
            {
                "filename": "fft_gamutrf_recording1645540092_140000000Hz_1024points_20000000sps.raw.gz",
                "nfft": 1024,
                "timestamp": 1645540092,
                "center_frequency": 140000000,
                "sample_rate": 20000000,
                "sample_dtype": np.dtype([("i", "<f4"), ("q", "<f4")]),
                "sample_len": 8,
                "sample_type": "float",
                "sample_bits": 32,
            },
            parse_filename(
                "fft_gamutrf_recording1645540092_140000000Hz_1024points_20000000sps.raw.gz"
            ),
        )
        self.assertEqual(
            {
                "filename": "fft_gamutrf_recording1645540092_140000000Hz_20000000sps.s16.gz",
                "timestamp": 1645540092,
                "nfft": None,
                "center_frequency": 140000000,
                "sample_rate": 20000000,
                "sample_dtype": np.dtype([("i", "<f4"), ("q", "<f4")]),
                "sample_len": 8,
                "sample_type": "signed-integer",
                "sample_bits": 32,
            },
            parse_filename(
                "fft_gamutrf_recording1645540092_140000000Hz_20000000sps.s16.gz"
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
