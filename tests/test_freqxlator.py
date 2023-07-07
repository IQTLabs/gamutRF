#!/usr/bin/python3
import unittest

from gamutrf.freqxlator import FreqXLator
from gamutrf.freqxlator import argument_parser


class FreqXlatorTestCase(unittest.TestCase):
    def test_freqxlator_smoke(self):
        FreqXLator(1e3, 100e3, 10e3, 10, "/dev/null", "/dev/null")

    def test_argument_parser(self):
        argument_parser()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
