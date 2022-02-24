#!/usr/bin/python3

import unittest
from gamutrf.freqxlator import FreqXLator


class FreqXlatorTestCase(unittest.TestCase):

    def test_freqxlator_smoke(self):
        FreqXLator(1e3, 100e3, 10e3, 10, '/dev/null', '/dev/null')


if __name__ == '__main__':
    unittest.main()
