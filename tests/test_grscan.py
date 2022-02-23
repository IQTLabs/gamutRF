#!/usr/bin/python3

import unittest
from gamutrf.grscan import grscan


class grscanTestCase(unittest.TestCase):

    def test_grscan_smoke(self):
        grscan(sdr=None)


if __name__ == '__main__':
    unittest.main()
