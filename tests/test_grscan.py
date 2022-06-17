#!/usr/bin/python3
import time
import unittest

from gamutrf.grscan import grscan


class grscanTestCase(unittest.TestCase):

    def test_grscan_smoke(self):
        tb = grscan(sdr=None)
        self.assertTrue(tb.freq_updated(2))
        time.sleep(1)
        self.assertFalse(tb.freq_updated(1))
        time.sleep(1)
        self.assertFalse(tb.freq_updated(1))
        tb.freq_update = time.time()
        self.assertTrue(tb.freq_updated(1))


if __name__ == '__main__':
    unittest.main()
