#!/usr/bin/python3
import time
import unittest

from gamutrf.grsource import get_source
from gamutrf.grscan import grscan


class FakeTb:
    pass


class GrscanTestCase(unittest.TestCase):
    def test_get_source_smoke(self):
        self.assertRaises(RuntimeError, get_source, FakeTb, "ettus", 1e3, 10)
        self.assertRaises(RuntimeError, get_source, FakeTb, "bladerf", 1e3, 10)

    def test_grscan_smoke(self):
        start = time.time()
        tb = grscan(sdr="file:/dev/zero")
        tb.start()
        for i in range(5):
            if tb.freq_update:
                break
            time.sleep(1)
        self.assertTrue(tb.freq_updated(5))
        self.assertGreater(tb.freq_update, start)
        tb.stop()
        tb.wait()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
