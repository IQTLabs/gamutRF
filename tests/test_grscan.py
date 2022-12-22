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
        tb = grscan(sdr="file:/dev/zero", samp_rate=int(1.024e6))
        tb.start()
        time.sleep(15)
        tb.stop()
        tb.wait()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
