#!/usr/bin/python3
import time
import unittest

from gnuradio import iqtlabs
from gnuradio import fft  # pytype: disable=import-error
from gnuradio.fft import window  # pytype: disable=import-error

from gamutrf.grsource import get_source
from gamutrf.grscan import grscan


class FakeTb:
    pass


class FakeWaveLearner:
    def fft(self, batch_fft_size, _fft_size, forward):
        return fft.fft_vcc(
            batch_fft_size, forward, window.blackmanharris(batch_fft_size), True, 1
        )


class GrscanTestCase(unittest.TestCase):
    def test_get_source_smoke(self):
        self.assertRaises(RuntimeError, get_source, FakeTb, "ettus", 1e3, 10)
        self.assertRaises(RuntimeError, get_source, FakeTb, "bladerf", 1e3, 10)

    def test_grscan_smoke(self):
        tb = grscan(
            sdr="file:/dev/zero",
            samp_rate=int(1.024e6),
            write_samples=1,
            sample_dir="/tmp",
            iqtlabs=iqtlabs,
            wavelearner=None,
        )
        tb.start()
        time.sleep(15)
        tb.stop()
        tb.wait()

    def test_grscan_wavelearner_smoke(self):
        tb = grscan(
            sdr="file:/dev/zero",
            samp_rate=int(1.024e6),
            write_samples=1,
            sample_dir="/tmp",
            iqtlabs=iqtlabs,
            wavelearner=FakeWaveLearner(),
        )
        tb.start()
        time.sleep(15)
        tb.stop()
        tb.wait()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
