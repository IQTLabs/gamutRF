#!/usr/bin/python3
import tempfile
import time
import unittest

from gnuradio import iqtlabs
from gnuradio import fft  # pytype: disable=import-error
from gnuradio.fft import window  # pytype: disable=import-error
from gnuradio import uhd

from gamutrf.grsource import get_source
from gamutrf.grscan import grscan


class FakeWaveLearner:
    def fft(self, batch_fft_size, _fft_size, forward):
        return fft.fft_vcc(
            batch_fft_size, forward, window.blackmanharris(batch_fft_size), True, 1
        )


class FakeUHDSource:
    def __init__(self):
        self.time_spec = None
        return

    def set_antenna(self, _antenna, _channel):
        return

    def set_samp_rate(self, _samp_rate):
        return

    def set_gain(self, _gain, _channel):
        return

    def set_rx_agc(self, _agc, _channel):
        return

    def set_time_now(self, time_spec, _mb):
        self.time_spec = time_spec

    def get_time_now(self):
        return self.time_spec


class FakeUHD:
    def __init__(self):
        self.time_spec = uhd.time_spec
        self.ALL_MBOARDS = uhd.ALL_MBOARDS

    def stream_args(self, cpu_format, args, channels):
        return None

    def usrp_source(self, sdrargs, sargs):
        return FakeUHDSource()


class FakeSoapySource:
    def set_sample_rate(self, _channel, _sample_rate):
        return

    def set_bandwidth(self, _channel, _bw):
        return


class FakeSoapy:
    def __init__(self):
        return

    def source(self, dev, cpu_format, arg1, arg2, stream_args, tune_args, settings):
        return FakeSoapySource()


class FakeTb:
    def __init__(self, test_sources, test_workaround, samp_rate):
        self.sources = test_sources
        self.workaround = test_workaround
        self.samp_rate = samp_rate

    def start(self):
        self.workaround(self)


class GrscanTestCase(unittest.TestCase):
    def test_fake_uhd(self):
        get_source("ettus", 1e3, 10, uhd_lib=FakeUHD())

    def test_fake_soapy(self):
        sources, _, workaround = get_source("SoapyAIRT", 1e3, 10, soapy_lib=FakeSoapy())
        tb = FakeTb(sources, workaround, 100e6)
        tb.start()

    def test_get_source_smoke(self):
        for sdr in ("ettus", "bladerf"):
            self.assertRaises(RuntimeError, get_source, sdr, 1e3, 10)

    def test_grscan_smoke(self):
        with tempfile.TemporaryDirectory() as tempdir:
            tb = grscan(
                sdr="tuneable_test_source",
                samp_rate=int(1.024e6),
                write_samples=1,
                sample_dir=tempdir,
                iqtlabs=iqtlabs,
                wavelearner=None,
            )
            tb.start()
            time.sleep(15)
            tb.stop()
            tb.wait()

    def test_grscan_wavelearner_smoke(self):
        with tempfile.TemporaryDirectory() as tempdir:
            tb = grscan(
                sdr="tuneable_test_source",
                samp_rate=int(1.024e6),
                write_samples=1,
                sample_dir=tempdir,
                iqtlabs=iqtlabs,
                wavelearner=FakeWaveLearner(),
            )
            tb.start()
            time.sleep(15)
            tb.stop()
            tb.wait()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
