#!/usr/bin/python3
import glob
import os
import subprocess
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

    def set_thread_priority(self, _priority):
        return


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

    def set_thread_priority(self, _priority):
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
        get_source("ettus", 1e3, 10, 1024, 1024, uhd_lib=FakeUHD())

    def test_fake_soapy(self):
        sources, _, workaround = get_source(
            "SoapyAIRT", 1e3, 10, 1024, 1024, soapy_lib=FakeSoapy()
        )
        tb = FakeTb(sources, workaround, 100e6)
        tb.start()

    def test_get_source_smoke(self):
        for sdr in ("ettus", "bladerf"):
            self.assertRaises(RuntimeError, get_source, sdr, 1e3, 10, 1024, 1024)

    def run_grscan_smoke(self, pretune, wavelearner, write_samples, test_file):
        with tempfile.TemporaryDirectory() as tempdir:
            freq_start = 1e9
            freq_end = 2e9
            samp_rate = int(1.024e6)
            sdr = "tuneable_test_source"
            if test_file:
                freq_end = 0
                sdr_file = os.path.join(
                    tempdir,
                    f"gamutrf_recording1_{int(freq_start)}Hz_{int(samp_rate)}sps.raw",
                )
                subprocess.check_call(
                    [
                        "dd",
                        "if=/dev/urandom",
                        f"of={sdr_file}",
                        f"bs={samp_rate}",
                        "count=10",
                    ]
                )
                sdr = "file:" + sdr_file
            tb = grscan(
                freq_start=freq_start,
                freq_end=freq_end,
                sdr=sdr,
                samp_rate=samp_rate,
                write_samples=write_samples,
                sample_dir=tempdir,
                iqtlabs=iqtlabs,
                wavelearner=wavelearner,
                rotate_secs=900,
                db_clamp_floor=-1e6,
                pretune=pretune,
                fft_batch_size=16,
                inference_output_dir=str(tempdir),
            )
            tb.start()
            time.sleep(3)
            tb.stop()
            tb.wait()
            del tb
            if not write_samples:
                return
            self.assertTrue([x for x in glob.glob(f"{tempdir}/*/*zst")])
            self.assertTrue([x for x in glob.glob(f"{tempdir}/*/*sigmf-meta")])

    def test_grscan_smoke(self):
        for pretune in (True, False):
            self.run_grscan_smoke(pretune, False, True, True)
            for wavelearner in (FakeWaveLearner(), None):
                for write_samples in (0, 1):
                    self.run_grscan_smoke(pretune, wavelearner, write_samples, False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
