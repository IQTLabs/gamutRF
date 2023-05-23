#!/usr/bin/python3
import os
import tempfile
import unittest

from gamutrf.specgram import process_all_recordings
from gamutrf.utils import parse_filename


class FakeArgs:
    def __init__(
        self,
        nfft,
        cmap,
        ytics,
        bare,
        noverlap,
        iext,
        skip_exist,
        recording,
        workers,
        skip_fft,
        width,
        height,
        dpi,
        skip_sample_secs,
        max_sample_secs,
    ):
        self.nfft = nfft
        self.cmap = cmap
        self.ytics = ytics
        self.bare = bare
        self.noverlap = noverlap
        self.iext = iext
        self.skip_exist = skip_exist
        self.recording = recording
        self.workers = workers
        self.skip_fft = skip_fft
        self.width = width
        self.height = height
        self.dpi = dpi
        self.skip_sample_secs = 0
        self.max_sample_secs = 0


class SpecgramTestCase(unittest.TestCase):
    def test_process_recordings(self):
        with tempfile.TemporaryDirectory() as tempdir:
            recording = os.path.join(str(tempdir), "testrecording_100Hz_1000sps.raw")
            (
                _freq_center,
                sample_rate,
                _sample_dtype,
                sample_len,
                _sample_type,
                _sample_bits,
            ) = parse_filename(recording)
            samples = chr(0) * int(sample_len * sample_rate)
            with open(recording, "wb") as f:
                f.write(samples.encode("utf8"))
            fakeargs = FakeArgs(
                256,
                "turbo",
                1,
                True,
                0,
                "png",
                False,
                recording,
                2,
                True,
                11,
                8,
                100,
                0,
                0,
            )
            process_all_recordings(fakeargs)
            fakeargs.workers = 1
            fakeargs.skip_exist = False
            fakeargs.skip_fft = False
            process_all_recordings(fakeargs)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
