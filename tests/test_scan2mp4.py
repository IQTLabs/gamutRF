#!/usr/bin/python3
import os
import tempfile
import unittest

from gamutrf.scan2mp4 import generate_frames
from gamutrf.sigwindows import ROLLOVERHZ, ROLLING_FACTOR


class FakeArgs:
    def __init__(self, csv, minhz, maxhz, nrows, mindb):
        self.csv = csv
        self.minhz = minhz
        self.maxhz = maxhz
        self.nrows = nrows
        self.mindb = mindb
        self.maxframe = 0
        self.db_rolling_factor = ROLLING_FACTOR


class Scan2MP4TestCase(unittest.TestCase):
    @staticmethod
    def _test_samples(test_csv, test_freqs, test_ts):
        with open(test_csv, "w", encoding="utf-8") as f:
            samples = []
            for ts in test_ts:
                frame_samples = [
                    "\t".join([str(ts), str(f), str(1 - 1 / f)]) for f in test_freqs
                ]
                samples.extend(frame_samples)
            f.write("\n".join(samples))

    def test_generate_frames(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_csv = os.path.join(str(tempdir), "test.csv")
            test_freqs = [f * ROLLOVERHZ for f in range(1, 10)]
            test_ts = (0, 60, 120)
            self._test_samples(test_csv, test_freqs, test_ts)
            args = FakeArgs(test_csv, 0, 1e3, 10e3, -40)
            self.assertTrue(generate_frames(args, tempdir))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
