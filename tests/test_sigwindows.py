#!/usr/bin/python3
import os
import tempfile
import unittest

from gamutrf.sigwindows import choose_record_signal
from gamutrf.sigwindows import choose_recorders
from gamutrf.sigwindows import freq_excluded
from gamutrf.sigwindows import parse_freq_excluded
from gamutrf.sigwindows import read_csv
from gamutrf.sigwindows import ROLLOVERHZ, ROLLING_FACTOR


TESTDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


class FakeArgs:
    def __init__(self, csv, minhz, maxhz, nrows, mindb):
        self.csv = csv
        self.minhz = minhz
        self.maxhz = maxhz
        self.nrows = nrows
        self.mindb = mindb
        self.maxframe = 0
        self.db_rolling_factor = ROLLING_FACTOR


class WindowsTestCase(unittest.TestCase):
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

    def test_truncated_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_csv = os.path.join(str(tempdir), "test.csv")
            with open(test_csv, "w", encoding="utf-8") as f:
                f.write("1 1 1\n")
                f.write("1 2\n")
                f.write("1\n")
            args = FakeArgs(test_csv, 0, 1e3, 10e3, -40)
            for frame, frame_df in read_csv(args):
                # TODO tests never get here
                self.assertEqual(0, frame)
                self.assertEqual(1, len(frame_df))

    def test_read_csv(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_csv = os.path.join(str(tempdir), "test.csv")
            test_freqs = [f * ROLLOVERHZ for f in range(1, 10)]
            test_ts = [0, 60, 120]
            self._test_samples(test_csv, test_freqs, test_ts)
            args = FakeArgs(test_csv, 0, 1e3, 10e3, -40)
            expected_frame = 0
            tses = []
            for frame, frame_df in read_csv(args):
                self.assertEqual(expected_frame, frame)
                expected_frame += 1
                self.assertEqual(list(frame_df["freq"]), [f / 1e6 for f in test_freqs])
                self.assertEqual(9, len(frame_df[frame_df["db"] != 0]))
                tses.append(frame_df["ts"].iat[0])
            self.assertEqual(tses, test_ts)

    def test_choose_recorders(self):
        recorder_freq_exclusions = {"c1": ((100, 199),), "c2": ((300, 399),)}
        self.assertEqual(
            [(100, "c2"), (200, "c1")],
            choose_recorders([100, 200], recorder_freq_exclusions, 1),
        )
        recorder_freq_exclusions = {"c1": (), "c2": ((100, 199),)}
        self.assertEqual(
            [(100, "c1"), (200, "c2")],
            choose_recorders([100, 200], recorder_freq_exclusions, 1),
        )

    def test_freq_excluded(self):
        self.assertTrue(freq_excluded(100, ((100, 200),)))
        self.assertFalse(freq_excluded(99, ((100, 200),)))
        self.assertTrue(freq_excluded(1e9, ((1e6, None),)))
        self.assertFalse(freq_excluded(1e6, ((1e9, None),)))
        self.assertFalse(freq_excluded(1e9, ((None, 1e6),)))

    def test_parse_excluded(self):
        self.assertEqual(
            ((100, 200), (200, None), (None, 100)),
            parse_freq_excluded(["100-200", "200-", "-100"]),
        )

    def test_choose_record_signal(self):
        # One signal, one recorder.
        self.assertEqual([16], choose_record_signal([16], 1))
        # One signal reported multiple times, two recorders, but we only need to record it once.
        self.assertEqual([16], choose_record_signal([16, 16, 16, 16], 2))
        # One signal received less often, so record that, since we have only one recorder.
        self.assertEqual([110], choose_record_signal([20, 20, 20, 20, 110], 1))
        # We have two recorders so can afford to record the more common one as well.
        self.assertEqual([110, 20], choose_record_signal([20, 20, 20, 20, 110], 2))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
