#!/usr/bin/python3
import os
import tempfile
import unittest

from gamutrf.scan2mp4 import generate_frames
from gamutrf.scan2mp4 import read_csv
from gamutrf.scan2mp4 import ROLLOVERHZ


class FakeArgs:

    def __init__(self, csv, minhz, maxhz, nrows, mindb):
        self.csv = csv
        self.minhz = minhz
        self.maxhz = maxhz
        self.nrows = nrows
        self.mindb = mindb
        self.maxframe = 0


class Scan2MP4TestCase(unittest.TestCase):

    @staticmethod
    def _test_samples(test_csv, test_freqs, test_ts):
        with open(test_csv, 'w', encoding='utf-8') as f:
            samples = []
            for ts in test_ts:
                frame_samples = [
                    '\t'.join([str(ts), str(f), str(1 - 1 / f)]) for f in test_freqs]
                samples.extend(frame_samples)
            f.write('\n'.join(samples))

    def test_truncated_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_csv = os.path.join(str(tempdir), 'test.csv')
            with open(test_csv, 'w', encoding='utf-8') as f:
                f.write('1 1 1\n')
                f.write('1 2\n')
                f.write('1\n')
            args = FakeArgs(test_csv, 0, 1e3, 10e3, -40)
            for frame, frame_df in read_csv(args):
                self.assertEqual(0, frame)
                self.assertEqual(1, len(frame_df))

    def test_read_csv(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_csv = os.path.join(str(tempdir), 'test.csv')
            test_freqs = [f * ROLLOVERHZ for f in range(1, 10)]
            test_ts = [0, 60, 120]
            self._test_samples(test_csv, test_freqs, test_ts)
            args = FakeArgs(test_csv, 0, 1e3, 10e3, -40)
            expected_frame = 0
            tses = []
            for frame, frame_df in read_csv(args):
                self.assertEqual(expected_frame, frame)
                expected_frame += 1
                self.assertEqual(list(frame_df['freq']), [
                                 f / 1e6 for f in test_freqs])
                self.assertEqual(9, len(frame_df[frame_df['db'] != 0]))
                tses.append(frame_df['ts'].iat[0])
            self.assertEqual(tses, test_ts)

    def test_generate_frames(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_csv = os.path.join(str(tempdir), 'test.csv')
            test_freqs = [f * ROLLOVERHZ for f in range(1, 10)]
            test_ts = (0, 60, 120)
            self._test_samples(test_csv, test_freqs, test_ts)
            args = FakeArgs(test_csv, 0, 1e3, 10e3, -40)
            self.assertTrue(generate_frames(args, tempdir))


if __name__ == '__main__':
    unittest.main()
