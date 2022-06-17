#!/usr/bin/python3
import os
import tempfile
import unittest

import pandas as pd

from gamutrf.sigwindows import choose_record_signal
from gamutrf.sigwindows import choose_recorders
from gamutrf.sigwindows import find_sig_windows
from gamutrf.sigwindows import freq_excluded
from gamutrf.sigwindows import parse_freq_excluded
from gamutrf.sigwindows import read_csv
from gamutrf.sigwindows import ROLLOVERHZ


TESTDIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'data')


class FakeArgs:

    def __init__(self, csv, minhz, maxhz, nrows, mindb):
        self.csv = csv
        self.minhz = minhz
        self.maxhz = maxhz
        self.nrows = nrows
        self.mindb = mindb
        self.maxframe = 0


class WindowsTestCase(unittest.TestCase):

    @staticmethod
    def _get_data(data):
        return os.path.join(TESTDIR, data)

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

    def test_choose_recorders(self):
        recorder_freq_exclusions = {
            'c1': ((100, 199),),
            'c2': ((300, 399),)}
        self.assertEqual(
            [(100, 'c2'), (200, 'c1')],
            choose_recorders([100, 200], recorder_freq_exclusions))
        recorder_freq_exclusions = {
            'c1': (),
            'c2': ((100, 199),)}
        self.assertEqual(
            [(100, 'c1'), (200, 'c2')],
            choose_recorders([100, 200], recorder_freq_exclusions))

    def test_freq_excluded(self):
        self.assertTrue(freq_excluded(100, ((100, 200),)))
        self.assertFalse(freq_excluded(99, ((100, 200),)))
        self.assertTrue(freq_excluded(1e9, ((1e6, None),)))
        self.assertFalse(freq_excluded(1e6, ((1e9, None),)))
        self.assertFalse(freq_excluded(1e9, ((None, 1e6),)))

    def test_parse_excluded(self):
        self.assertEqual(((100, 200), (200, None), (None, 100)),
                         parse_freq_excluded(['100-200', '200-', '-100']))

    def test_verybusy1g1(self):
        df = pd.read_csv(self._get_data('verybusy1g1.csv'),
                         delim_whitespace=True)
        signals = find_sig_windows(df)
        self.assertIn((757.877504, 775.853632, -7.159875173989217, -
                      24.08811337302156, 1.5591977316304422), signals, signals)
        self.assertIn((265.036816, 288.313568, -47.332711711197504, -
                      23.3951832550739, 10.327983621238149), signals, signals)
        self.assertIn((932.511936, 938.6112, -3.840928760465317,
                      7.672896900099766, 7.672896900099766), signals, signals)

    def test_find_wifi(self):
        df = pd.read_csv(self._get_data('wifi24.csv'), delim_whitespace=True)
        self.assertEqual(
            [(2420.422912, 2437.485056, -32.40808890037803, -44.99829069187393, -13.27136572108361)], find_sig_windows(df))

    def test_choose_record_signal(self):
        # One signal, one recorder.
        self.assertEqual([16], choose_record_signal([16], 1))
        # One signal reported multiple times, two recorders, but we only need to record it once.
        self.assertEqual([16], choose_record_signal([16, 16, 16, 16], 2))
        # One signal received less often, so record that, since we have only one recorder.
        self.assertEqual([110], choose_record_signal([20, 20, 20, 20, 110], 1))
        # We have two recorders so can afford to record the more common one as well.
        self.assertEqual([110, 20], choose_record_signal(
            [20, 20, 20, 20, 110], 2))


if __name__ == '__main__':
    unittest.main()
