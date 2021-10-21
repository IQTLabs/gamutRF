#!/usr/bin/python3
import os
import unittest

import pandas as pd

from gamutrf.sigwindows import find_sig_windows, choose_record_signal

TESTDIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'data')


class WindowsTestCase(unittest.TestCase):

    @staticmethod
    def _get_data(data):
        return os.path.join(TESTDIR, data)

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
        self.assertEqual([16], choose_record_signal([20], 1, 8))
        # One signal reported multiple times, two recorders, but we only need to record it once.
        self.assertEqual([16], choose_record_signal([20, 20, 20, 20], 2, 8))
        # One signal received less often, so record that, since we have only one recorder.
        self.assertEqual([112], choose_record_signal([20, 20, 20, 20, 110], 1, 8))
        # We have two recorders so can afford to record the more common one as well.
        self.assertEqual([112, 16], choose_record_signal([20, 20, 20, 20, 110], 2, 8))


if __name__ == '__main__':
    unittest.main()
