#!/usr/bin/python3
import os
import tempfile
import unittest

from gamutrf.sigwindows import choose_record_signal
from gamutrf.sigwindows import choose_recorders
from gamutrf.sigwindows import freq_excluded
from gamutrf.sigwindows import parse_freq_excluded
from gamutrf.sigwindows import ROLLOVERHZ, ROLLING_FACTOR


TESTDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


class WindowsTestCase(unittest.TestCase):
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
