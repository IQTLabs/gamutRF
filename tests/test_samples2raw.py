#!/usr/bin/python3

import os
import unittest

from gamutrf.samples2raw import make_procs_args


class Samples2RawTestCase(unittest.TestCase):

    def test_procs_args_(self):
        self.assertEqual(
            [['gunzip', '-c', 'gamutrf_recording1644265099_320000000Hz_20000000sps.s16.gz'],
             ['sox', '-t', 'raw', '-r', '20000000', '-c', '1', '-b', '16', '-e', 'signed-integer',
              '-', '-e', 'float', 'gamutrf_recording1644265099_320000000Hz_20000000sps.raw']],
            make_procs_args('gamutrf_recording1644265099_320000000Hz_20000000sps.s16.gz'))
        self.assertEqual(
            [['sox', '-t', 'raw', '-r', '20000000', '-c', '1', '-b', '16', '-e', 'signed-integer',
             'gamutrf_recording1644265099_320000000Hz_20000000sps.s16',
             '-e', 'float', 'gamutrf_recording1644265099_320000000Hz_20000000sps.raw']],
            make_procs_args('gamutrf_recording1644265099_320000000Hz_20000000sps.s16'))


if __name__ == '__main__':
    unittest.main()
