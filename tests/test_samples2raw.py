#!/usr/bin/python3

import tempfile
import os
import unittest

import numpy as np

from gamutrf.samples2raw import make_procs_args, run_procs, argument_parser


class Samples2RawTestCase(unittest.TestCase):

    def test_argument_parser(self):
        argument_parser()

    def test_s2r(self):
        with tempfile.TemporaryDirectory() as tempdir:
            base_test_name = 'gamutrf_recording1_1Hz_100sps'
            test_file_name = os.path.join(tempdir, '.'.join((base_test_name, 's16')))
            out_file_name = os.path.join(tempdir, '.'.join((base_test_name, 'raw')))
            test_data = np.int16([-(2**15)] * int(1e2 * 2))
            test_float_data = np.float32([-1] * int(1e2 * 2))
            test_data.tofile(test_file_name)
            run_procs(make_procs_args(test_file_name, 'float'))
            converted_data = np.fromfile(out_file_name, dtype='<f4')
            self.assertTrue(np.array_equal(converted_data, test_float_data), converted_data)


if __name__ == '__main__':
    unittest.main()
