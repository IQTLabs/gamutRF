#!/usr/bin/python3

import os
import tempfile
import time
import unittest

from gamutrf.sigfinder import process_fft_lines, init_prom_vars


class FakeArgs:

    def __init__(self, log, rotatesecs, window, threshold, freq_start, freq_end):
        self.log = log
        self.rotatesecs = rotatesecs
        self.window = window
        self.threshold = threshold
        self.freq_start = freq_start
        self.freq_end = freq_end


class FakeSock:

    def __init__(self, txt):
        self.txt = txt

    def recvfrom(self, _):
        result = (self.txt, None)
        if self.txt:
            self.txt = ''.encode('utf8')
        return result


class SigFinderTestCase(unittest.TestCase):

    def test_process_fft_lines(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_log = os.path.join(str(tempdir), 'test.csv')
            args = FakeArgs(test_log, 60, 4, 1, 1e9, 2e9)
            prom_vars = init_prom_vars()
            sock = FakeSock(f'{time.time()} 1e9 0.1\n'.encode('utf8'))
            process_fft_lines(args, prom_vars, sock, 'csv')


if __name__ == '__main__':
    unittest.main()
