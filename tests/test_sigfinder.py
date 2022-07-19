#!/usr/bin/python3
import concurrent.futures
import os
import tempfile
import time
import unittest

from gamutrf.sigfinder import init_prom_vars
from gamutrf.sigfinder import process_fft_lines
from gamutrf.sigfinder import ROLLOVERHZ


class FakeArgs:

    def __init__(self, log, rotatesecs, window, threshold, freq_start, freq_end,
                 fftlog, width, prominence, bin_mhz, record_bw_mbps, history, recorder,
                 fftgraph):
        self.log = log
        self.rotatesecs = rotatesecs
        self.window = window
        self.threshold = threshold
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.fftlog = fftlog
        self.width = width
        self.threshold = threshold
        self.prominence = prominence
        self.bin_mhz = bin_mhz
        self.record_bw_mbps = record_bw_mbps
        self.history = history
        self.recorder = recorder
        self.fftgraph = fftgraph 


class EmptyFifo(Exception):
    pass


class FakeFifo:

    def __init__(self, test_data):
        self.test_data = test_data

    def read(self):
        if not self.test_data:
            raise EmptyFifo
        test_lines = []
        for line in self.test_data[0]:
            test_lines.append(' '.join((str(i) for i in line)))
        result = ('\n'.join(test_lines) + '\n').encode('utf8')
        self.test_data = self.test_data[1:]
        return result


class SigFinderTestCase(unittest.TestCase):

    def test_process_fft_lines(self):
        with concurrent.futures.ProcessPoolExecutor(1) as executor:
            with tempfile.TemporaryDirectory() as tempdir:
                test_log = os.path.join(str(tempdir), 'test.csv')
                test_fftlog = os.path.join(str(tempdir), 'fft.csv')
                test_fftgraph = os.path.join(str(tempdir), 'fft.png')
                args = FakeArgs(test_log, 60, 4, -40, 100e6, 400e6, test_fftlog, None, 5, 20, 21, 1, '', test_fftgraph)
                prom_vars = init_prom_vars()
                test_lines_1 = [(time.time(), ROLLOVERHZ + (1e6 * i), 0.001) for i in range(100)]
                test_lines_2 = [(time.time(), ROLLOVERHZ + (1e6 * (i + 100)), 0.5) for i in range(5)]
                test_lines_3 = [(time.time(), ROLLOVERHZ + (1e6 * (i + 105)), 0.1) for i in range(100)]
                test_lines_4 = [(time.time(), ROLLOVERHZ - 1e6, 0.1)]
                fifo = FakeFifo([test_lines_1, test_lines_2, test_lines_3, test_lines_4])
                self.assertRaises(EmptyFifo, lambda: process_fft_lines(args, prom_vars, fifo, 'csv', executor))


if __name__ == '__main__':
    unittest.main()
