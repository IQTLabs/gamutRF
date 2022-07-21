#!/usr/bin/python3

import os
import socket
import tempfile
import time
import unittest
import concurrent.futures

from gamutrf.sigfinder import init_prom_vars
from gamutrf.sigfinder import process_fft_lines
from gamutrf.sigfinder import ROLLOVERHZ
from gamutrf.sigfinder import udp_proxy


class FakeArgs:

    def __init__(self, log, rotatesecs, window, threshold, freq_start, freq_end,
                 fftlog, width, prominence, bin_mhz, record_bw_mbps, history, recorder,
                 fftgraph, logaddr, logport, max_raw_power):
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
        self.logaddr = logaddr
        self.logport = logport
        self.max_raw_power = max_raw_power


class EmptyFifo(Exception):
    pass


class FakeFifo:

    def __init__(self, test_data):
        self.test_data = test_data

    def read(self):
        if not self.test_data:
            raise EmptyFifo
        test_lines = []
        if self.test_data[0]:
            for line in self.test_data[0]:
                test_lines.append(' '.join((str(i) for i in line)))
            result = ('\n'.join(test_lines) + '\n').encode('utf8')
        else:
            result = None
        self.test_data = self.test_data[1:]
        return result


class SigFinderTestCase(unittest.TestCase):

    def test_process_fft_lines(self):
        with concurrent.futures.ProcessPoolExecutor(1) as executor:
            with tempfile.TemporaryDirectory() as tempdir:
                test_log = os.path.join(str(tempdir), 'test.csv')
                test_fftlog = os.path.join(str(tempdir), 'fft.csv')
                test_fftgraph = os.path.join(str(tempdir), 'fft.png')
                args = FakeArgs(test_log, 60, 4, -40, 100e6, 400e6, test_fftlog, 1, 5, 20, 21, 1, '', test_fftgraph, '127.0.0.1', 9999, 100)
                prom_vars = init_prom_vars()
                line_count = 0
                test_lines_1 = [(time.time(), ROLLOVERHZ + (1e5 * i), 0.001) for i in range(1000)]
                line_count += len(test_lines_1)
                test_lines_2 = [(time.time(), ROLLOVERHZ + (1e5 * (i + line_count)), 0.5) for i in range(100)]
                line_count += len(test_lines_2)
                test_lines_3 = [(time.time(), ROLLOVERHZ + (1e5 * (i + line_count)), 0.1) for i in range(1000)]
                line_count += len(test_lines_3)
                test_lines_4 = [(time.time(), ROLLOVERHZ + 1e6, 0.1)]
                fifo = FakeFifo([test_lines_1, test_lines_2, test_lines_3, test_lines_4, None])
                self.assertRaises(EmptyFifo, lambda: process_fft_lines(args, prom_vars, fifo, 'csv', executor))
                self.assertTrue(os.path.exists(test_fftlog))
                self.assertTrue(os.path.exists(test_fftgraph))

    def disabled_hangs_in_gha_test_udp_proxy(self):
        args = FakeArgs('', 60, 4, -40, 100e6, 400e6, '', None, 5, 20, 21, 1, '', '', '127.0.0.1', 9999, 100)

        with tempfile.TemporaryDirectory() as tempdir:
            fifo_name = os.path.join(tempdir, 'fftfifo')
            os.mkfifo(fifo_name)
            returned = None
            test_bytes = b'1, 2, 3\n4, 5, 6\n'

            with concurrent.futures.ProcessPoolExecutor(1) as executor:
                executor.submit(udp_proxy, args, fifo_name, 1, 0)
                with open(fifo_name, 'rb') as fifo:
                    os.set_blocking(fifo.fileno(), False)
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                        for retry in range(5):
                            sock.sendto(test_bytes, (args.logaddr, args.logport))
                            returned = fifo.read()
                            if returned:
                                break
                            time.sleep(1)
                        self.assertGreater(returned.find(test_bytes), 0)


if __name__ == '__main__':
    unittest.main()
