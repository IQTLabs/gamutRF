#!/usr/bin/python3

import os
import socket
import tempfile
import time
import unittest
import concurrent.futures

from gamutrf.sigfinder import init_prom_vars
from gamutrf.sigfinder import process_fft_lines
from gamutrf.sigfinder import udp_proxy
from gamutrf.sigfinder import argument_parser
from gamutrf.utils import rotate_file_n


class FakeArgs:
    def __init__(
        self,
        log,
        rotatesecs,
        window,
        threshold,
        freq_start,
        freq_end,
        fftlog,
        width,
        prominence,
        bin_mhz,
        record_bw_msps,
        history,
        recorder,
        fftgraph,
        logaddr,
        logport,
        max_raw_power,
        nfftgraph,
    ):
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
        self.record_bw_msps = record_bw_msps
        self.history = history
        self.recorder = recorder
        self.fftgraph = fftgraph
        self.logaddr = logaddr
        self.logport = logport
        self.max_raw_power = max_raw_power
        self.nfftgraph = nfftgraph


class SigFinderTestCase(unittest.TestCase):
    def test_rotate_file_n(self):

        rotate_file_n("notthere.log", 100, require_initial=False)

        with tempfile.TemporaryDirectory() as tempdir:
            test_fftgraph = os.path.join(str(tempdir), "fft.png")

            def write_file():
                with open(test_fftgraph, "w") as f:
                    f.write("test")

            def exists(name):
                return os.path.exists(os.path.join(str(tempdir), name))

            write_file()
            rotate_file_n(test_fftgraph, 3)
            self.assertTrue(exists("fft.1.png"))
            write_file()
            rotate_file_n(test_fftgraph, 3)
            self.assertTrue(exists("fft.2.png"))
            write_file()
            rotate_file_n(test_fftgraph, 3)
            self.assertTrue(exists("fft.3.png"))
            write_file()
            rotate_file_n(test_fftgraph, 3)
            self.assertFalse(exists("fft.4.png"))

    def test_argument_parser(self):
        argument_parser()

    def test_process_fft_lines(self):
        with concurrent.futures.ProcessPoolExecutor(1) as executor:
            with tempfile.TemporaryDirectory() as tempdir:
                test_log = os.path.join(str(tempdir), "test.csv")
                test_fftlog = os.path.join(str(tempdir), "fft.csv")
                test_fftgraph = os.path.join(str(tempdir), "fft.png")
                buff_file = os.path.join(str(tempdir), "buff_file")
                args = FakeArgs(
                    test_log,
                    60,
                    4,
                    -40,
                    100e6,
                    400e6,
                    test_fftlog,
                    1,
                    5,
                    20,
                    21,
                    1,
                    "",
                    test_fftgraph,
                    "127.0.0.1",
                    9999,
                    100,
                    10,
                )
                prom_vars = init_prom_vars()
                with open(buff_file, "w", encoding="utf8") as bf:
                    for _ in range(2):
                        freq = 100e6
                        while freq < 400e6:
                            bf.write(f"{int(time.time())} {int(freq)} 0.001\n")
                            freq += 1e5
                process_fft_lines(
                    args, prom_vars, buff_file, "csv", executor, runonce=True
                )
                self.assertTrue(os.path.exists(test_fftlog))
                self.assertTrue(os.path.exists(test_fftgraph))

    def test_udp_proxy(self):
        args = FakeArgs(
            "",
            60,
            4,
            -40,
            100e6,
            400e6,
            "",
            None,
            5,
            20,
            21,
            1,
            "",
            "",
            "127.0.0.1",
            9999,
            100,
            10,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            buff_file = os.path.join(tempdir, "buff_file")
            test_bytes = b"1, 2, 3\n4, 5, 6\n"
            shutdown_str = b"shutdown\n"

            with concurrent.futures.ProcessPoolExecutor(1) as executor:
                executor.submit(udp_proxy, args, buff_file, 1, shutdown_str)
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    for _ in range(5):
                        sock.sendto(test_bytes, (args.logaddr, args.logport))
                        if os.path.exists(buff_file):
                            break
                        time.sleep(1)
                    sock.sendto(shutdown_str, (args.logaddr, args.logport))
                    with open(buff_file, "rb") as bf:
                        content = bf.read()
                    self.assertGreater(
                        content.find(b"4, 5, 6\n"), -1, (content, test_bytes)
                    )
                    os.remove(buff_file)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
