#!/usr/bin/python3

import os
import tempfile
import time
import unittest
import concurrent.futures
import zmq
import zstandard

from gamutrf.sigfinder import Result
from gamutrf.sigfinder import argument_parser
from gamutrf.sigfinder import error_response
from gamutrf.sigfinder import falcon_response
from gamutrf.sigfinder import fft_proxy
from gamutrf.sigfinder import init_prom_vars
from gamutrf.sigfinder import ok_response
from gamutrf.sigfinder import process_fft_lines
from gamutrf.sigwindows import ROLLING_FACTOR
from gamutrf.utils import rotate_file_n


def null_proxy():
    for i in range(10):
        time.sleep(1)


class FakeResponse:
    def __init__(self):
        self.status = 200
        self.text = ""
        self.content_type = "text/html"


class FakeRequest:
    def __init__(self):
        self.media = {
            "worker": "foo",
            "frequency": "foo",
            "bandwidth": "foo",
            "duration": "foo",
            "repeat": "-1",
        }


def test_falcon_response():
    resp = FakeResponse()
    falcon_response(resp, "test", 500)


def test_ok_response():
    resp = FakeResponse()
    ok_response(resp)


def test_error_response():
    resp = FakeResponse()
    error_response(resp)


def test_result_on_post():
    resp = FakeResponse()
    req = FakeRequest()
    result = Result()
    result.on_post(req, resp)
    req.media["repeat"] = 2
    result.on_post(req, resp)


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
        max_recorder_signals,
        running_fft_secs,
        nfftplots,
        skip_tune_step_fft,
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
        self.max_recorder_signals = max_recorder_signals
        self.running_fft_secs = running_fft_secs
        self.nfftplots = nfftplots
        self.skip_tune_step_fft = skip_tune_step_fft
        self.db_rolling_factor = ROLLING_FACTOR


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
            proxy_result = executor.submit(null_proxy)
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
                    1,
                    1,
                    1,
                    0,
                )
                prom_vars = init_prom_vars()
                context = zstandard.ZstdCompressor()
                with open(buff_file, "wb") as zbf:
                    with context.stream_writer(zbf) as bf:
                        for _ in range(2):
                            freq = 100e6
                            while freq < 400e6:
                                bf.write(
                                    f"{int(time.time())} {int(freq)} 0.001\n".encode(
                                        "utf8"
                                    )
                                )
                                freq += 1e5
                process_fft_lines(
                    args, prom_vars, buff_file, executor, proxy_result, runonce=True
                )
                self.assertTrue(os.path.exists(test_fftlog))
                self.assertTrue(os.path.exists(test_fftgraph))

    def test_fft_proxy(self):
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
            1,
            1,
            1,
            0,
        )
        zstd_context = zstandard.ZstdDecompressor()

        with tempfile.TemporaryDirectory() as tempdir:
            buff_file = os.path.join(tempdir, "buff_file")
            test_bytes = b"1, 2, 3\n4, 5, 6\n"
            shutdown_str = b"shutdown\n"
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.bind(f"tcp://{args.logaddr}:{args.logport}")

            with concurrent.futures.ProcessPoolExecutor(1) as executor:
                executor.submit(fft_proxy, args, buff_file, 1, shutdown_str)
                for _ in range(5):
                    socket.send(test_bytes)
                    if os.path.exists(buff_file):
                        break
                    time.sleep(1)
                socket.send(shutdown_str)
                with open(buff_file, "rb") as zbf:
                    with zstd_context.stream_reader(zbf) as bf:
                        content = bf.read()
                self.assertGreater(
                    content.find(b"4, 5, 6\n"), -1, (content, test_bytes)
                )
                os.remove(buff_file)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
