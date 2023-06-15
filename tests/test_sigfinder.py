#!/usr/bin/python3

import json
import os
import pathlib
import tempfile
import threading
import time
import unittest
import concurrent.futures
import zmq
import zstandard

from gamutrf.sigfinder import Result
from gamutrf.sigfinder import argument_parser
from gamutrf.sigfinder import error_response
from gamutrf.sigfinder import falcon_response
from gamutrf.sigfinder import init_prom_vars
from gamutrf.sigfinder import ok_response
from gamutrf.sigfinder import process_scans
from gamutrf.sigwindows import ROLLING_FACTOR
from gamutrf.utils import rotate_file_n
from gamutrf.zmqreceiver import fft_proxy, ZmqReceiver, parse_scanners


def null_proxy(*args, **kwargs):
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
        fftlog,
        width,
        prominence,
        bin_mhz,
        record_bw_msps,
        history,
        recorder,
        fftgraph,
        scanners,
        nfftgraph,
        max_recorder_signals,
        running_fft_secs,
        nfftplots,
        skip_tune_step_fft,
        buff_path,
    ):
        self.log = log
        self.rotatesecs = rotatesecs
        self.window = window
        self.threshold = threshold
        self.fftlog = fftlog
        self.width = width
        self.threshold = threshold
        self.prominence = prominence
        self.bin_mhz = bin_mhz
        self.record_bw_msps = record_bw_msps
        self.history = history
        self.recorder = recorder
        self.fftgraph = fftgraph
        self.scanners = scanners
        self.nfftgraph = nfftgraph
        self.max_recorder_signals = max_recorder_signals
        self.running_fft_secs = running_fft_secs
        self.nfftplots = nfftplots
        self.skip_tune_step_fft = skip_tune_step_fft
        self.db_rolling_factor = ROLLING_FACTOR
        self.buff_path = buff_path


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

    def test_process_scans(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            with tempfile.TemporaryDirectory() as tempdir:
                test_log = os.path.join(str(tempdir), "test.csv")
                test_fftlog = os.path.join(str(tempdir), "fft.csv")
                test_fftgraph = os.path.join(str(tempdir), "fft.png")
                args = FakeArgs(
                    test_log,
                    60,
                    4,
                    -40,
                    test_fftlog,
                    1,
                    5,
                    20,
                    21,
                    1,
                    "",
                    test_fftgraph,
                    "127.0.0.1:9999",
                    10,
                    1,
                    1,
                    1,
                    0,
                    str(tempdir),
                )
                zmqr = ZmqReceiver(
                    scanners=parse_scanners(args.scanners), proxy=null_proxy
                )
                prom_vars = init_prom_vars()
                context = zstandard.ZstdCompressor()
                freq_start = 100e6
                freq_end = 400e6
                scan_config = {
                    "freq_start": freq_start,
                    "freq_end": freq_end,
                    "sample_rate": int(1e6),
                }
                with open(zmqr.scanners[0].buff_file, "wb") as zbf:
                    with context.stream_writer(zbf) as bf:
                        for _ in range(2):
                            output = {
                                "ts": int(time.time()),
                                "sweep_start": int(time.time()),
                                "config": {
                                    "freq_start": freq_start,
                                    "freq_end": freq_end,
                                    "sample_rate": int(1e6),
                                },
                                "buckets": {},
                            }
                            freq = freq_start
                            while freq < freq_end:
                                output["buckets"][str(freq)] = -50
                                freq += 1e5
                            bf.write(bytes(json.dumps(output) + "\n", encoding="utf8"))
                            time.sleep(1)
                process_thread = threading.Thread(
                    target=process_scans,
                    args=(
                        args,
                        prom_vars,
                        executor,
                        zmqr,
                    ),
                )
                process_thread.start()
                for i in range(10):
                    if os.path.exists(test_fftlog) and os.path.exists(test_fftgraph):
                        break
                    time.sleep(1)
                zmqr.stop()
                process_thread.join()
                self.assertTrue(os.path.exists(test_fftlog))
                self.assertTrue(os.path.exists(test_fftgraph))

    def test_fft_proxy(self):
        args = FakeArgs(
            "",
            60,
            4,
            -40,
            "",
            None,
            5,
            20,
            21,
            1,
            "",
            "",
            "127.0.0.1:9999",
            10,
            1,
            1,
            1,
            0,
            "",
        )
        zstd_context = zstandard.ZstdDecompressor()

        with tempfile.TemporaryDirectory() as tempdir:
            live_file = pathlib.Path(os.path.join(tempdir, "live_file"))
            live_file.touch()
            buff_file = os.path.join(tempdir, "buff_file")
            test_bytes = b"1, 2, 3\n4, 5, 6\n"
            shutdown_str = b"shutdown\n"
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            scanners = parse_scanners(args.scanners)
            scanner = scanners[0]
            addr, port = scanner
            socket.bind(f"tcp://{addr}:{port}")

            with concurrent.futures.ProcessPoolExecutor(1) as executor:
                executor.submit(
                    fft_proxy,
                    addr,
                    port,
                    buff_file,
                    1,
                    live_file=live_file,
                )
                for _ in range(5):
                    socket.send(test_bytes)
                    if os.path.exists(buff_file):
                        break
                    time.sleep(1)
                live_file.unlink()
                with open(buff_file, "rb") as zbf:
                    with zstd_context.stream_reader(zbf) as bf:
                        content = bf.read()
                self.assertGreater(
                    content.find(b"4, 5, 6\n"), -1, (content, test_bytes)
                )
                os.remove(buff_file)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
