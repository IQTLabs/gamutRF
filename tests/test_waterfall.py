#!/usr/bin/python3
import copy
import logging
import glob
import os
import tempfile
import time
import unittest
import pandas as pd
from gamutrf.waterfall import argument_parser, waterfall


class FakeZmqReceiver:
    def __init__(self, run_secs):
        self.start_time = time.time()
        self.run_secs = run_secs
        self.fake_results = [
            ({}, pd.DataFrame([{"ts": 1.0, "freq": 1, "db": 50.0}])),
            (None, None),
        ]
        self.serve_results = None

    def healthy(self):
        return time.time() - self.start_time < self.run_secs

    def read_buff(self):
        if not self.serve_results:
            self.serve_results = copy.deepcopy(self.fake_results)
        return self.serve_results.pop()

    def stop(self):
        return


class UtilsTestCase(unittest.TestCase):
    def test_arg_parser(self):
        self.assertTrue(argument_parser())

    def test_run_waterfall(self):
        with tempfile.TemporaryDirectory() as tempdir:
            savefig = os.path.join(tempdir, "test.png")
            zmqr = FakeZmqReceiver(90)
            waterfall(
                1e6,  # args.min_freq,
                2e6,  # args.max_freq,
                True,  # args.plot_snr,
                1,  # args.n_detect,
                256,  # args.nfft,
                1e6,  # args.sampling_rate,
                tempdir,  # args.save_path,
                1,  # args.save_time,
                "wideband",  # detection_type,
                "agg",  # engine,
                savefig,  # savefig_path,
                60,  # args.rotate_secs,
                zmqr,
            )
            self.assertTrue(os.path.exists(savefig))
            for dump_match in ("*json", "*csv", "waterfall*png"):
                self.assertTrue(
                    [p for p in glob.glob(os.path.join(tempdir, "*/*/" + dump_match))],
                    dump_match,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
