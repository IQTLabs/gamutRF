#!/usr/bin/python3
import copy
import csv
import logging
import glob
import os
import tempfile
import time
import unittest
import pandas as pd
from gamutrf.waterfall import argument_parser, waterfall
from gamutrf.peak_finder import get_peak_finder


class FakeZmqReceiver:
    def __init__(self, run_secs, peak_min, peak_max, peak_val):
        self.start_time = time.time()
        self.run_secs = run_secs
        df = pd.DataFrame(
            [
                {"ts": 1.0, "freq": 1 + (i * 0.001), "db": peak_val / 2}
                for i in range(1000)
            ]
        )
        df.loc[(df.freq >= peak_min) & (df.freq <= peak_max), "db"] = peak_val
        self.fake_results = [
            ({}, df),
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
            peak_min = 1.50
            peak_max = 1.51
            peak_val = 100
            savefig = os.path.join(tempdir, "test.png")
            zmqr = FakeZmqReceiver(90, peak_min, peak_max, peak_val)
            peak_finder = get_peak_finder("narrowband")
            waterfall(
                1e6,  # args.min_freq,
                2e6,  # args.max_freq,
                True,  # args.plot_snr,
                1,  # args.n_detect,
                256,  # args.nfft,
                1e6,  # args.sampling_rate,
                tempdir,  # args.save_path,
                1,  # args.save_time,
                peak_finder,
                "agg",  # engine,
                savefig,  # savefig_path,
                60,  # args.rotate_secs,
                10,  # args.width,
                5,  # args.height,
                True,  # args.batch
                zmqr,
            )
            self.assertTrue(os.path.exists(savefig))
            for dump_match in ("*json", "*csv", "waterfall*png"):
                self.assertTrue(
                    [p for p in glob.glob(os.path.join(tempdir, "*/*/" + dump_match))],
                    dump_match,
                )
            detections_files = [
                p for p in glob.glob(os.path.join(tempdir, "*/*/detections*csv"))
            ]
            self.assertTrue(detections_files)
            for f in detections_files:
                with open(f) as csv_file:
                    for row in csv.DictReader(csv_file):
                        # timestamp,start_freq,end_freq,dB,type
                        # 1.0,1.49609375,1.5078125,100.0,narrowband
                        self.assertEqual(
                            peak_min, round(float(row["start_freq"]), 2), row
                        )
                        self.assertEqual(
                            peak_max, round(float(row["end_freq"]), 2), row
                        )
                        self.assertEqual(peak_val, float(row["dB"]), row)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
