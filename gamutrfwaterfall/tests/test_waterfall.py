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
from gamutrfwaterfall.argparser import argument_parser
from gamutrfwaterfall import waterfall
from gamutrflib.peak_finder import get_peak_finder


class FakeZmqReceiver:
    def __init__(self, run_secs, peak_min, peak_max, peak_val):
        self.start_time = time.time()
        self.run_secs = run_secs
        self.serve_results = None
        self.peak_min = peak_min
        self.peak_max = peak_max
        self.peak_val = peak_val

    def healthy(self):
        return time.time() - self.start_time < self.run_secs

    def read_buff(self, scan_fres):
        if not self.serve_results:
            freq_min = 1e6
            freq_max = 10e6
            rows = int((freq_max - freq_min) / scan_fres)
            df = pd.DataFrame(
                [
                    {
                        "ts": time.time(),
                        "freq": freq_min + (i * scan_fres),
                        "db": self.peak_val / 2,
                        "tune_count": i,
                    }
                    for i in range(rows)
                ]
            )
            df.loc[(df.freq >= self.peak_min) & (df.freq <= self.peak_max), "db"] = (
                self.peak_val
            )
            df["freq"] /= 1e6
            self.serve_results = [
                (
                    [
                        {
                            "sample_rate": 1e6,
                            "nfft": 256,
                            "freq_start": df.freq.min(),
                            "freq_end": df.freq.max(),
                            "tune_step_hz": scan_fres,
                            "tune_step_fft": 256,
                        }
                    ],
                    df,
                ),
                (None, None),
            ]
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
                tempdir,  # args.save_path,
                1,  # args.save_time,
                peak_finder,
                "agg",  # engine,
                savefig,  # savefig_path,
                60,  # args.rotate_secs,
                10,  # args.width,
                5,  # args.height,
                10,  # args.waterfall_height,
                100,  # args.waterfall_width,
                5,  # args.refresh,
                True,  # args.batch
                zmqr,
                None,  # api_endpoint
                None,  # config_vars
                None,  # config_vars_path
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
                            round(peak_min, 1), round(float(row["start_freq"]), 1), row
                        )
                        self.assertEqual(
                            round(peak_max, 1), round(float(row["end_freq"]), 1), row
                        )
                        self.assertEqual(peak_val, float(row["dB"]), row)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
