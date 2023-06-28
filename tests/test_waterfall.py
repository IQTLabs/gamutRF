#!/usr/bin/python3
import os
import tempfile
import unittest
import pandas as pd
from gamutrf.waterfall import waterfall


class FakeZmqReceiver:
    def __init__(self):
        self.fake_results = [
            ({}, pd.DataFrame([{"ts": 1, "freq": 1e6, "db": 50}])),
            (None, None),
        ]

    def healthy(self):
        if self.fake_results:
            return True
        return False

    def read_buff(self):
        if self.fake_results:
            result = self.fake_results.pop()
            return result
        return (None, None)

    def stop(self):
        return


class UtilsTestCase(unittest.TestCase):
    def test_run_waterfall(self):
        with tempfile.TemporaryDirectory() as tempdir:
            savefig = os.path.join(tempdir, "test.png")
            zmqr = FakeZmqReceiver()
            waterfall(
                1e6, # args.min_freq,
                2e6, # args.max_freq,
                True, # args.plot_snr,
                1, # args.n_detect,
                256, # args.nfft,
                1e6, # args.sampling_rate,
                tempdir, # args.save_path,
                1, # args.save_time,
                "wideband", # detection_type,
                "agg", # engine,
                savefig, # savefig_path,
                60, # args.rotate_secs,
                zmqr,
            )
            self.assertTrue(os.path.exists(savefig))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
