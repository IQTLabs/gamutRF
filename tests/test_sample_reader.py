#!/usr/bin/python3
import re
import os
import tempfile
import unittest
import numpy as np

from gamutrf.sample_reader import read_recording, parse_filename, get_samples

TEST_META = """
      {
        "global": {
            "core:sample_rate": 20480000.0,
            "core:antenna": "omni",
            "core:gain": 10,
            "core:hw": "",
            "core:license": "",
            "core:author": "",
            "core:description": "FPV Drone - fpv flying",
            "core:version": "0.0.2",
            "core:datatype": "ci16_le"
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": 2450000000.0,
                "core:datetime": "2023-08-02T14:48:21.987701000Z"
            }
        ],
        "annotations": []
    }
"""


class ReadRecordingTestCase(unittest.TestCase):
    def test_get_samples(self):
        with tempfile.TemporaryDirectory() as tempdir:
            meta = os.path.join(str(tempdir), "test.sigmf-meta")
            data = os.path.join(str(tempdir), "test.sigmf-data")
            with open(data, "wb") as f:
                f.write(bytes("\x00\x00\x00\x00", encoding="utf8"))
            with open(meta, "w", encoding="utf8") as f:
                f.write(TEST_META)
            data_filename, _samples, parsed_meta = get_samples(meta)
            self.assertEqual(data, data_filename)
            self.assertEqual(1690987701.988, parsed_meta["timestamp"])

    def test_read_recording(self):
        with tempfile.TemporaryDirectory() as tempdir:
            recording = os.path.join(
                str(tempdir), "testrecording_123_100Hz_1000sps.ci16"
            )
            meta = parse_filename(recording)
            samples = chr(0) * int(meta["sample_len"] * meta["sample_rate"])
            with open(recording, "wb") as f:
                f.write(samples.encode("utf8"))
            self.assertEqual(100, meta["center_frequency"])
            self.assertEqual(1000, meta["sample_rate"])
            self.assertEqual("signed-integer", meta["sample_type"])
            self.assertEqual(16, meta["sample_bits"])
            self.assertEqual(
                np.dtype([("i", "<i2"), ("q", "<i2")]), meta["sample_dtype"]
            )
            # iterate over recording, returning chunks (since this test file has only 1e3 samples in it,
            # we get one chunk with 1e3, csingles.
            for i, sample_chunk in enumerate(
                read_recording(
                    recording,
                    meta["sample_rate"],
                    meta["sample_dtype"],
                    meta["sample_len"],
                )
            ):
                self.assertTrue(isinstance(sample_chunk, np.ndarray))
                self.assertEqual((1000,), sample_chunk.shape)
                self.assertTrue(isinstance(sample_chunk[0], np.csingle))
                self.assertEqual(0, sample_chunk[0])
                self.assertEqual(0, i)

    def test_sjip_read_recording(self):
        with tempfile.TemporaryDirectory() as tempdir:
            recording = os.path.join(
                str(tempdir), "testrecording_123_100Hz_1000sps.ci16"
            )
            meta = parse_filename(recording)
            samples = chr(0) * int(meta["sample_len"] * meta["sample_rate"]) * 3
            with open(recording, "wb") as f:
                f.write(samples.encode("utf8"))
            self.assertEqual(100, meta["center_frequency"])
            self.assertEqual(1000, meta["sample_rate"])
            self.assertEqual("signed-integer", meta["sample_type"])
            self.assertEqual(16, meta["sample_bits"])
            self.assertEqual(
                np.dtype([("i", "<i2"), ("q", "<i2")]), meta["sample_dtype"]
            )
            # iterate over recording, returning chunks (since this test file has 3*1e3 samples in it,
            # skipping the first and last, we get one chunk with 1e3 / 2, csingles.
            for i, sample_chunk in enumerate(
                read_recording(
                    recording,
                    meta["sample_rate"],
                    meta["sample_dtype"],
                    meta["sample_len"],
                    skip_sample_secs=1.0,
                    max_sample_secs=0.5,
                )
            ):
                self.assertTrue(isinstance(sample_chunk, np.ndarray))
                self.assertEqual((1e3 / 2,), sample_chunk.shape)
                self.assertTrue(isinstance(sample_chunk[0], np.csingle))
                self.assertEqual(0, sample_chunk[0])
                self.assertEqual(0, i)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
