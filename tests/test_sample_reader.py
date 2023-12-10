#!/usr/bin/python3
import os
import tempfile
import unittest
import numpy as np

from gamutrf.sample_reader import read_recording, parse_filename


class ReadRecordingTestCase(unittest.TestCase):
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
