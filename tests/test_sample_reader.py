#!/usr/bin/python3
import os
import tempfile
import unittest
import numpy as np

from gamutrf.sample_reader import read_recording
from gamutrf.utils import parse_filename


class ReadRecordingTestCase(unittest.TestCase):

    def test_read_recording(self):
        with tempfile.TemporaryDirectory() as tempdir:
            recording = os.path.join(
                str(tempdir), 'testrecording_100Hz_1000sps.s16')
            # parse filename determines sample dtype, et al from filename
            freq_center, sample_rate, sample_dtype, sample_len, sample_type, sample_bits = parse_filename(
                recording)
            samples = chr(0) * int(sample_len * sample_rate)
            with open(recording, 'wb') as f:
                f.write(samples.encode('utf8'))
            self.assertEqual(100, freq_center)
            self.assertEqual(1000, sample_rate)
            self.assertEqual('signed-integer', sample_type)
            self.assertEqual(16, sample_bits)
            self.assertEqual(np.dtype([('i', '<i2'), ('q', '<i2')]), sample_dtype)
            # iterate over recording, returning chunks (since this test file has only 1e3 samples in it,
            # we get one chunk with 1e3, csingles.
            for sample_chunk in read_recording(recording, sample_rate, sample_dtype, sample_len):
                self.assertTrue(isinstance(sample_chunk, np.ndarray))
                self.assertEqual((1000,), sample_chunk.shape)
                self.assertTrue(isinstance(sample_chunk[0], np.csingle))
                self.assertEqual(0, sample_chunk[0])


if __name__ == '__main__':
    unittest.main()
