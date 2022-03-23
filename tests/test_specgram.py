#!/usr/bin/python3

import os
import tempfile
import time
import unittest

from gamutrf.utils import replace_ext, parse_filename
from gamutrf.specgram import read_recording, plot_spectrogram


class SpecgramTestCase(unittest.TestCase):

    def test_process_fft_lines(self):
        with tempfile.TemporaryDirectory() as tempdir:
            recording = os.path.join(str(tempdir), 'testrecording_100Hz_1000sps.raw')
            freq_center, sample_rate, sample_dtype, sample_len, _sample_type, _sample_bits = parse_filename(recording)
            samples = chr(0) * int(sample_len * sample_rate)
            with open(recording, 'wb') as f:
                f.write(samples.encode('utf8'))
            samples = read_recording(recording, len(samples), sample_dtype, sample_len)
            plot_spectrogram(
                samples,
                replace_ext(recording, 'jpg'),
                256,
                sample_rate,
                freq_center,
                'turbo',
                10,
                True)


if __name__ == '__main__':
    unittest.main()
