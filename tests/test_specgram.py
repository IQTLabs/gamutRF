#!/usr/bin/python3

import os
import tempfile
import time
import unittest

from gamutrf.specgram import read_recording, plot_spectrogram, replace_ext


class SpecgramTestCase(unittest.TestCase):

    def test_process_fft_lines(self):
        with tempfile.TemporaryDirectory() as tempdir:
            freq_center = 1e2
            sample_rate = 1e3
            recording = os.path.join(str(tempdir), 'test.raw')
            samples = chr(0) * int(4 * sample_rate)
            with open(recording, 'wb') as f:
                f.write(samples.encode('utf8'))
            samples = read_recording(recording)
            plot_spectrogram(
                samples,
                replace_ext(recording, 'jpg'),
                256,
                sample_rate,
                freq_center,
                cmap='twilight_r')


if __name__ == '__main__':
    unittest.main()
