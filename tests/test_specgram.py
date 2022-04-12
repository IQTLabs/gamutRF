#!/usr/bin/python3

import os
import tempfile
import time
import unittest

from gamutrf.specgram import process_all_recordings
from gamutrf.utils import parse_filename

class FakeArgs:

    def __init__(self, nfft, cmap, ytics, bare, noverlap, iext, skip_test, recording, workers):
        self.nfft = nfft
        self.cmap = cmap
        self.ytics = ytics
        self.bare = bare
        self.noverlap = noverlap
        self.iext = iext
        self.skip_exist = True
        self.recording = recording
        self.workers = workers


class SpecgramTestCase(unittest.TestCase):

    def test_process_fft_lines(self):
        with tempfile.TemporaryDirectory() as tempdir:
            recording = os.path.join(str(tempdir), 'testrecording_100Hz_1000sps.raw')
            freq_center, sample_rate, sample_dtype, sample_len, _sample_type, _sample_bits = parse_filename(recording)
            samples = chr(0) * int(sample_len * sample_rate)
            with open(recording, 'wb') as f:
                f.write(samples.encode('utf8'))
            fakeargs = FakeArgs(256, 'turbo', 1, True, 0, 'png', False, recording, 2)
            process_all_recordings(fakeargs)
            fakeargs.workers = 1
            fakeargs.skip_exist = False
            process_all_recordings(fakeargs)


if __name__ == '__main__':
    unittest.main()
