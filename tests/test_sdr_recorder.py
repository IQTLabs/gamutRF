#!/usr/bin/python3

import os
import subprocess
import tempfile
import unittest
from gamutrf.sdr_recorder import SDRRecorder


class TestRecorder(SDRRecorder):

    def record_args(self, sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb):
        args = ['dd', 'if=/dev/zero', f'of={sample_file}', f'count={int(sample_count)}', f'bs={int(sample_rate)}']
        return args


class SDRRecorderTestCase(unittest.TestCase):

    def test_sdr_recorder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sdr_recorder = TestRecorder()
            # TODO: sigmf 1.0.0 can't parse .zst files, but it can write the metadata fine.
            record_status, sample_file = sdr_recorder.run_recording(tmpdir, 1e3, 1e3, 1e3, 0, False, 0, sigmf_=True)
            self.assertTrue(os.path.exists(sample_file))
            self.assertGreater(os.path.getsize(sample_file), 0)
            self.assertTrue(os.path.exists(sample_file + '.sigmf-meta'))
            sdr_recorder.tmpdir.cleanup()


if __name__ == '__main__':
    unittest.main()
