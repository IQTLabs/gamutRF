#!/usr/bin/python3
import os
import subprocess
import tempfile
import unittest

from gamutrf.sdr_recorder import get_recorder


class SDRRecorderTestCase(unittest.TestCase):
    SAMPLES = 1e3 * 4

    def get_file_recorder(self, tmpdir):
        filename = os.path.join(tmpdir, "gamutrf_recording1_1000Hz_1000sps.raw")
        subprocess.check_call(
            ["dd", "if=/dev/zero", "of=" + filename, "bs=1M", "count=1"]
        )
        return get_recorder("file:" + filename, 3600)

    def test_sdr_recorder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sdr_recorder = self.get_file_recorder(tmpdir)
            sample_file = os.path.join(tmpdir, "test_file.zst")
            fft_file = os.path.join(tmpdir, "fft_test_file.zst")
            with open(fft_file, "wb") as f:
                f.write(b"\x00" * 4 * 2048 * 10)
            sdr_recorder.fft_spectrogram(sample_file, fft_file, 2048, 1e6, 1e6, 2048)
        with tempfile.TemporaryDirectory() as tmpdir:
            sdr_recorder = self.get_file_recorder(tmpdir)
            record_status, sample_file = sdr_recorder.run_recording(
                tmpdir,
                self.SAMPLES,
                self.SAMPLES,
                self.SAMPLES,
                0,
                False,
                0,
                sigmf_=False,
                sdr="zero",
                antenna="omni",
            )
            self.assertEqual(0, record_status)
            self.assertTrue(os.path.exists(sample_file))
            sdr_recorder.tmpdir.cleanup()
        with tempfile.TemporaryDirectory() as tmpdir:
            sdr_recorder = self.get_file_recorder(tmpdir)
            # TODO: sigmf 1.0.0 can't parse .zst files, but it can write the metadata fine.
            record_status, sample_file = sdr_recorder.run_recording(
                tmpdir,
                self.SAMPLES,
                self.SAMPLES,
                self.SAMPLES,
                0,
                False,
                0,
                sigmf_=True,
                sdr="zero",
                antenna="directional",
            )
            self.assertTrue(os.path.exists(sample_file))
            self.assertGreater(os.path.getsize(sample_file), 0)
            self.assertTrue(os.path.exists(sample_file + ".sigmf-meta"))
            sdr_recorder.tmpdir.cleanup()

        with tempfile.TemporaryDirectory() as tmpdir:
            sdr_recorder = self.get_file_recorder(tmpdir)
            self.assertNotEqual(None, sdr_recorder.validate_request([], 1e6, 0, 0))
            self.assertNotEqual(None, sdr_recorder.validate_request([], 1e6, 1, 1))
            self.assertEqual(None, sdr_recorder.validate_request([], 1e6, 1e6, 1e6))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
