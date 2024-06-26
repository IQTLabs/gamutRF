import datetime
import json
import logging
import os
import subprocess
import tempfile
import time
from urllib.parse import urlparse

import sigmf

from gamutrf.sigwindows import freq_excluded
from gamutrf.sigwindows import parse_freq_excluded
from gamutrf.utils import (
    ETTUS_ANT,
    ETTUS_ARGS,
    endianstr,
)

IMSHOW_INTERPOLATION = os.getenv("IMSHOW_INTERPOLATION", "bilinear")
NFFT_OVERLAP = 512
SAMPLE_TYPE = "i16"
MIN_SAMPLE_RATE = int(1e6)
MAX_SAMPLE_RATE = int(30 * 1e6)


class SDRRecorder:
    def __init__(self, sdrargs, rotate_secs):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.zst_fifo = os.path.join(self.tmpdir.name, "zstfifo")
        self.sdrargs = sdrargs
        self.rotate_secs = rotate_secs
        os.mkfifo(self.zst_fifo)

    @staticmethod
    def validate_request(freqs_excluded, center_freq, sample_count, sample_rate):
        for arg in [center_freq, sample_count, sample_rate]:
            try:
                int(float(arg))
            except ValueError:
                return "Invalid values in request"
        if freq_excluded(int(center_freq), parse_freq_excluded(freqs_excluded)):
            return "Requested frequency is excluded"
        if int(sample_rate) < MIN_SAMPLE_RATE or int(sample_rate) > MAX_SAMPLE_RATE:
            return "sample rate {sample_rate} out of range {MIN_SAMPLE_RATE} to {MAX_SAMPLE_RATE}"
        duration_sec = int(sample_count) / int(sample_rate)
        if duration_sec < 1:
            return "cannot record for less than 1 second"
        return None

    @staticmethod
    def get_sample_file(path, epoch_time, center_freq, sample_rate, sdr, antenna, gain):
        return os.path.join(
            path,
            f"gamutrf_recording_{sdr}_{antenna}_gain{gain}_{epoch_time}_{int(center_freq)}Hz_{int(sample_rate)}sps.{SAMPLE_TYPE}.zst",
        )

    def record_args(
        self, sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb
    ):
        raise NotImplementedError

    def write_recording(
        self, sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb
    ):
        record_status = -1
        args = self.record_args(
            self.zst_fifo, sample_rate, sample_count, center_freq, gain, agc, rxb
        )
        dotfile = os.path.join(
            os.path.dirname(sample_file), "." + os.path.basename(sample_file)
        )
        zstd_args = ["nice", "zstd", "-1", self.zst_fifo, "-o", dotfile]
        logging.info("starting recording: %s", args)
        with subprocess.Popen(zstd_args, stdin=subprocess.DEVNULL) as zstd_proc:
            record_status = subprocess.check_call(args)
            zstd_proc.communicate()
        if os.path.exists(dotfile):
            os.rename(dotfile, sample_file)
        return record_status

    def run_recording(
        self,
        path,
        sample_rate,
        sample_count,
        center_freq,
        gain,
        agc,
        rxb,
        sigmf_,
        sdr,
        antenna,
    ):
        epoch_time = int(time.time())
        meta_time = datetime.datetime.utcnow().isoformat() + "Z"
        if self.rotate_secs:
            ts_dir = int(epoch_time / self.rotate_secs) * self.rotate_secs
            path = os.path.join(path, str(ts_dir))
            if not os.path.exists(path):
                os.makedirs(path)
        sample_file = self.get_sample_file(
            path, str(epoch_time), center_freq, sample_rate, sdr, antenna, gain
        )
        record_status = -1
        try:
            record_status = self.write_recording(
                sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb
            )
            if sigmf_:
                sigmf_file = sample_file + ".sigmf-meta"
                if os.path.exists(sample_file):
                    if not os.path.exists(sigmf_file):
                        meta = sigmf.SigMFFile(
                            skip_checksum=True,  # expensive for large files
                            # data_file=sample_file, # don't set for ZST, confuses sigmf
                            global_info={
                                sigmf.SigMFFile.DATATYPE_KEY: "_".join(
                                    ("c" + SAMPLE_TYPE, endianstr())
                                ),
                                sigmf.SigMFFile.SAMPLE_RATE_KEY: sample_rate,
                                sigmf.SigMFFile.VERSION_KEY: sigmf.__version__,
                            },
                        )
                        # TODO: add capture_details, source_file and gain when supported.
                        meta.add_capture(
                            0,
                            metadata={
                                sigmf.SigMFFile.FREQUENCY_KEY: center_freq,
                                sigmf.SigMFFile.DATETIME_KEY: meta_time,
                            },
                        )
                        meta.tofile(sigmf_file)
                else:
                    logging.error("{sample_file} missing, cannot write sigmf file")
        except subprocess.CalledProcessError as err:
            logging.error("record failed: %s", err)
        logging.info("record status: %d", record_status)
        return (record_status, sample_file)


class EttusRecorder(SDRRecorder):
    def __init__(self, sdrargs, rotate_secs):
        super().__init__(sdrargs, rotate_secs)
        self.worker_subprocess = None
        self.last_worker_line = None
        if not self.sdrargs:
            self.sdrargs = ETTUS_ARGS
        try:
            subprocess.check_call(
                [
                    "/usr/local/bin/uhd_sample_recorder",
                    "--rate",
                    str(1e6),
                    "--args",
                    self.sdrargs,
                    "--ant",
                    ETTUS_ANT,
                    "--duration",
                    "1",
                    "--null",
                    "--fftnull",
                    "--novkfft",
                    "--nfft",
                    "0",
                ]
            )
        except subprocess.CalledProcessError:
            raise ValueError

    def record_args(
        self, sample_file, sample_rate, sample_count, center_freq, gain, _agc, rxb
    ):
        # Ettus "nsamps" API has an internal limit, so translate "stream for druation".
        duration = round(sample_count / sample_rate)
        rxb = min(rxb, sample_rate)

        args = [
            "/usr/local/bin/uhd_sample_recorder",
            "--json",
            "--rate",
            str(sample_rate),
            "--bw",
            str(sample_rate),
            "--gain",
            str(gain),
            "--args",
            self.sdrargs,
            "--ant",
            ETTUS_ANT,
            "--spb",
            str(rxb),
        ]

        json_args = {
            "file": sample_file,
            "duration": duration,
            "freq": center_freq,
        }
        return (args, json_args)

    def write_recording(
        self, sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb
    ):
        # Ettus doesn't need a wrapper, it can do its own zst compression.
        record_status = -1
        args, json_args = self.record_args(
            sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb
        )

        try:

            def worker_read():
                self.last_worker_line = (
                    self.worker_subprocess.stdout.readline().decode("utf-8").strip()
                )
                logging.info(self.last_worker_line)

            def worker_write(s):
                self.worker_subprocess.stdin.write(("%s\n" % s).encode("utf-8"))
                self.worker_subprocess.stdin.flush()

            if self.worker_subprocess is None:
                logging.info("starting worker subprocess: %s", args)
                self.worker_subprocess = subprocess.Popen(
                    args, stdin=subprocess.PIPE, stdout=subprocess.PIPE
                )
                worker_read()

            logging.info("starting recording: %s", json_args)
            worker_write(json.dumps(json_args))
            worker_read()
            last_error = json.loads(self.last_worker_line).get("last_error")
            if not last_error:
                record_status = 0
        except (
            subprocess.SubprocessError,
            BrokenPipeError,
            json.decoder.JSONDecodeError,
        ) as e:
            logging.error(e)
            if self.worker_subprocess:
                self.worker_subprocess.kill()
                self.worker_subprocess = None
        return record_status


class BladeRecorder(SDRRecorder):
    def record_args(
        self, sample_file, sample_rate, sample_count, center_freq, gain, agc, _rxb
    ):
        gain_args = [
            "-e",
            "set agc rx off",
            "-e",
            f"set gain rx {gain}",
        ]
        if agc:
            gain_args = [
                "-e",
                "set agc rx on",
            ]
        return (
            ["bladeRF-cli"]
            + gain_args
            + [
                "-e",
                f"set samplerate rx {sample_rate}",
                "-e",
                f"set bandwidth rx {sample_rate}",
                "-e",
                f"set frequency rx {center_freq}",
                "-e",
                f"rx config file={sample_file} format=bin n={sample_count}",
                "-e",
                "rx start",
                "-e",
                "rx wait",
            ]
        )


class LimeRecorder(SDRRecorder):
    def record_args(
        self, sample_file, sample_rate, sample_count, center_freq, gain, agc, _rxb
    ):
        gain_args = []
        if gain:
            gain_args = [
                "-g",
                f"{gain}",
            ]
        return (
            ["/usr/local/bin/LimeStream"]
            + gain_args
            + [
                "-f",
                f"{center_freq}",
                "-s",
                f"{sample_rate}",
                "-C",
                f"{sample_count}",
                "-r",
                f"{sample_file}",
            ]
        )


class FileTestRecorder(SDRRecorder):
    def record_args(
        self, sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb
    ):
        args = [
            "dd",
            f"if={urlparse(self.sdrargs).path}",
            f"of={sample_file}",
            f"count={int(sample_count)}",
            f"bs={int(sample_rate)}",
        ]
        return args


RECORDER_MAP = {
    "ettus": EttusRecorder,
    "bladerf": BladeRecorder,
    "lime": LimeRecorder,
}


def get_recorder(recorder_name, sdrargs=None, rotate_secs=0):
    try:
        return RECORDER_MAP[recorder_name](sdrargs, rotate_secs)
    except KeyError:
        return FileTestRecorder(recorder_name, rotate_secs)
