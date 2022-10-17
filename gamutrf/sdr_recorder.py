import datetime
import json
import logging
import os
import subprocess
import tempfile
import time
from urllib.parse import urlparse

import sigmf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from gamutrf.sigwindows import freq_excluded
from gamutrf.sigwindows import parse_freq_excluded
from gamutrf.utils import (
    ETTUS_ANT,
    ETTUS_ARGS,
    MPL_BACKEND,
    WIDTH,
    HEIGHT,
    DPI,
    DS_PIXELS,
)

IMSHOW_INTERPOLATION = os.getenv("IMSHOW_INTERPOLATION", "bilinear")
NFFT = int(os.getenv("NFFT", "0"))
NFFT_OVERLAP = 512
SAMPLE_TYPE = "s16"
MIN_SAMPLE_RATE = int(1e6)
MAX_SAMPLE_RATE = int(30 * 1e6)
FFT_FILE = "/dev/shm/fft.dat"  # nosec


class SDRRecorder:
    def __init__(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.zst_fifo = os.path.join(self.tmpdir.name, "zstfifo")
        os.mkfifo(self.zst_fifo)

    @staticmethod
    def validate_request(freqs_excluded, center_freq, sample_count, sample_rate):
        for arg in [center_freq, sample_count, sample_rate]:
            try:
                int(float(arg))
            except ValueError:
                return "Invalid values in request"
        if freq_excluded(center_freq, parse_freq_excluded(freqs_excluded)):
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

    @staticmethod
    def fft_spectrogram(
        sample_file, fft_file, sample_count, sample_rate, center_freq, nfft
    ):
        def matplotlib_gc(fig, axes, im):
            axes.images.remove(im)
            fig.clear()
            plt.close()
            plt.cla()
            plt.clf()

        def get_fig(width, height):
            fig = plt.figure()
            fig.set_size_inches(width, height)
            axes = fig.add_subplot(111)
            return (fig, axes)

        def imshow(axes, i, extent):
            return axes.imshow(
                i,
                cmap="turbo",
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation=IMSHOW_INTERPOLATION,
            )

        def plot_fft(i, sample_file, extent):
            png_file = sample_file.replace(".zst", ".png")
            logging.info("generating spectrogram: %s", png_file)
            fig, axes = get_fig(WIDTH, HEIGHT)
            axes.set_xlabel("time (s)")
            axes.set_ylabel("freq (MHz)")
            axes.axis("auto")
            axes.minorticks_on()
            im = imshow(axes, i, extent)
            plt.colorbar(im, ax=axes)
            plt.sci(im)
            plt.savefig(png_file, dpi=DPI)
            matplotlib_gc(fig, axes, im)

        def plot_ds_fft(i, sample_file, extent):
            ds_png_file = sample_file.replace(".zst", ".ds.png")
            logging.info("generating downsampled spectrogram: %s", ds_png_file)
            fig, axes = get_fig(DS_PIXELS / DPI, DS_PIXELS / DPI)
            axes.set_axis_off()
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
            im = imshow(axes, i, extent)
            plt.sci(im)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(ds_png_file, dpi=DPI)
            matplotlib_gc(fig, axes, im)

        if os.path.exists(fft_file):
            matplotlib.use(MPL_BACKEND)
            i = np.memmap(fft_file, dtype=np.float32, mode="r+")
            i = np.roll(i.reshape(-1, nfft).swapaxes(0, 1), int(nfft / 2), 0)
            fc = center_freq / 1e6
            fo = sample_rate / 1e6 / 2
            extent = (0, sample_count / sample_rate, fc - fo, fc + fo)
            plot_fft(i, sample_file, extent)
            plot_ds_fft(i, sample_file, extent)
            os.remove(fft_file)

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
        epoch_time = str(int(time.time()))
        meta_time = datetime.datetime.utcnow().isoformat() + "Z"
        sample_file = self.get_sample_file(
            path, epoch_time, center_freq, sample_rate, sdr, antenna, gain
        )
        record_status = -1
        try:
            record_status = self.write_recording(
                sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb
            )
            if NFFT:
                self.fft_spectrogram(
                    sample_file, FFT_FILE, sample_count, sample_rate, center_freq, NFFT
                )

            if sigmf_:
                meta = sigmf.SigMFFile(
                    data_file=sample_file,
                    global_info={
                        sigmf.SigMFFile.DATATYPE_KEY: SAMPLE_TYPE,
                        sigmf.SigMFFile.SAMPLE_RATE_KEY: sample_rate,
                        sigmf.SigMFFile.VERSION_KEY: sigmf.__version__,
                    },
                )
                meta.add_capture(
                    0,
                    metadata={
                        sigmf.SigMFFile.FREQUENCY_KEY: center_freq,
                        sigmf.SigMFFile.DATETIME_KEY: meta_time,
                    },
                )
                meta.tofile(sample_file + ".sigmf-meta")
        except subprocess.CalledProcessError as err:
            logging.debug("record failed: %s", err)
        logging.info("record status: %d", record_status)
        return (record_status, sample_file)


class EttusRecorder(SDRRecorder):
    def __init__(self):
        super().__init__()
        self.worker_subprocess = None
        self.last_worker_line = None
        # TODO: troubleshoot why this doesn't find an Ettus initially, still.
        # subprocess.call(['uhd_find_devices'])

    def record_args(
        self, sample_file, sample_rate, sample_count, center_freq, gain, _agc, rxb
    ):
        # Ettus "nsamps" API has an internal limit, so translate "stream for druation".
        duration = round(sample_count / sample_rate)
        rxb = min(rxb, sample_rate)

        args = [
            "/usr/local/bin/mt_rx_samples_to_file",
            "--json",
            "--rate",
            str(sample_rate),
            "--bw",
            str(sample_rate),
            "--gain",
            str(gain),
            "--args",
            ETTUS_ARGS,
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
        if NFFT:
            json_args.update(
                {
                    "nfft": NFFT,
                    "nfft_overlap": NFFT_OVERLAP,
                    "fft_file": FFT_FILE,
                }
            )
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

    test_file = None

    def record_args(
        self, sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb
    ):
        args = [
            "dd",
            f"if={urlparse(self.test_file).path}",
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


def get_recorder(recorder_name):
    try:
        return RECORDER_MAP[recorder_name]()
    except KeyError:
        recorder = FileTestRecorder
        recorder.test_file = recorder_name
        return recorder
