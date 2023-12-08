#!/usr/bin/env python3

import re
import os
import gzip
import sigmf
import zstandard
import numpy as np
from gamutrf.utils import SAMPLE_DTYPES, SAMPLE_FILENAME_RE, is_fft

FFT_FILENAME_RE = re.compile(
    r"^.+_([0-9]+)_([0-9]+)points_([0-9]+)Hz_([0-9]+)sps\.(s\d+|raw).*$"
)


def get_reader(filename):
    # nosemgrep:github.workflows.config.useless-inner-function
    def gzip_reader(x):
        return gzip.open(x, "rb")

    # nosemgrep:github.workflows.config.useless-inner-function
    def zst_reader(x):
        return zstandard.ZstdDecompressor().stream_reader(
            open(x, "rb"), read_across_frames=True
        )

    def default_reader(x):
        return open(x, "rb")

    if filename.endswith(".gz"):
        return gzip_reader
    if filename.endswith(".zst"):
        return zst_reader

    return default_reader


def parse_filename(filename):
    # FFT is always float not matter the original sample type.
    if is_fft(filename):
        sample_type = "raw"
        match = FFT_FILENAME_RE.match(filename)
        try:
            timestamp = int(match.group(1))
            nfft = int(match.group(2))
            freq_center = int(match.group(3))
            sample_rate = int(match.group(4))
            # sample_type = match.group(3)
        except AttributeError:
            timestamp = None
            nfft = None
            freq_center = None
            sample_rate = None
            sample_type = None
    else:
        match = SAMPLE_FILENAME_RE.match(filename)
        nfft = None
        try:
            timestamp = int(match.group(1))
            freq_center = int(match.group(2))
            sample_rate = int(match.group(3))
            sample_type = match.group(4)
        except AttributeError:
            timestamp = None
            freq_center = None
            sample_rate = None
            sample_type = None

    sample_dtype, sample_type = SAMPLE_DTYPES.get(sample_type, (None, None))
    sample_bits = None
    sample_len = None
    if sample_dtype:
        if is_fft(filename):
            sample_dtype = np.float32
            sample_bits = 32
            sample_len = 4
        else:
            sample_dtype = np.dtype([("i", sample_dtype), ("q", sample_dtype)])
            sample_bits = sample_dtype[0].itemsize * 8
            sample_len = sample_dtype[0].itemsize * 2
    file_info = {
        "filename": filename,
        "freq_center": freq_center,
        "sample_rate": sample_rate,
        "sample_dtype": sample_dtype,
        "sample_len": sample_len,
        "sample_type": sample_type,
        "sample_bits": sample_bits,
        "nfft": nfft,
        "timestamp": timestamp,
    }
    return file_info


def read_recording(
    filename,
    sample_rate,
    sample_dtype,
    sample_len,
    sample_secs=1.0,
    skip_sample_secs=0,
    max_sample_secs=0,
):
    """Read an I/Q recording and iterate over it, returning 1-D numpy arrays of csingles, of size sample_rate * sample_secs.

    Args:
        filename: str, recording to read.
        sample_rate: int, samples per second
        sample_dtype: numpy.dtype, binary format of original I/Q recording.
        sample_len: int, length of one sample.
        sample_secs: float, number of seconds worth of samples per iteration.
        skip_sample_secs: float, number of seconds worth of samples to skip initially.
        max_sample_secs: float, maximum number of seconds of samples to read (or None for all).
    Returns:
        numpy arrays of csingles.
    """
    read_size = int(sample_rate * sample_secs) * sample_len
    reader = get_reader(filename)
    samples_remaining = 0
    if max_sample_secs:
        samples_remaining = int(sample_rate * max_sample_secs)
    with reader(filename) as infile:
        if skip_sample_secs:
            infile.seek(int(sample_rate * skip_sample_secs) * sample_len)
        while True:
            if max_sample_secs and not samples_remaining:
                break
            sample_buffer = infile.read(read_size)
            buffered_samples = int(len(sample_buffer) / sample_len)
            if buffered_samples == 0:
                break
            if max_sample_secs:
                if buffered_samples <= samples_remaining:
                    samples_remaining -= buffered_samples
                else:
                    buffered_samples = samples_remaining
                    samples_remaining = 0
            x1d = np.frombuffer(
                sample_buffer, dtype=sample_dtype, count=buffered_samples
            )
            yield x1d["i"] + np.csingle(1j) * x1d["q"]


def get_nosigmf_samples(filename):
    meta = parse_filename(filename)
    sample_rate = meta["sample_rate"]
    sample_dtype = meta["sample_dtype"]
    sample_len = meta["sample_len"]
    center_frequency = meta["freq_center"]
    samples = None
    for samples_buffer in read_recording(
        filename, sample_rate, sample_dtype, sample_len, max_sample_secs=None
    ):
        if samples is None:
            samples = samples_buffer
        else:
            samples = np.concatenate([samples, samples_buffer])
    return filename, samples, center_frequency


def get_samples(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    meta_ext = filename.find(".sigmf-meta")
    if meta_ext == -1:
        return get_nosigmf_samples(filename)

    meta = sigmf.sigmffile.fromfile(filename)
    data_filename = filename[:meta_ext]
    meta.set_data_file(data_filename)
    # read_samples() always converts to host cf32.
    samples = meta.read_samples()
    global_meta = meta.get_global_info()
    sample_rate = global_meta["core:sample_rate"]
    sample_type = global_meta["core:datatype"]
    captures_meta = meta.get_captures()
    center_frequency = None
    if captures_meta:
        center_frequency = captures_meta[0].get("core:frequency", None)
    return data_filename, samples, center_frequency
