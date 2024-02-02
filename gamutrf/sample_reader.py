#!/usr/bin/env python3

import re
import os
import gzip
import sigmf
import zstandard
import numpy as np
from gamutrf.utils import SAMPLE_DTYPES, SAMPLE_FILENAME_RE, is_fft

POINTS_RE = re.compile(r"^.+\D([0-9]+)points_.+$")


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
    timestamp = None
    nfft = None
    center_frequency = None
    sample_rate = None
    sample_type = None

    # FFT is always float not matter the original sample type.
    if is_fft(filename):
        sample_type = "raw"
        match = POINTS_RE.match(filename)
        if match is not None:
            nfft = int(match.group(1))

    match = SAMPLE_FILENAME_RE.match(filename)
    if match is not None:
        timestamp, center_frequency, sample_rate, sample_type = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            match.group(4),
        )

    sample_dtype, sample_type = SAMPLE_DTYPES.get(sample_type, (None, None))
    sample_bits = None
    sample_len = None
    if sample_dtype:
        if is_fft(filename):
            sample_dtype = np.dtype([("i", np.float32), ("q", np.float32)])
        else:
            sample_dtype = np.dtype([("i", sample_dtype), ("q", sample_dtype)])
        sample_bits = sample_dtype[0].itemsize * 8
        sample_len = sample_dtype[0].itemsize * 2
    meta = {
        "filename": filename,
        "center_frequency": center_frequency,
        "sample_rate": sample_rate,
        "sample_dtype": sample_dtype,
        "sample_len": sample_len,
        "sample_type": sample_type,
        "sample_bits": sample_bits,
        "nfft": nfft,
        "timestamp": timestamp,
    }
    return meta


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
    samples = None
    for samples_buffer in read_recording(
        filename,
        meta["sample_rate"],
        meta["sample_dtype"],
        meta["sample_len"],
        max_sample_secs=None,
    ):
        if samples is None:
            samples = samples_buffer
        else:
            samples = np.concatenate([samples, samples_buffer])
    return filename, samples, meta


def get_samples(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    meta_ext = filename.find(".sigmf-meta")
    if meta_ext == -1:
        return get_nosigmf_samples(filename)

    meta = sigmf.sigmffile.fromfile(filename)
    data_filename = filename[:meta_ext]
    if os.path.splitext(data_filename)[-1] != ".sigmf-data":
        data_filename = data_filename + ".sigmf-data"
    if not os.path.exists(data_filename):
        raise FileNotFoundError(data_filename)

    meta.set_data_file(data_filename)
    samples = meta.read_samples()
    global_meta = meta.get_global_info()
    captures_meta = meta.get_captures()
    center_frequency = None
    timestamp = None
    if captures_meta:
        center_frequency = captures_meta[0].get("core:frequency", None)
        timestamp = captures_meta[0].get("core:datetime", None)
        if timestamp is not None:
            TS_RE = re.compile(r"^([^\.]+)([\.\d]+)Z$")
            ts_match = TS_RE.match(timestamp)
            if ts_match is None:
                print(
                    "invalid SigMF timestamp (does not match %s): %s"
                    % (TS_RE, timestamp)
                )
                timestamp = None
            else:
                secs_str, remainder = ts_match.group(1), float(ts_match.group(2))
                remainder_str = ("%.3f" % remainder)[1:]
                try:
                    timestamp = sigmf.utils.parse_iso8601_datetime(
                        secs_str + remainder_str + "Z"
                    ).timestamp()
                except ValueError as e:
                    print("invalid SigMF timestamp: %s" % e)
    if timestamp is None:
        filename_meta = parse_filename(data_filename)
        timestamp = filename_meta.get("timestamp", timestamp)
    if timestamp is None:
        print("warning: no SigMF or filename timestamp available, using ctime")
        timestamp = os.stat(data_filename).st_ctime
    meta = {
        "sample_rate": global_meta["core:sample_rate"],
        "sample_dtype": global_meta["core:datatype"],
        "sample_len": samples[
            0
        ].itemsize,  # read_samples() always converts to host cf32.
        "center_frequency": center_frequency,
        "timestamp": timestamp,
    }
    return data_filename, samples, meta
