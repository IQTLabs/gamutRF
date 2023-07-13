#!/usr/bin/env python3

import gzip
import zstandard
import numpy as np


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
