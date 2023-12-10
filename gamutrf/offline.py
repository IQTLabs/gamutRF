#!/usr/bin/python3
from argparse import ArgumentParser, BooleanOptionalAction
import os
import logging
import tempfile
import time

from gnuradio import iqtlabs
from gamutrf.grscan import grscan
from gamutrf.sample_reader import get_samples


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument("filename", type=str, help="Recording filename")
    parser.add_argument(
        "--tune-step-fft",
        dest="tune_step_fft",
        type=int,
        default=512,
        help="tune FFT step [default=%(default)r]",
    )
    parser.add_argument(
        "--vkfft",
        dest="vkfft",
        default=True,
        action=BooleanOptionalAction,
        help="use VkFFT",
    )
    parser.add_argument(
        "--db_clamp_floor",
        dest="db_clamp_floor",
        type=float,
        default=-150,
        help="clamp dB output floor",
    )
    parser.add_argument(
        "--db_clamp_ceil",
        dest="db_clamp_ceil",
        type=float,
        default=50,
        help="clamp dB output ceil",
    )
    parser.add_argument(
        "--nfft",
        dest="nfft",
        type=int,
        default=1024,
        help="FFTI size [default=%(default)r]",
    )
    parser.add_argument(
        "--n_image",
        dest="n_image",
        type=int,
        default=10,
        help="if > 1, only log 1/n images",
    )
    return parser


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
    options = argument_parser().parse_args()
    filename = options.filename
    out_dir = os.path.dirname(filename)
    _data_filename, _samples, meta = get_samples(filename)
    freq_start = int(meta["center_frequency"] - (meta["sample_rate"] / 2))
    tb = grscan(
        db_clamp_floor=-1e9,
        fft_batch_size=256,
        freq_end=0,
        freq_start=freq_start,
        inference_min_db=-1e9,
        inference_output_dir=out_dir,
        iqtlabs=iqtlabs,
        n_image=options.n_image,
        nfft=options.nfft,
        pretune=True,
        samp_rate=int(meta["sample_rate"]),
        sample_dir=out_dir,
        sdr="file:" + filename,
        tune_step_fft=options.tune_step_fft,
        vkfft=options.vkfft,
    )
    tb.start()
    while not tb.sources[0].complete():
        time.sleep(1)
    tb.stop()
    tb.wait()
