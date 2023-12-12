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
        "--scaling",
        dest="scaling",
        type=str,
        default="spectrum",
        help="""Same as --scaling parameter in scipy.signal.spectrogram(). 
        Selects between computing the power spectral density ('density') 
        where `Sxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Sxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'spectrum'.""",
    )
    parser.add_argument(
        "--fft_batch_size",
        dest="fft_batch_size",
        type=int,
        default=256,
        help="offload FFT batch size",
    )
    parser.add_argument(
        "--tuneoverlap",
        dest="tuneoverlap",
        type=float,
        default=0.85,
        help="multiple of samp_rate when retuning",
    )
    parser.add_argument(
        "--bucket_range",
        dest="bucket_range",
        type=float,
        default=0.85,
        help="what proportion of FFT buckets to use",
    )
    parser.add_argument(
        "--db_clamp_floor",
        dest="db_clamp_floor",
        type=float,
        default=-200,
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
        "--dc_block_len",
        dest="dc_block_len",
        type=int,
        default=0,
        help="if > 0, use dc_block_cc filter with length",
    )
    parser.add_argument(
        "--dc_block_long",
        dest="dc_block_long",
        action="store_true",
        help="Use dc_block_cc long form",
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
    parser.add_argument(
        "--inference_min_db",
        dest="inference_min_db",
        type=float,
        default=-150,
        help="run inference over minimum dB power",
    )
    parser.add_argument(
        "--inference_model_server",
        dest="inference_model_server",
        type=str,
        default="",
        help="torchserve model server inference API address (e.g. localhost:1234)",
    )
    parser.add_argument(
        "--inference_model_name",
        dest="inference_model_name",
        type=str,
        default="",
        help="torchserve model name (e.g. yolov8)",
    )
    parser.add_argument(
        "--logaddr",
        dest="logaddr",
        type=str,
        default="0.0.0.0",  # nosec
        help="Log FFT results to this address",
    )
    parser.add_argument(
        "--logport",
        dest="logport",
        type=int,
        default=8001,
        help="Log FFT results to this port",
    )
    return parser


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
    options = argument_parser().parse_args()
    filename = options.filename
    out_dir = os.path.dirname(filename)
    if out_dir == "":
        out_dir = "."
    _data_filename, _samples, meta = get_samples(filename)
    freq_start = int(meta["center_frequency"] - (meta["sample_rate"] / 2))
    scan_args = {
        "iqtlabs": iqtlabs,
        "freq_end": 0,
        "freq_start": freq_start,
        "inference_output_dir": out_dir,
        "pretune": True,
        "samp_rate": int(meta["sample_rate"]),
        "sample_dir": out_dir,
        "sdr": "file:" + filename,
    }
    scan_args.update(
        {
            k: getattr(options, k)
            for k in dir(options)
            if not k.startswith("_") and k != "filename"
        }
    )
    tb = grscan(**scan_args)
    tb.start()
    while not tb.sources[0].complete():
        time.sleep(1)
    tb.stop()
    tb.wait()
