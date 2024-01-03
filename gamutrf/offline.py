#!/usr/bin/python3
import glob
import os
import logging
import time

from gnuradio import iqtlabs
from gamutrf.scan import argument_parser, DYNAMIC_EXCLUDE_OPTIONS
from gamutrf.grscan import grscan
from gamutrf.sample_reader import get_samples


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
    parser = argument_parser()
    parser.add_argument("filename", type=str, help="Recording filename (or glob)")
    options = parser.parse_args()
    for filename in glob.glob(options.filename):
        out_dir = os.path.dirname(filename)
        if out_dir == "":
            out_dir = "."
        _data_filename, _samples, meta = get_samples(filename)
        freq_start = int(meta["center_frequency"] - (meta["sample_rate"] / 2))
        scan_args = {
            k: getattr(options, k)
            for k in dir(options)
            if not k.startswith("_")
            and k != "filename"
            and k not in DYNAMIC_EXCLUDE_OPTIONS
        }
        for override_dir in ("inference_output_dir", "sample_dir"):
            override_val = getattr(options, override_dir)
            if not override_val:
                override_val = out_dir
            scan_args[override_dir] = override_val
        scan_args.update(
            {
                "iqtlabs": iqtlabs,
                "freq_end": 0,
                "freq_start": freq_start,
                "samp_rate": int(meta["sample_rate"]),
                "sdr": "file:" + filename,
                "pretune": True,
                "fft_batch_size": 1,
            }
        )
        tb = grscan(**scan_args)
        tb.start()
        tb.wait()
        tb.stop()
