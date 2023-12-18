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
    filename = options.filename
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
        scan_args.update(
            {
                "iqtlabs": iqtlabs,
                "freq_end": 0,
                "freq_start": freq_start,
                "inference_output_dir": getattr(
                    options, "inference_output_dir", out_dir
                ),
                "pretune": True,
                "samp_rate": int(meta["sample_rate"]),
                "sample_dir": getattr(options, "sample_dir", out_dir),
                "sdr": "file:" + filename,
            }
        )
        tb = grscan(**scan_args)
        tb.start()
        while not tb.sources[0].complete():
            time.sleep(1)
        tb.stop()
        tb.wait()
