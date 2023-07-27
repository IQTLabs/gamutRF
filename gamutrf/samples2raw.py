import argparse
import subprocess

from gamutrf.utils import parse_filename
from gamutrf.utils import replace_ext


def make_procs_args(sample_filename, outfmt):
    procs_args = []
    out_filename = sample_filename

    if sample_filename.endswith(".gz"):
        procs_args.append(["gunzip", "-c", sample_filename])
        out_filename = replace_ext(out_filename, "")
    elif sample_filename.endswith(".zst"):
        procs_args.append(["zstdcat", sample_filename])
        out_filename = replace_ext(out_filename, "")

    _, _, sample_rate, _sample_dtype, _sample_len, in_format, in_bits = parse_filename(
        out_filename
    )
    print(_, sample_rate, _sample_dtype, _sample_len, in_format, in_bits)
    out_filename = replace_ext(out_filename, "raw", all_ext=True)
    sox_in = sample_filename
    if procs_args:
        sox_in = "-"
    procs_args.append(
        [
            "sox",
            "-D",  # disable dithering.
            "-t",
            "raw",
            "-r",
            str(sample_rate),
            "-c",
            "1",
            "-b",
            str(in_bits),
            "-e",
            in_format,
            sox_in,
            "-e",
            outfmt,
            out_filename,
        ]
    )
    return procs_args


def run_procs(procs_args):
    procs = []
    for proc_args in procs_args:
        stdin = None
        if procs:
            stdin = procs[-1].stdout
        procs.append(subprocess.Popen(proc_args, stdout=subprocess.PIPE, stdin=stdin))
    for proc in procs[:-1]:
        if proc.stdout is not None:
            proc.stdout.close()
    procs[-1].communicate()


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Convert (possibly compressed) sample recording to a float raw file (gnuradio style)"
    )
    parser.add_argument("samplefile", default="", help="sample file to read")
    parser.add_argument("--outfmt", default="float", help="output format")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    procs_args = make_procs_args(args.samplefile, args.outfmt)
    run_procs(procs_args)
