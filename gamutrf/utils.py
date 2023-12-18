#!/usr/bin/python3
import logging
import os
import re
import sys
from pathlib import Path

import requests

MTU = 8962
SCAN_FRES = 1e4
SCAN_FROLL = 1e6

# Use max recv_frame_size for USB - because we don't mind latency,
# we are optimizing for lower CPU.
# https://files.ettus.com/manual/page_transport.html
# https://github.com/EttusResearch/uhd/blob/master/host/lib/usrp/b200/b200_impl.hpp
# Should result in no overflows:
# UHD_IMAGES_DIR=/usr/share/uhd/images ./examples/rx_samples_to_file --args num_recv_frames=1000,recv_frame_size=16360 --file test.gz --nsamps 200000000 --rate 20000000 --freq 101e6 --spb 20000000
ETTUS_ARGS = "num_recv_frames=1000,recv_frame_size=16360,type=b200"
ETTUS_ANT = "TX/RX"
SAMPLE_FILENAME_RE = re.compile(r"^.+\D(\d+)_(\d+)Hz.*\D(\d+)sps\.c*([fisu]\d+|raw).*$")

SAMPLE_DTYPES = {
    "i8": ("<i1", "signed-integer"),
    "i16": ("<i2", "signed-integer"),
    "i32": ("<i4", "signed-integer"),
    "s8": ("<i1", "signed-integer"),
    "s16": ("<i2", "signed-integer"),
    "s32": ("<i4", "signed-integer"),
    "u8": ("<u1", "unsigned-integer"),
    "u16": ("<u2", "unsigned-integer"),
    "u32": ("<u4", "unsigned-integer"),
    "f32": ("<u4", "unsigned-integer"),
    "raw": ("<f4", "float"),
}
WIDTH = 11
HEIGHT = 8
DPI = 75
MPL_BACKEND = "cairo"
DS_PIXELS = 256
SAMP_RATE = 8.192e6
MIN_FREQ = 2.3e9
MAX_FREQ = 2.6e9


def endianstr():
    if sys.byteorder == "little":
        return "le"
    return "be"


def rotate_file_n(initial_name, n, require_initial=True):
    if require_initial and not os.path.exists(initial_name):
        return
    base_name = initial_name
    ext = ""
    dot = initial_name.rfind(".")
    if dot != -1:
        ext = base_name[dot:]
        base_name = base_name[:dot]
    for i in range(n, 1, -1):
        from_name = f"{base_name}.{i-1}{ext}"
        to_name = f"{base_name}.{i}{ext}"
        if os.path.exists(from_name):
            os.rename(from_name, to_name)
    if require_initial:
        os.rename(initial_name, f"{base_name}.1{ext}")


def replace_ext(filename, ext, all_ext=False):
    basename = os.path.basename(filename)
    if all_ext:
        dot = basename.index(".")
    else:
        dot = basename.rindex(".")
    new_basename = basename[: (dot + 1)] + ext
    return filename.replace(basename, new_basename)


def is_fft(filename):
    return os.path.basename(filename).startswith("fft_")


def get_nondot_files(filedir, glob="*.s*.*"):
    return [
        str(path)
        for path in Path(filedir).rglob(glob)
        if not os.path.basename(path).startswith(".")
    ]


def http_get(url, timeout=10):
    try:
        req = requests.get(url, timeout=timeout)
        logging.debug(str(req))
        return req
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as err:
        logging.error(f"{url}: {str(err)}")
        return None
