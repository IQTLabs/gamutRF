#!/usr/bin/python3

import re
import os
from pathlib import Path


# Use max recv_frame_size for USB - because we don't mind latency,
# we are optimizing for lower CPU.
# https://files.ettus.com/manual/page_transport.html
# https://github.com/EttusResearch/uhd/blob/master/host/lib/usrp/b200/b200_impl.hpp
# Should result in no overflows:
# UHD_IMAGES_DIR=/usr/share/uhd/images ./examples/rx_samples_to_file --args num_recv_frames=128,recv_frame_size=16360 --file test.gz --nsamps 200000000 --rate 20000000 --freq 101e6 --spb 20000000
ETTUS_ARGS = 'num_recv_frames=128,recv_frame_size=16360'
ETTUS_ANT = 'TX/RX'


def replace_ext(filename, ext):
    basename = os.path.basename(filename)
    dot = basename.index('.')
    new_basename = basename[:(dot + 1)] + ext
    return filename.replace(basename, new_basename)


def parse_filename(filename):
    # TODO: parse from sigmf.
    filename_re = re.compile('^.+_([0-9]+)Hz_([0-9]+)sps.+$')
    match = filename_re.match(filename)
    freq_center = int(match.group(1))
    sample_rate = int(match.group(2))
    return (freq_center, sample_rate)


def get_nondot_files(filedir, glob='*.s*.*'):
    return [str(path) for path in Path(filedir).rglob(glob)
            if not os.path.basename(path).startswith('.')]
