#!/usr/bin/python3

import re
import os


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
