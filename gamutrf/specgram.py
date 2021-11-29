#!/usr/bin/python3

import argparse
import gzip
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def read_recording(filename):
    # TODO: assume int16.int16 (parse from sigmf).
    dtype = np.dtype([('i','<i2'), ('q','<i2')])
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rb') as infile:
            x1d = np.frombuffer(infile.read(), dtype=dtype)
    else:
        x1d = np.fromfile(filename, dtype=dtype)
    x = x1d['i'] + 1j * x1d['q']
    return x


def plot_spectrogram(x, spectrogram_filename, nfft, fs, fc, cmap):
    plt.xlabel('time (s)')
    plt.ylabel('freq (Hz)')
    # overlap must be 0, for maximum detail.
    plt.specgram(x, NFFT=nfft, Fs=fs, cmap=cmap, Fc=fc, noverlap=0)
    plt.gcf().set_size_inches(11, 8)
    plt.savefig(spectrogram_filename)


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


def main():
    parser = argparse.ArgumentParser(
        description='draw spectogram from recording')
    parser.add_argument('recording', default='', type=str,
                        help='filename of recording')
    parser.add_argument('--nfft', default=int(65536), type=int,
                        help='number of FFT points')
    parser.add_argument('--cmap', default='twilight_r', type=str,
                        help='pyplot colormap (see https://matplotlib.org/stable/tutorials/colors/colormaps.html)')
    args = parser.parse_args()
    freq_center, sample_rate = parse_filename(args.recording)
    samples = read_recording(args.recording)
    plot_spectrogram(
        samples,
        replace_ext(args.recording, 'jpg'),
        args.nfft,
        sample_rate,
        freq_center,
        cmap=args.cmap)


if __name__ == '__main__':
    main()
