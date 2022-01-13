#!/usr/bin/python3

import argparse
import gzip
import os
import re
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.mlab import detrend, detrend_none, window_hanning, stride_windows
import numpy as np
from scipy.fft import fft, fftfreq


def _gamutrf_spectral_helper(x, y=None, NFFT=None, Fs=None, detrend_func=None,
                             window=None, noverlap=None, pad_to=None,
                             sides=None, scale_by_freq=None, mode=None):
    """
    Private helper implementing the common parts between the psd, csd,
    spectrogram and complex, magnitude, angle, and phase spectrums.
    """
    if y is None:
        # if y is None use x for y
        same_data = True
    else:
        # The checks for if y is x are so that we can use the same function to
        # implement the core of psd(), csd(), and spectrogram() without doing
        # extra calculations.  We return the unaveraged Pxy, freqs, and t.
        same_data = y is x

    if Fs is None:
        Fs = 2
    if noverlap is None:
        noverlap = 0
    if detrend_func is None:
        detrend_func = detrend_none
    if window is None:
        window = window_hanning

    # if NFFT is set to None use the whole signal
    if NFFT is None:
        NFFT = 256

    if mode is None or mode == 'default':
        mode = 'psd'
    #_api.check_in_list(
    #    ['default', 'psd', 'complex', 'magnitude', 'angle', 'phase'],
    #    mode=mode)

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is not 'psd'")

    # Make sure we're dealing with a numpy array. If y and x were the same
    # object to start with, keep them that way
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)

    if sides is None or sides == 'default':
        if np.iscomplexobj(x):
            sides = 'twosided'
        else:
            sides = 'onesided'
    #_api.check_in_list(['default', 'onesided', 'twosided'], sides=sides)

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, NFFT)
        x[n:] = 0

    if not same_data and len(y) < NFFT:
        n = len(y)
        y = np.resize(y, NFFT)
        y[n:] = 0

    if pad_to is None:
        pad_to = NFFT

    if mode != 'psd':
        scale_by_freq = False
    elif scale_by_freq is None:
        scale_by_freq = True

    # For real x, ignore the negative frequencies unless told otherwise
    if sides == 'twosided':
        numFreqs = pad_to
        if pad_to % 2:
            freqcenter = (pad_to - 1)//2 + 1
        else:
            freqcenter = pad_to//2
        scaling_factor = 1.
    elif sides == 'onesided':
        if pad_to % 2:
            numFreqs = (pad_to + 1)//2
        else:
            numFreqs = pad_to//2 + 1
        scaling_factor = 2.

    if not np.iterable(window):
        window = window(np.ones(NFFT, x.dtype))
    if len(window) != NFFT:
        raise ValueError(
            "The window length must match the data's first dimension")

    result = stride_windows(x, NFFT, noverlap, axis=0)
    result = detrend(result, detrend_func, axis=0)
    # result = result * window.reshape((-1, 1))
    result = fft(result, n=pad_to, axis=0)[:numFreqs, :]
    freqs = fftfreq(pad_to, 1/Fs)[:numFreqs]

    if not same_data:
        # if same_data is False, mode must be 'psd'
        resultY = stride_windows(y, NFFT, noverlap)
        resultY = detrend(resultY, detrend_func, axis=0)
        resultY = resultY * window.reshape((-1, 1))
        resultY = np.fft.fft(resultY, n=pad_to, axis=0)[:numFreqs, :]
        result = np.conj(result) * resultY
    elif mode == 'psd':
        result = np.conj(result) * result
    elif mode == 'magnitude':
        result = np.abs(result) / np.abs(window).sum()
    elif mode == 'angle' or mode == 'phase':
        # we unwrap the phase later to handle the onesided vs. twosided case
        result = np.angle(result)
    elif mode == 'complex':
        result /= np.abs(window).sum()

    if mode == 'psd':

        # Also include scaling factors for one-sided densities and dividing by
        # the sampling frequency, if desired. Scale everything, except the DC
        # component and the NFFT/2 component:

        # if we have a even number of frequencies, don't scale NFFT/2
        if not NFFT % 2:
            slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
        else:
            slc = slice(1, None, None)

        result[slc] *= scaling_factor

        # MATLAB divides by the sampling frequency so that density function
        # has units of dB/Hz and can be integrated by the plotted frequency
        # values. Perform the same scaling here.
        if scale_by_freq:
            result /= Fs
            # Scale the spectrum by the norm of the window to compensate for
            # windowing loss; see Bendat & Piersol Sec 11.5.2.
            result /= (np.abs(window)**2).sum()
        else:
            # In this case, preserve power in the segment, not amplitude
            result /= np.abs(window).sum()**2

    t = np.arange(NFFT/2, len(x) - NFFT/2 + 1, NFFT - noverlap)/Fs

    if sides == 'twosided':
        # center the frequency range at zero
        freqs = np.roll(freqs, -freqcenter, axis=0)
        result = np.roll(result, -freqcenter, axis=0)
    elif not pad_to % 2:
        # get the last value correctly, it is negative otherwise
        freqs[-1] *= -1

    # we unwrap the phase here to handle the onesided vs. twosided case
    if mode == 'phase':
        result = np.unwrap(result, axis=0)

    return result, freqs, t


def read_recording(filename):
    # TODO: assume int16.int16 (parse from sigmf).
    dtype = np.dtype([('i','<i2'), ('q','<i2')])
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rb') as infile:
            x1d = np.frombuffer(infile.read(), dtype=dtype)
    else:
        x1d = np.fromfile(filename, dtype=dtype)
    x = x1d['i'] + np.csingle(1j) * x1d['q']
    return x


def plot_spectrogram(x, spectrogram_filename, nfft, fs, fc, cmap):
    mlab._spectral_helper = _gamutrf_spectral_helper
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
