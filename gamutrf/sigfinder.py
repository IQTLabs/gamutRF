#!/usr/bin/python3
import argparse
import logging
import os
import subprocess
import time

import numpy as np
import pandas as pd
from sigwindows import find_sig_windows

CSV = '.csv'
ROLLOVERHZ = 100e6


def process_fft(args, ts, fftbuffer, lastbins):
    tsc = time.ctime(ts)
    logging.info(f'new frame at {tsc}')
    df = pd.DataFrame(fftbuffer, columns=['ts', 'freq', 'db'])
    df['freq'] /= 1e6
    df['db'] = 20 * np.log10(df['db'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['rollingdiffdb'] = df['db'].rolling(5).mean().diff()
    df = df.dropna()
    monitor_bins = set()
    for signal in find_sig_windows(df, window=args.window, threshold=args.threshold):
        start_freq, end_freq = signal[:2]
        center_freq = start_freq + ((end_freq - start_freq) / 2)
        center_freq = int(center_freq / args.bin_mhz) * args.bin_mhz
        monitor_bins.add(center_freq)
    peakrollingdiffdb = df['rollingdiffdb'].max()
    logging.info('current bins %f to %f MHz peak dB %f peak rolling diff dB %f: %s',
        df['freq'].min(), df['freq'].max(), df['db'].max(), peakrollingdiffdb, sorted(monitor_bins))
    new_bins = monitor_bins - lastbins
    if new_bins:
        logging.info('new bins: %s', sorted(new_bins))
    old_bins = lastbins - monitor_bins
    if old_bins:
        logging.info('old bins: %s', sorted(old_bins))
    return monitor_bins


def main():
    parser = argparse.ArgumentParser(
        description='watch an ettus scan log and find signals')
    parser.add_argument('csv', help='log file to parse')
    parser.add_argument('--bin_mhz', default=8, type=int, help='monitoring bin width in MHz')
    parser.add_argument('--window', default=4, type=int, help='signal finding sample window size')
    parser.add_argument('--threshold', default=1.5, type=float, help='signal finding threshold')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    if not args.csv.endswith(CSV):
        logging.fatal(f'{args.csv} must be {CSV}')
    if not os.path.exists(args.csv):
        logging.fatal(f'{args.csv} must exist')

    with subprocess.Popen(
            ['tail', '-F', args.csv],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE) as f:
        lastfreq = 0
        fftbuffer = []
        lastbins = set()
        while True:
            line = f.stdout.readline().strip()
            ts, freq, pw = [float(x) for x in line.split()]
            if abs(freq - lastfreq) > ROLLOVERHZ and fftbuffer:
                lastbins = process_fft(args, ts, fftbuffer, lastbins)
                fftbuffer = []
            fftbuffer.append((ts, freq, pw))
            lastfreq = freq


if __name__ == '__main__':
    main()
