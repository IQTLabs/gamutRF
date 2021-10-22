#!/usr/bin/python3
import argparse
import logging
import os
import requests
import subprocess
import time

import pandas as pd

from gamutrf.sigwindows import calc_db
from gamutrf.sigwindows import find_sig_windows, choose_record_signal

ROLLOVERHZ = 100e6


def process_fft(args, ts, fftbuffer, lastbins):
    tsc = time.ctime(ts)
    logging.info(f'new frame at {tsc}')
    df = pd.DataFrame(fftbuffer, columns=['ts', 'freq', 'db'])
    df['freq'] /= 1e6
    df = calc_db(df)
    monitor_bins = set()
    peak_dbs = {}
    for signal in find_sig_windows(df, window=args.window, threshold=args.threshold):
        start_freq, end_freq = signal[:2]
        peak_db = signal[-1]
        center_freq = start_freq + ((end_freq - start_freq) / 2)
        center_freq = int(center_freq / args.bin_mhz) * args.bin_mhz
        monitor_bins.add(center_freq)
        peak_dbs[center_freq] = peak_db
    logging.info('current bins %f to %f MHz: %s',
                 df['freq'].min(), df['freq'].max(), sorted(peak_dbs.items()))
    new_bins = monitor_bins - lastbins
    if new_bins:
        logging.info('new bins: %s', sorted(new_bins))
    old_bins = lastbins - monitor_bins
    if old_bins:
        logging.info('old bins: %s', sorted(old_bins))
    return monitor_bins


def main():
    parser = argparse.ArgumentParser(
        description='watch a scan UDP stream and find signals')
    parser.add_argument('--log', default='scan.log', type=str,
                        help='base path for scan logging')
    parser.add_argument('--rotatesecs', default=3600, type=int,
                        help='rotate scan log after this many seconds')
    parser.add_argument('--logaddr', default='127.0.0.1', type=str,
                        help='UDP stream address')
    parser.add_argument('--logport', default=8001, type=int,
                        help='UDP stream port')
    parser.add_argument('--bin_mhz', default=20, type=int,
                        help='monitoring bin width in MHz')
    parser.add_argument('--window', default=4, type=int,
                        help='signal finding sample window size')
    parser.add_argument('--threshold', default=1.5, type=float,
                        help='signal finding threshold')
    parser.add_argument('--history', default=50, type=int,
                        help='number of frames of signal history to keep')
    parser.add_argument('--recorders', default=[], type=str, nargs='*',
                        help='list of SDR recorder base URLs (e.g. http://host:port/)')
    parser.add_argument('--record_bw_mbps', default=20, type=int,
                        help='record bandwidth in mbps')
    parser.add_argument('--record_secs', default=10, type=int,
                        help='record time duration in seconds')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    try:
        ext = args.log[args.log.rindex('.'):]
    except ValueError:
        logging.fatal(f'cannot parse extension from {args.log}')

    with subprocess.Popen(
            ['nc', '-u', '-l', args.logaddr, str(args.logport)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE) as f:
        lastfreq = 0
        fftbuffer = []
        lastbins_history = []
        lastbins = set()
        mode = 'wb'
        if os.path.exists(args.log):
            logging.info(f'{args.log} exists, will append first')
            mode = 'ab'

        while True:
            logging.info(f'reopening {args.log}')
            openlogts = int(time.time())
            with open(args.log, mode=mode) as l:
                while True:
                    line = f.stdout.readline()  # pytype: disable=attribute-error
                    now = int(time.time())
                    try:
                        ts, freq, pw = [float(x) for x in line.strip().split()]
                    except ValueError:
                        continue
                    if pw < 0 or pw > 1:
                        continue
                    if freq < 0 or freq > 10e9:
                        continue
                    if abs(now - ts) > 60:
                        continue
                    l.write(line)
                    rollover = abs(freq - lastfreq) > ROLLOVERHZ and fftbuffer
                    fftbuffer.append((ts, freq, pw))
                    lastfreq = freq
                    if rollover:
                        lastbins = process_fft(args, ts, fftbuffer, lastbins)
                        if lastbins:
                            lastbins_history = [lastbins] + lastbins_history
                            lastbins_history = lastbins_history[:args.history]
                        if lastbins_history:
                            signals = []
                            for bins in lastbins_history:
                                signals.extend(list(bins))
                            recorder_count = len(args.recorders)
                            record_signals = choose_record_signal(signals, recorder_count, args.record_bw_mbps)
                            for signal, recorder in zip(record_signals, args.recorders):
                                signal_hz = int(signal * 1e6)
                                record_bps = int(args.record_bw_mbps * 1e6)
                                record_samples = int(record_bps * args.record_secs)
                                url = f'{recorder}/v1/record/{signal_hz}/{record_samples}/{record_bps}'
                                logging.debug('requesting %s (center %f MHz)', url, signal)
                                try:
                                    req = requests.get(url, timeout=args.record_secs)
                                    logging.debug(str(req))
                                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as err:
                                    logging.debug(str(err))
                        fftbuffer = []
                        if now - openlogts > args.rotatesecs:
                            break
            new_log = args.log.replace(ext, f'{openlogts}{ext}')
            os.rename(args.log, new_log)
            mode = 'wb'


if __name__ == '__main__':
    main()
