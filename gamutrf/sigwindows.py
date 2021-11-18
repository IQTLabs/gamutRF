#!/usr/bin/python3

import random
from collections import defaultdict, Counter
import numpy as np


def choose_recorders(signals, recorder_freq_exclusions):
    suitable_recorders = defaultdict(set)
    for signal in sorted(signals):
        for recorder, excluded in sorted(recorder_freq_exclusions.items()):
            if not freq_excluded(signal, excluded):
                suitable_recorders[signal].add(recorder)
    recorder_assignments = []
    busy_recorders = set()
    for signal, recorders in sorted(suitable_recorders.items(), key=lambda x: x[1]):
        if not recorders:
            continue
        free_recorders = recorders - busy_recorders
        if not free_recorders:
            continue
        recorder = random.choice(list(free_recorders))  # nosec
        busy_recorders.add(recorder)
        recorder_assignments.append((signal, recorder))
    return recorder_assignments


def parse_freq_excluded(freq_exclusions_raw):
    freq_exclusions = []
    for pair in freq_exclusions_raw:
        freq_min, freq_max = pair.split('-')
        if len(freq_min):
            freq_min = int(freq_min)
        else:
            freq_min = None
        if len(freq_max):
            freq_max = int(freq_max)
        else:
            freq_max = None
        freq_exclusions.append((freq_min, freq_max))
    return tuple(freq_exclusions)


def freq_excluded(freq, freq_exclusions):
    for freq_min, freq_max in freq_exclusions:
        if freq_min is not None and freq_max is not None:
            if freq >= freq_min and freq <= freq_max:
                return True
            continue
        if freq_min is None:
            if freq <= freq_max:
                return True
            continue
        if freq >= freq_min:
            return True
    return False


def calc_db(df):
    df['db'] = 20 * np.log10(df[df['db'] != 0]['db'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['rollingdiffdb'] = df[df['db'].notna()]['db'].rolling(5).mean().diff()
    return df


def find_sig_windows(df, window=4, threshold=2, min_bw_mhz=1):
    window_df = df[(df['rollingdiffdb'].rolling(
        window).sum().abs() > (window * threshold))]
    freq_diff = window_df['freq'].diff().fillna(min_bw_mhz)
    signals = []
    in_signal = None
    for row in window_df[freq_diff >= min_bw_mhz].itertuples():
        if in_signal is not None:
            start_freq = in_signal.freq
            end_freq = row.freq
            signal_df = df[(df['freq'] >= start_freq)
                           & (df['freq'] <= end_freq)]
            signals.append((start_freq, end_freq, in_signal.db,
                           row.db, signal_df['db'].max()))
            in_signal = None
        else:
            in_signal = row
    return signals


def choose_record_signal(signals, recorders, record_bw):
    recorder_buckets = Counter()

    # Convert signals into buckets of record_bw size, count how many of each size
    for center_freq in signals:
        bucket = round(center_freq / record_bw) * record_bw
        recorder_buckets[bucket] += 1

    # Now count number of buckets of each count.
    buckets_by_count = defaultdict(set)
    for bucket, count in recorder_buckets.items():
        buckets_by_count[count].add(bucket)

    recorder_freqs = []
    # From least occuring bucket to most occurring, choose a random bucket for each recorder.
    for _count, buckets in sorted(buckets_by_count.items()):
        while buckets and len(recorder_freqs) < recorders:
            bucket = random.choice(list(buckets))  # nosec
            buckets = buckets.remove(bucket)
            recorder_freqs.append(bucket)
        if len(recorder_freqs) == recorders:
            break
    return recorder_freqs
