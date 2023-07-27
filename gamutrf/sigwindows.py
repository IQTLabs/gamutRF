#!/usr/bin/python3
import random
from collections import Counter
from collections import defaultdict

import numpy as np
from gamutrf.peak_finder import get_peak_finder
from gamutrf.utils import SCAN_FRES, SCAN_FROLL


ROLLOVERHZ = 100e6
CSV = ".csv"
ROLLING_FACTOR = int(SCAN_FROLL / SCAN_FRES)


def choose_recorders(signals, recorder_freq_exclusions, max_recorder_signals):
    suitable_recorders = defaultdict(set)
    for signal in sorted(signals):
        for recorder, excluded in sorted(recorder_freq_exclusions.items()):
            if not freq_excluded(signal, excluded):
                suitable_recorders[signal].add(recorder)
    recorder_assignments = []
    busy_count = defaultdict(int)
    for signal, recorders in sorted(suitable_recorders.items(), key=lambda x: x[1]):
        if not recorders:
            continue
        busy_recorders = set(
            recorder
            for recorder, count in busy_count.items()
            if count >= max_recorder_signals
        )
        free_recorders = recorders - busy_recorders
        if not free_recorders:
            continue
        recorder = random.choice(list(free_recorders))  # nosec
        busy_count[recorder] += 1
        recorder_assignments.append((signal, recorder))
    return recorder_assignments


def parse_freq_excluded(freq_exclusions_raw):
    freq_exclusions = []
    for pair in freq_exclusions_raw:
        freq_min, freq_max = pair.split("-")
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


def calc_db(df, rolling_factor=ROLLING_FACTOR):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    meandb = df["db"].mean()
    if rolling_factor:
        df["db"] = df["db"].rolling(rolling_factor).mean().fillna(meandb)
    return df


def find_sig_windows(df, detection_type):
    if detection_type:
        peak_finder = get_peak_finder(detection_type)
        if peak_finder:
            db_data = df.db.to_numpy()
            peaks, _ = peak_finder.find_peaks(db_data)
            return [(df.iloc[peak].freq, df.iloc[peak].db) for peak in peaks]
    return []


def get_center(signal_mhz, freq_start_mhz, bin_mhz, record_bw):
    return int(
        int((signal_mhz - freq_start_mhz) / record_bw) * bin_mhz + freq_start_mhz
    )


def choose_record_signal(signals, max_signals):
    recorder_buckets = Counter()

    # Convert signals into buckets of record_bw size, count how many of each size
    for bucket in signals:
        recorder_buckets[bucket] += 1

    # Now count number of buckets of each count.
    buckets_by_count = defaultdict(set)
    for bucket, count in recorder_buckets.items():
        buckets_by_count[count].add(bucket)

    recorder_freqs = []
    # From least occuring bucket to most occurring, choose a random bucket for each recorder.
    for _count, buckets in sorted(buckets_by_count.items()):
        while buckets and len(recorder_freqs) < max_signals:
            bucket = random.choice(list(buckets))  # nosec
            buckets = buckets.remove(bucket)
            recorder_freqs.append(bucket)
        if len(recorder_freqs) == max_signals:
            break
    return recorder_freqs
