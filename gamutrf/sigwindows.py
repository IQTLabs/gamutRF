#!/usr/bin/python3
import random
from collections import Counter
from collections import defaultdict

import numpy as np
from gamutrf.utils import SCAN_FRES, SCAN_FROLL


ROLLOVERHZ = 100e6
CSV = ".csv"
ROLLING_FACTOR = int(SCAN_FROLL / SCAN_FRES)


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


def get_center(signal_mhz, freq_start_mhz, bin_mhz, record_bw):
    return int(
        int((signal_mhz - freq_start_mhz) / record_bw) * bin_mhz + freq_start_mhz
    )
