#!/usr/bin/python3


import pandas as pd


def find_sig_windows(df, window=4, threshold=2, min_bw_mhz=1):
    window_df = df[(df['rollingdiffdb'].rolling(window).sum().abs() > (window * threshold))]
    freq_diff = window_df['freq'].diff().fillna(min_bw_mhz)
    signals = []
    in_signal = None
    for row in window_df[freq_diff >= min_bw_mhz].itertuples():
        if in_signal is not None:
            signals.append((in_signal.freq, row.freq, in_signal.db, row.db))
            in_signal = None
        else:
            in_signal = row
    return signals
