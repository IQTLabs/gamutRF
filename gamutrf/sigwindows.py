#!/usr/bin/python3


import numpy as np


def calc_db(df):
    df['db'] = 20 * np.log10(df['db'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    df['rollingdiffdb'] = df['db'].rolling(5).mean().diff()
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
            signal_df = df[(df['freq'] >= start_freq) & (df['freq'] <= end_freq)]
            signals.append((start_freq, end_freq, in_signal.db, row.db, signal_df['db'].max()))
            in_signal = None
        else:
            in_signal = row
    return signals
