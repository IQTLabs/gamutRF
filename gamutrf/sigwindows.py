#!/usr/bin/python3
import logging
import random
import os
import time
from collections import Counter
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
import matplotlib.pyplot as plt
from gamutrf.utils import SCAN_FRES, SCAN_FROLL, WIDTH, HEIGHT, DPI, MPL_BACKEND


ROLLOVERHZ = 100e6
CSV = ".csv"
CSVCOLS = ["ts", "freq", "db"]
ROLLING_FACTOR = int(SCAN_FROLL / SCAN_FRES)

logging.getLogger("matplotlib.font_manager").disabled = True


def read_csv_chunks(args):
    minmhz = args.minhz / 1e6
    leftover_df = pd.DataFrame()

    def detect_frames(df):
        freqdiff = df["freq"].diff().abs()
        df["frame"] = 0
        # Detect tuning wraparound, where frequency changed by more than 100MHz
        df.loc[freqdiff > (ROLLOVERHZ / 1e6), ["frame"]] = 1
        df[
            "frame"
        ] = (  # pylint: disable=unsupported-assignment-operation,disable=unsubscriptable-object
            df["frame"].cumsum().fillna(0).astype(np.uint64)
        )  # pylint: disable=unsubscriptable-object

    def preprocess_frames(df):
        df.set_index("frame", inplace=True)
        df["ts"] = df.groupby("frame", sort=False)["ts"].transform(min)

    with pd.read_csv(
        args.csv, header=None, delim_whitespace=True, chunksize=args.nrows
    ) as reader:
        for chunk in reader:
            read_rows = len(chunk)
            logging.info(f"read chunk of {read_rows} from {args.csv}")
            chunk.columns = CSVCOLS
            chunk["freq"] /= 1e6
            df = pd.concat([leftover_df, chunk])
            detect_frames(df)
            read_frames = df["frame"].max()  # pylint: disable=unsubscriptable-object
            if read_frames == 0 and len(chunk) == args.nrows:
                leftover_df = leftover_df.append(chunk[CSVCOLS])
                logging.info(f"buffering incomplete frame - {args.nrows} too small?")
                continue
            leftover_df = df[df["frame"] == read_frames][
                CSVCOLS
            ]  # pylint: disable=unsubscriptable-object
            df = df[
                (df["frame"] < read_frames) & (df["freq"] >= minmhz)
            ]  # pylint: disable=unsubscriptable-object
            df = calc_db(df)
            df = df[df["db"] >= args.mindb]
            preprocess_frames(df)
            yield df

    if len(leftover_df):
        df = leftover_df
        detect_frames(df)
        df = df[
            (df["frame"] < read_frames) & (df["freq"] >= minmhz)
        ]  # pylint: disable=unsubscriptable-object
        df = calc_db(df)
        df = df[df["db"] >= args.mindb]
        preprocess_frames(df)
        yield df


def read_csv(args):
    frames = 0

    for df in read_csv_chunks(args):
        for _, frame_df in df.groupby("frame"):
            yield (frames, frame_df)
            if args.maxframe and frames == args.maxframe:
                return
            frames += 1


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


def calc_db(df):
    df["db"] = 20 * np.log10(df[df["db"] != 0]["db"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    meandb = df["db"].mean()
    df["db"] = df["db"].rolling(ROLLING_FACTOR).mean().fillna(meandb)
    return df


def scipy_find_sig_windows(df, width, prominence, threshold):
    data = df.db.to_numpy()
    peaks, _ = find_peaks(data, prominence=prominence, width=(width,), height=threshold)
    return [(df.iloc[peak].freq, df.iloc[peak].db) for peak in peaks]


def graph_fft_peaks(graph_path, df, mean_running_df, sample_count_df, signals):
    maxdb = df.db.max()
    df["peaks"] = df.db.min()
    for peak_freq, _ in signals:
        df.loc[df.freq == peak_freq, "peaks"] = maxdb

    peak_df = df[df.peaks == maxdb].sort_values("db", ascending=False)[:5]
    peak_signals = ",".join(
        ["%.1f MHz %.1f dB" % (row.freq, row.db) for row in peak_df.itertuples()]
    )
    if peak_signals:
        peak_signals = f"strongest peak signals: {peak_signals}"

    matplotlib.use(MPL_BACKEND)
    plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    plt.plot(
        df.freq,
        df.db,
        "b",
        df.freq,
        df.peaks,
        "y",
        mean_running_df.freq,
        mean_running_df.db,
        "k",
        # sample_count_df["freq"],
        # sample_count_df["size"],
        # "c",
    )
    plt.xlabel("freq (MHz)")
    plt.ylabel("power (dB)")
    plt.legend(("power", "peak status", "mean power"), loc="upper right")
    ts_min = df.ts.min()
    ts_max = df.ts.max()
    time_min = time.ctime(ts_min)
    time_max = time.ctime(ts_max)
    duration = ts_max - ts_min
    plt.title(
        f"gamutRF scanner FFT {time_min} to {time_max}, {duration}s\n{peak_signals}"
    )
    real_path = os.path.realpath(graph_path)
    basename = os.path.basename(real_path)
    dirname = os.path.dirname(real_path)
    tmp_graph_path = os.path.join(dirname, "." + basename)
    plt.savefig(tmp_graph_path)
    plt.cla()
    plt.close("all")
    os.rename(tmp_graph_path, graph_path)


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
