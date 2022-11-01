import argparse
import datetime
import logging
import os
import time
from statistics import mean

import numpy as np

from gamutrf.sigwindows import CSV
from gamutrf.sigwindows import read_csv

MAX_WORKERS = 4


def generate_csv(args):
    with open(args.outcsv, "w", encoding="utf-8") as outcsv:
        for frame, frame_df in read_csv(args):
            ts = frame_df["ts"].iat[0]
            dt = datetime.datetime.fromtimestamp(ts)
            frame_df.freq *= 1e6
            frame_df.freq = frame_df.freq.astype(np.int64)
            freq = frame_df.freq
            logging.info(f"frame {frame} at {time.ctime(ts)}")
            freqmin = freq.min()
            freqmax = freq.max()
            steps = min(len(frame_df), args.fftmax)
            stephz = int((freqmax - freqmin) / steps)
            stepi = int(len(frame_df) / steps)
            line = [
                dt.strftime("%Y-%m-%d %H:%M:%S"),
                "",
                freqmin,
                freqmax,
                stephz,
                1,
            ]
            dbt = []
            for i, row in enumerate(frame_df.itertuples()):
                dbt.append(row.db)
                if i % stepi == 0:
                    line.append(mean(dbt))
                    dbt = []
            if dbt:
                line.append(mean(dbt))
            outcsv.write(",".join([str(i) for i in line]) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a scan log to a CSV file compatible with rtl-gopow (https://github.com/dhogborg/rtl-gopow)"
    )
    parser.add_argument("csv", help="log file to parse")
    parser.add_argument("outcsv", help="rtl_power style CSV output")
    parser.add_argument(
        "--minhz", default=int(70 * 1e6), type=int, help="minimum frequency to process"
    )
    parser.add_argument(
        "--maxhz", default=int(6 * 1e9), type=int, help="maximum frequency to process"
    )
    parser.add_argument(
        "--mindb", default=int(-40), type=int, help="minimum dB to process"
    )
    parser.add_argument(
        "--maxframe", default=int(0), type=int, help="maximum frame number to process"
    )
    parser.add_argument(
        "--nrows", default=int(1e7), type=int, help="number of rows to read at once"
    )
    parser.add_argument(
        "--fftmax", default=int(10000), type=int, help="max number of FFT bins"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

    if not args.csv.endswith(CSV):
        logging.fatal(f"{args.csv} must be {CSV}")
    if not os.path.exists(args.csv):
        logging.fatal(f"{args.csv} must exist")
    generate_csv(args)
