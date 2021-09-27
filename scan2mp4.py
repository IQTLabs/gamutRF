#!/usr/bin/python3

import argparse
import logging
import os
import subprocess
import tempfile
import time
import numpy as np
import pandas as pd

ROLLOVERHZ = 100e6


def read_csv(args):
    frames = 0
    skiprows = 0
    minmhz = args.minhz / 1e6
    rolloverhz = ROLLOVERHZ / 1e6
    while True:
        df = pd.read_csv(
            args.csv, header=None, delim_whitespace=True, skiprows=skiprows, nrows=args.nrows)
        read_rows = len(df)
        if read_rows == 0:
            break
        df.columns = ['ts', 'freq', 'db']
        df['freq'] /= 1e6
        freqdiff = df['freq'].diff().abs()
        df['frame'] = 0
        # Detect tuning wraparound, where frequency changed by more than 100MHz
        df.loc[freqdiff > rolloverhz, ['frame']] = 1
        df['frame'] = df['frame'].cumsum().fillna(0).astype(np.uint64)  # pylint: disable=unsupported-assignment-operation,disable=unsubscriptable-object
        read_frames = df['frame'].max()  # pylint: disable=unsubscriptable-object
        if read_rows < args.nrows:
            frames_rows = read_rows
        else:
            frames_rows = len(df[df['frame'] < read_frames])  # pylint: disable=unsubscriptable-object
            skiprows += frames_rows
            df = df[:frames_rows]  # pylint: disable=unsubscriptable-object
        logging.info(f'read {skiprows} total rows from {args.csv}')
        df = df[df['freq'] >= minmhz]
        df['db'] = 20 * np.log10(df['db'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        df['rollingdiffdb'] = df['db'].rolling(5).mean().diff()
        df.set_index('frame', inplace=True)
        df['ts'] = df.groupby('frame', sort=False)['ts'].transform(min)
        for _, frame_df in df.groupby('frame'):
            yield (frames, frame_df)
            frames += 1
        if read_rows < args.nrows:
            break


def generate_frames(args, tempdir):
    tsmap = []
    db_min = None
    db_max = None
    diff_min = None
    diff_max = None
    freq_min = None
    freq_max = None
    lastts = None

    for frame, frame_df in read_csv(args):
        ts = frame_df['ts'].iat[0]
        tsc = time.ctime(ts)
        frame_f = os.path.join(str(tempdir), str(ts))
        tsmap.append((frame, ts, tsc, frame_f))
        frame_df.to_csv(frame_f, sep='\t', columns=['freq', 'db', 'rollingdiffdb'], index=False)
        db = frame_df.db
        rollingdiffdb = frame_df.rollingdiffdb
        freq = frame_df.freq
        if frame:
            db_min = min(db_min, db.min())
            db_max = max(db_max, db.max())
            diff_min = min(diff_min, rollingdiffdb.min())
            diff_max = max(diff_max, rollingdiffdb.max())
            freq_min = min(freq_min, freq.min())
            freq_max = max(freq_max, freq.max())
        else:
            db_min = db.min()
            db_max = db.max()
            diff_min = rollingdiffdb.min()
            diff_max = rollingdiffdb.max()
            freq_min = freq.min()
            freq_max = freq.max()
            lastts = ts
        offset = ts - lastts
        logging.info(f'frame {frame} offset {offset}s at {tsc}')
        lastts = ts

    return (tsmap, freq_min, freq_max, db_min, db_max, diff_min, diff_max)


def run_gnuplot(tsmap, freq_min, freq_max, db_min, db_max, diff_min, diff_max, args, tempdir):
    logging.info('creating gunplot commands')
    y_min = min(diff_min, db_min)
    y_max = max(diff_max, db_max)
    xtics = (freq_max - freq_min) / args.xtics

    gnuplot_cmds = [
        'set terminal png truecolor rounded size 1920,720 enhanced',
        'set ytics format "%.4f"',
        'set xtics rotate by 90 right',
        'set grid xtics',
        'set grid mxtics',
        'set grid ytics',
        'set xlabel "freq (MHz)"',
        'set ylabel "power (dB)"']

    gnuplot_cmds.extend([
        f'set xtics {xtics}',
        f'set xrange [{freq_min}:{freq_max}]',
        f'set yrange [{y_min}:{y_max}]'])
    for frame, _ts, tsc, frame_f in tsmap:
        gnuplot_cmds.extend([
            f'set output "{tempdir}/{frame:06}.png"',
            f'set title "{tsc}"',
            f'plot "{frame_f}" using 1:2 with linespoints title "dB", "{frame_f}" using 1:3 with linespoints title "rolling diff dB"'])

    plot_f = os.path.join(str(tempdir), 'plot.cmd')
    with open(plot_f, 'w', encoding='utf8') as plot_c:
        plot_c.write('\n'.join(gnuplot_cmds))

    logging.info('running gnuplot')
    subprocess.check_call(['gnuplot', plot_f])


def run_ffmpeg(args, tempdir, mp4):
    logging.info('running ffmpeg')
    subprocess.check_call([
        'ffmpeg',
        '-loglevel', 'error',
        '-framerate', str(args.framerate),
        '-pattern_type', 'glob', '-i', os.path.join(str(tempdir), '*.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-r', '30', '-y', mp4])


def main():
    parser = argparse.ArgumentParser(
        description='Convert an ettus_scan log to a timelapse graph')
    parser.add_argument('csv', help='log file to parse')
    parser.add_argument('--minhz', default=int(70 * 1e6), help='minimum frequency to process')
    parser.add_argument('--framerate', default=int(5), help='frame rate')
    parser.add_argument('--xtics', default=int(40), help='xtics')
    parser.add_argument('--nrows', default=int(1e7), help='number of rows to read at once')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    ext = '.csv'
    if not args.csv.endswith(ext):
        logging.fatal(f'{args.csv} must be {ext}')
    if not os.path.exists(args.csv):
        logging.fatal(f'{args.csv} must exist')
    mp4 = args.csv.replace(ext, '.mp4')

    with tempfile.TemporaryDirectory() as tempdir:
        tsmap, freq_min, freq_max, db_min, db_max, diff_min, diff_max = generate_frames(
            args, tempdir)
        run_gnuplot(
            tsmap, freq_min, freq_max, db_min, db_max, diff_min, diff_max, args, tempdir)
        run_ffmpeg(args, tempdir, mp4)


if __name__ == '__main__':
    main()
