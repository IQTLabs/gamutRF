#!/usr/bin/python3
import argparse
import concurrent.futures
import logging
import os
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from gamutrf.sigwindows import calc_db

MAX_WORKERS = 4
ROLLOVERHZ = 100e6
CSV = '.csv'


def read_csv(args):
    frames = 0
    skiprows = 0
    minmhz = args.minhz / 1e6
    rolloverhz = ROLLOVERHZ / 1e6
    complete = False
    while not complete:
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
        df['frame'] = (   # pylint: disable=unsupported-assignment-operation,disable=unsubscriptable-object
                df['frame'].cumsum().fillna(0).astype(np.uint64))  # pylint: disable=unsubscriptable-object
        read_frames = df['frame'].max()  # pylint: disable=unsubscriptable-object
        if read_rows < args.nrows:
            frames_rows = read_rows
            skiprows += read_rows
        else:
            frames_rows = len(
                df[df['frame'] < read_frames])  # pylint: disable=unsubscriptable-object
            skiprows += frames_rows
            df = df[:frames_rows]  # pylint: disable=unsubscriptable-object
        logging.info(f'read {skiprows} total rows from {args.csv}')
        df = df[df['freq'] >= minmhz]
        df = calc_db(df)
        df.set_index('frame', inplace=True)
        df['ts'] = df.groupby('frame', sort=False)['ts'].transform(min)
        for _, frame_df in df.groupby('frame'):
            yield (frames, frame_df)
            if args.maxframe and frames == args.maxframe:
                complete = True
                break
            frames += 1
        if read_rows < args.nrows:
            complete = True


def generate_frames(args, tempdir):
    tsmap = []
    db_min = None
    db_max = None
    diff_min = None
    diff_max = None
    freq_min = None
    freq_max = None
    lastts = None

    def write_frame(frame, frame_f, frame_df):
        frame_fs = [frame_f]
        if frame in args.dumpframes:
            dumpfile = args.csv.replace(CSV, f'-{frame}.csv')
            frame_fs.append(dumpfile)
        for f in frame_fs:
            frame_df.to_csv(f, sep='\t', columns=[
                            'freq', 'db', 'rollingdiffdb'], index=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        maxmhz = args.maxhz / 1e6
        for frame, frame_df in read_csv(args):
            ts = frame_df['ts'].iat[0]
            frame_df = frame_df[frame_df['freq'] <= maxmhz]
            tsc = time.ctime(ts)
            frame_f = os.path.join(str(tempdir), str(ts))
            tsmap.append((frame, ts, tsc, frame_f))
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
            executor.submit(write_frame, frame, frame_f, frame_df.copy())
            frame_df.to_csv(
                os.path.join(str(tempdir), 'graph.csv') , sep='\t', mode='a',
                columns=['ts', 'freq', 'db'], index=False, header=False)
        executor.shutdown(wait=True)

    return (tsmap, freq_min, freq_max, db_min, db_max, diff_min, diff_max)


def call_gnuplot(gnuplot_cmds, tempdir):
    plot_f = os.path.join(str(tempdir), 'plot.cmd')
    with open(plot_f, 'w', encoding='utf8') as plot_c:
        plot_c.write('\n'.join(gnuplot_cmds))

    logging.info('running gnuplot')
    subprocess.check_call(['gnuplot', plot_f])


def run_gnuplot(tsmap, freq_min, freq_max, db_min, db_max, diff_min, diff_max, args, tempdir, graph, mindb):
    logging.info('creating gunplot commands')
    y_min = min(diff_min, db_min)
    y_max = max(diff_max, db_max)
    freq_min = int(freq_min / 100) * 100
    freq_max = round(freq_max / 100) * 100
    xtics = (freq_max - freq_min) / args.xtics

    common_gnuplot_cmds = [
        'set terminal png truecolor rounded size 1920,720 enhanced',
        'set grid xtics',
        'set grid mxtics',
        'set grid ytics',
    ]

    gnuplot_cmds = common_gnuplot_cmds + [
        'set ytics format "%.4f"',
        'set xtics rotate by 90 right',
        'set xlabel "freq (MHz)"',
        'set ylabel "power (dB)"']

    gnuplot_cmds.extend([
        f'set xtics {xtics}',
        f'set xrange [{freq_min}:{freq_max}]',
        f'set yrange [{y_min}:{y_max}]'])
    for frame, _ts, tsc, frame_f in tsmap:
        db_plot = (
            f'"{frame_f}" using 1:2 with linespoints title "dB"')
        rolling_diff_plot = (
            f'"{frame_f}" using 1:3 with linespoints title "rolling diff dB"')
        gnuplot_cmds.extend([
            f'set output "{tempdir}/{frame:06}.png"',
            f'set title "{tsc} frame {frame}"',
            f'plot {db_plot}, {rolling_diff_plot}'])

    call_gnuplot(gnuplot_cmds, tempdir)

    gnuplot_cmds = common_gnuplot_cmds + [
        'set xdata time',
        'set timefmt "%s"',
        'set view 0,270',
        'set xlabel "time"',
        'set ylabel "freq (MHz)"',
        'set palette negative',
        'set pm3d map',
        f'set output "{graph}"',
        f'splot "{tempdir}/graph.csv" using 1:2:($3>{mindb}?$3:(1/0)) palette with circles title "dB"',
    ]

    call_gnuplot(gnuplot_cmds, tempdir)


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
        description='Convert an ettus_scan log to a timelapse graph/video')
    parser.add_argument('csv', help='log file to parse')
    parser.add_argument('--minhz', default=int(70 * 1e6), type=int,
                        help='minimum frequency to process')
    parser.add_argument('--maxhz', default=int(6 * 1e9), type=int,
                        help='maximum frequency to process')
    parser.add_argument('--framerate', default=int(5), help='frame rate')
    parser.add_argument('--xtics', default=int(40), help='xtics')
    parser.add_argument('--nrows', default=int(1e7),
                        help='number of rows to read at once')
    parser.add_argument('--tmproot', default='',
                        help='root of temporary directory')
    parser.add_argument('--maxframe', default=int(0), type=int,
                        help='maximum frame number to process')
    parser.add_argument('--dumpframes', default=[], nargs='+', type=int,
                        help='list of frame numbers to dump as csvs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    if not args.csv.endswith(CSV):
        logging.fatal(f'{args.csv} must be {CSV}')
    if not os.path.exists(args.csv):
        logging.fatal(f'{args.csv} must exist')
    mp4 = args.csv.replace(CSV, '.mp4')
    graph = args.csv.replace(CSV, '.png')

    tmproot = None
    if args.tmproot:
        tmproot = args.tmproot

    with tempfile.TemporaryDirectory(dir=tmproot) as tempdir:
        tsmap, freq_min, freq_max, db_min, db_max, diff_min, diff_max = generate_frames(
            args, tempdir)
        run_gnuplot(
            tsmap, freq_min, freq_max, db_min, db_max, diff_min, diff_max, args, tempdir, graph, -40)
        run_ffmpeg(args, tempdir, mp4)


if __name__ == '__main__':
    main()
