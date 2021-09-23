#!/usr/bin/python3

import os
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd

MINF = 100e6
FRAMERATE = 5
XTICS = 40
EXT = '.csv'

GNUPLOT_CMDS_HEADER = [
   'set terminal png truecolor rounded size 1920,720 enhanced',
   'set ytics format "%.4f"',
   'set xtics rotate by 90 right',
   'set grid xtics',
   'set grid mxtics',
   'set grid ytics',
   'set xlabel "freq (MHz)"',
   'set ylabel "power (dB)"']

if len(sys.argv) == 1:
    print('need CSV')
    sys.exit(1)

CSV = sys.argv[1]
if not CSV.endswith(EXT):
    print(f'{CSV} must be .csv')
    sys.exit(1)

if not os.path.exists(CSV):
    print(f'{CSV} must exist')
    sys.exit(1)

MP4 = CSV.replace(EXT, '.mp4')

df = pd.read_csv(CSV, header=None, delim_whitespace=True)
print('read', CSV)
df.columns = ['ts', 'freq', 'db']
df = df[df['freq'] >= MINF]
df['db'] = 20 * np.log10(df['db'])
df['freq'] /= 1e6
freqdiff = df['freq'].diff().abs()
df['frame'] = 0
# Detect tuning wraparound, where frequency changed by more than 100MHz
df.loc[freqdiff > 100, ['frame']] = 1
df['frame'] = df['frame'].cumsum().fillna(0).astype(np.uint64)
df.set_index('frame', inplace=True)
df['ts'] = df.groupby('frame', sort=False)['ts'].transform(min)

with tempfile.TemporaryDirectory() as tempdir:
    tsmap = []

    for frame, frame_df in df.groupby('frame'):
        ts = frame_df['ts'].iat[0]
        tsc = time.ctime(ts)
        print(frame, ts, tsc)
        gnuplot_df = frame_df[['freq', 'db']]
        frame_f = os.path.join(str(tempdir), str(ts))
        tsmap.append((frame, ts, tsc, frame_f))
        gnuplot_df.to_csv(frame_f, index=False, sep='\t')

    print('creating gnuplot commands')
    db_min = df.db.min()
    db_max = df.db.max()
    freq_min = df.freq.min()
    freq_max = df.freq.max()
    xtics = (freq_max - freq_min) / XTICS

    gnuplot_cmds = GNUPLOT_CMDS_HEADER
    gnuplot_cmds.extend([
        f'set xtics {xtics}',
        f'set xrange [{freq_min}:{freq_max}]',
        f'set yrange [{db_min}:{db_max}]'])
    for frame, ts, tsc, frame_f in tsmap:
        gnuplot_cmds.extend([
            f'set output "{tempdir}/{frame:06}.png"',
            f'plot [{freq_min}:{freq_max}] "{frame_f}" using 1:2 with linespoints title "{tsc}"'])

    plot_f = os.path.join(str(tempdir), 'plot.cmd')
    with open(plot_f, 'w', encoding='utf8') as plot_c:
        plot_c.write('\n'.join(gnuplot_cmds))

    print('running gnuplot')
    subprocess.check_call(['gnuplot', plot_f])
    print('running ffmpeg')
    subprocess.check_call([
        'ffmpeg',
        '-loglevel', 'error',
        '-framerate', str(FRAMERATE),
        '-pattern_type', 'glob', '-i', os.path.join(str(tempdir), '*.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-r', '30', '-y', MP4])
