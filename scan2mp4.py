#!/usr/bin/python3

import os
import subprocess
import sys
import tempfile
import time
from statistics import mean

MINF = 100e6
FRAMETIME = 60
FRAMERATE = 5
EXT = '.csv'

if len(sys.argv) == 1:
    print('need CSV')
    sys.exit(1)

CSV = sys.argv[1]
if not CSV.endswith(EXT):
    print('must be .csv')
    sys.exit(1)

if not os.path.exists(CSV):
    print('%s must exist' % CSV)
    sys.exit(1)

MP4 = CSV.replace(EXT, '.mp4')

GNUPLOT_HEADER = """
set terminal png truecolor rounded size 1920,720 enhanced
set xtics 20
set ytics format "%.4f"
set xtics rotate by 90 right
set mxtics 5
set grid xtics
set grid mxtics
set grid ytics
"""


with tempfile.TemporaryDirectory() as tempdir:
    tses = []
    min_freq = 10e9
    max_freq = 0
    peak_pws = []
    lastframets = None
    frames = 0

    with open(CSV) as f:
        frame = {}
        lastfq = None
        for l in f:
            try:
                ts, fq, pw = (float(x) for x in l.strip().split())
            except ValueError as err:
                print('ignoring truncated line: %s' % l)
                continue
            if fq < MINF:
                continue
            fq /= 1e6
            min_freq = min(min_freq, fq)
            max_freq = max(max_freq, fq)
            if lastfq and abs(fq - lastfq) > 1e2 and (lastframets is None or ts - lastframets >= FRAMETIME):
                frames += 1
                lastframets = ts
                frame_f = os.path.join(str(tempdir), str(ts))
                frame_peak_pw = max(frame.values())
                print(frames, ts, time.ctime(ts), frame_peak_pw)
                tses.append(ts)
                peak_pws.append(frame_peak_pw)
                with open(frame_f, 'w') as w:
                    for x, y in sorted(frame.items()):
                        w.write('%u\t%f\n' % (x, y))
                frame = {}
            frame[fq] = pw
            lastfq = fq

    assert tses

    print('creating gnuplot commands')
    plot_f = os.path.join(str(tempdir), 'plot.cmd')
    threshold_95 = int(len(peak_pws) * 0.95)
    try:
        peak_pw_95 = sorted(peak_pws)[:threshold_95][-1]
    except IndexError:
        peak_pw_95 = max(peak_pws)
    with open(plot_f, 'w') as cmd:
        cmd.write(GNUPLOT_HEADER)
        cmd.write('set yrange [0:%.4f]\n' % peak_pw_95)
        cmd.write('set xrange [%f:%f]\n' % (min_freq, max_freq))
        for i, ts in enumerate(tses, start=0):
            f = os.path.join(str(tempdir), str(ts))
            tsc = time.ctime(int(ts))
            cmd.write(('''

set output "%s/%6.6u.png"
plot [%u:%u] "%s" using 1:2 with points title "%s"
''' % (str(tempdir), i, min_freq, max_freq, f, tsc)))

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
