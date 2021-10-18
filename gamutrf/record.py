#!/usr/bin/python3

import subprocess


# Convert I/Q sample recording to "gnuradio" I/Q format (float)
# Default input format is signed, 16 bit I/Q (bladeRF-cli default)
def raw2grraw(in_file, gr_file, sample_rate, in_file_bits=16, in_file_fmt='signed-integer'):
    raw_args = ['-t', 'raw', '-r', str(sample_rate), '-c', str(1)]
    return subprocess.check_call(
        ['sox'] + raw_args + ['-b', str(in_file_bits), '-e', in_file_fmt, in_file] + raw_args + ['-e', 'float', gr_file])


# Record I/Q from bladeRF
# Example:
#   bladerf_record('bladerf.raw', 108e6, 10e6, sample_rate=1e6, agc=True) # record center 108MHz, 1MHz, 10e6 samples with AGC.
#   raw2grraw('bladerf.raw', 'gnuradio.raw', 1e6) # Convert to gnuradio float format.
def bladerf_record(sample_file, center_freq, sample_count, sample_rate=1e6, sample_bw=1e6, gain=0, agc=True):
    gain_args = [
        '-e', 'set agc rx off',
        '-e', f'set gain rx {gain}',
    ]
    if agc:
        gain_args = [
            '-e', 'set agc rx on',
        ]
    args = [
        'bladeRF-cli',
    ] + gain_args + [
        '-e', f'set samplerate rx {sample_rate}',
        '-e', f'set bandwidth rx {sample_bw}',
        '-e', f'set frequency rx {center_freq}',
        '-e', f'rx config file={sample_file} format=bin n={sample_count}',
        '-e', 'rx start',
        '-e', 'rx wait']
    return subprocess.check_call(args)


# Record I/Q from Ettus
# example:
#   uhd_record('uhd.raw', 108e6, 10e6, sample_rate=1e6, gain=30) # record center 108MHz, 1MHz, 10e6 samples at 30dB gain
#   raw2grraw('uhd.raw', 'gnuradio.raw', 1e6) # Convert to gnuradio float format.
def uhd_record(sample_file, center_freq, sample_count, sample_rate=1e6, sample_bw=1e6, gain=0):
    args = [
        '/usr/lib/uhd/examples/rx_samples_to_file',
        '--file', sample_file,
        '--rate', str(sample_rate),
        '--bw', str(sample_bw),
        '--nsamps', str(int(sample_count)),
        '--freq', str(center_freq),
        '--gain', str(gain)]
    return subprocess.check_call(args)
