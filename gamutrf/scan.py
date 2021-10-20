#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Derivative work from:
# https://github.com/ThomasHabets/radiostuff/blob/922944c9a7c9c51a15e369ac07a7f8963b5f67bd/broadband-scan/broadband_scan.grc
import functools
import signal
import sys
import threading
import time
from argparse import ArgumentParser

import habets39  # pytype: disable=import-error
from gnuradio import analog  # pytype: disable=import-error
from gnuradio import blocks  # pytype: disable=import-error
from gnuradio import eng_notation  # pytype: disable=import-error
from gnuradio import fft  # pytype: disable=import-error
from gnuradio import gr  # pytype: disable=import-error
from gnuradio import uhd  # pytype: disable=import-error
from gnuradio.eng_arg import eng_float  # pytype: disable=import-error
from gnuradio.eng_arg import intx  # pytype: disable=import-error
from gnuradio.fft import window  # pytype: disable=import-error
from gnuradio import soapy  # pytype: disable=import-error
# TODO: add test/pylint coverage with gnuradio


class scan(gr.top_block):

    def __init__(self, freq_end=1e9, freq_start=100e6, igain=0, samp_rate=4e6, sweep_sec=30,
                 logaddr='127.0.0.1', logport=8001, sdr='ettus'):
        gr.top_block.__init__(self, 'scan', catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.freq_end = freq_end
        self.freq_start = freq_start
        self.igain = igain
        self.samp_rate = samp_rate
        self.sweep_sec = sweep_sec

        ##################################################
        # Variables
        ##################################################
        self.sweep_freq = sweep_freq = 1/sweep_sec
        self.scan_samp_rate = scan_samp_rate = 32000
        self.fft_size = fft_size = 1024
        self.center_freq = center_freq = freq_start

        ##################################################
        # Blocks
        ##################################################
        self.blocks_probe_signal_x_0 = blocks.probe_signal_f()

        def _center_freq_probe():
            while True:

                val = self.blocks_probe_signal_x_0.level()
                try:
                    try:
                        self.doc.add_next_tick_callback(
                            functools.partial(self.set_center_freq, val))
                    except AttributeError:
                        self.set_center_freq(val)
                except AttributeError:
                    pass
                time.sleep(1.0 / (97))

        self.freq_setter = None
        self.source = None
        if sdr == 'ettus':
            self.source = uhd.usrp_source(
                ','.join(('', '')),
                uhd.stream_args(
                    cpu_format='fc32',
                    args='',
                    channels=list(range(0, 1)),
                ),
            )
            self.source.set_time_now(
                uhd.time_spec(time.time()), uhd.ALL_MBOARDS)
            self.source.set_antenna('TX/RX', 0)
            self.set_samp_rate(samp_rate)
            self.set_igain(igain)
            self.freq_setter = lambda x: self.source.set_center_freq(x, 0)
        elif sdr == 'bladerf':
            dev = 'driver=bladerf'
            stream_args = ''
            tune_args = ['']
            settings = ['']
            self.source = soapy.source(dev, "fc32", 1, '',
                stream_args, tune_args, settings)
            self.source.set_sample_rate(0, samp_rate)
            self.source.set_bandwidth(0, 0.0)
            self.source.set_frequency_correction(0, 0)
            self.source.set_gain(0, igain)
            self.freq_setter = lambda x: self.source.set_frequency(0, x)

        _center_freq_thread = threading.Thread(target=_center_freq_probe)
        _center_freq_thread.daemon = True
        _center_freq_thread.start()

        self.habets39_sweepsinkv_0 = habets39.sweepsinkv(
            'rx_freq', fft_size, samp_rate)
        self.fft_vxx_0 = fft.fft_vcc(
            fft_size, True, window.blackmanharris(fft_size), True, 1)
        self.blocks_throttle_0 = blocks.throttle(
            gr.sizeof_float*1, scan_samp_rate, True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(
            gr.sizeof_gr_complex*1, fft_size)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(
            freq_end-freq_start)
        self.blocks_udp_sink_0 = blocks.udp_sink(
            gr.sizeof_char*1, logaddr, logport, 1472, True)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(fft_size)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff(freq_start)
        self.analog_sig_source_x_0 = analog.sig_source_f(
            scan_samp_rate, analog.GR_SAW_WAVE, sweep_freq, 1, 0, 0)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0),
                     (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0),
                     (self.blocks_throttle_0, 0))
        self.connect((self.blocks_complex_to_mag_0, 0),
                     (self.habets39_sweepsinkv_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0),
                     (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.blocks_throttle_0, 0),
                     (self.blocks_probe_signal_x_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.habets39_sweepsinkv_0, 0),
                     (self.blocks_udp_sink_0, 0))
        self.connect((self.source, 0),
                     (self.blocks_stream_to_vector_0, 0))

    def get_freq_end(self):
        return self.freq_end

    def set_freq_end(self, freq_end):
        self.freq_end = freq_end
        self.blocks_multiply_const_vxx_0.set_k(self.freq_end-self.freq_start)

    def get_freq_start(self):
        return self.freq_start

    def set_freq_start(self, freq_start):
        self.freq_start = freq_start
        self.set_center_freq(self.freq_start)
        self.blocks_add_const_vxx_0.set_k(self.freq_start)
        self.blocks_multiply_const_vxx_0.set_k(self.freq_end-self.freq_start)

    def get_igain(self):
        return self.igain

    def set_igain(self, igain):
        self.igain = igain
        self.source.set_gain(self.igain, 0)  # pytype: disable=attribute-error

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        # pytype: disable=attribute-error
        self.source.set_samp_rate(self.samp_rate)

    def get_sweep_sec(self):
        return self.sweep_sec

    def set_sweep_sec(self, sweep_sec):
        self.sweep_sec = sweep_sec
        self.set_sweep_freq(1/self.sweep_sec)

    def get_sweep_freq(self):
        return self.sweep_freq

    def set_sweep_freq(self, sweep_freq):
        self.sweep_freq = sweep_freq
        self.analog_sig_source_x_0.set_frequency(self.sweep_freq)

    def get_scan_samp_rate(self):
        return self.scan_samp_rate

    def set_scan_samp_rate(self, scan_samp_rate):
        self.scan_samp_rate = scan_samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.scan_samp_rate)
        self.blocks_throttle_0.set_sample_rate(self.scan_samp_rate)

    def get_fft_size(self):
        return self.fft_size

    def set_fft_size(self, fft_size):
        self.fft_size = fft_size

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        if self.center_freq:
            self.freq_setter(self.center_freq)


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--freq-end', dest='freq_end', type=eng_float, default=eng_notation.num_to_str(float(1e9)),
        help='Set freq_end [default=%(default)r]')
    parser.add_argument(
        '--freq-start', dest='freq_start', type=eng_float, default=eng_notation.num_to_str(float(100e6)),
        help='Set freq_start [default=%(default)r]')
    parser.add_argument(
        '--igain', dest='igain', type=intx, default=0,
        help='Set igain[default=%(default)r]')
    parser.add_argument(
        '--samp-rate', dest='samp_rate', type=eng_float, default=eng_notation.num_to_str(float(4e6)),
        help='Set samp_rate [default=%(default)r]')
    parser.add_argument(
        '--sweep-sec', dest='sweep_sec', type=intx, default=30,
        help='Set sweep_sec [default=%(default)r]')
    parser.add_argument(
        '--logaddr', dest='logaddr', type=str, default='127.0.0.1',
        help='Log UDP results to this address')
    parser.add_argument(
        '--logport', dest='logport', type=int, default=8001,
        help='Log UDP results to this port')
    parser.add_argument(
        '--sdr', dest='sdr', type=str, default='ettus',
        help='SDR to use (ettus or bladerf)')
    return parser


def main(top_block_cls=scan, options=None):
    if options is None:
        options = argument_parser().parse_args()
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print('Error: failed to enable real-time scheduling.')

    if options.freq_start > options.freq_end:
        print('Error: freq_start must be less than freq_end')
        sys.exit(1)

    if options.freq_end > 6e9:
        print('Error: freq_end must be less than 6GHz')
        sys.exit(1)

    tb = top_block_cls(freq_end=options.freq_end, freq_start=options.freq_start,
                       igain=options.igain, samp_rate=options.samp_rate,
                       sweep_sec=options.sweep_sec,
                       logaddr=options.logaddr, logport=options.logport,
                       sdr=options.sdr)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
