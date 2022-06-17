#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Derivative work from:
# https://github.com/ThomasHabets/radiostuff/blob/922944c9a7c9c51a15e369ac07a7f8963b5f67bd/broadband-scan/broadband_scan.grc
import functools
import threading
import time

from gnuradio import analog  # pytype: disable=import-error
from gnuradio import blocks  # pytype: disable=import-error
from gnuradio import fft  # pytype: disable=import-error
from gnuradio import gr  # pytype: disable=import-error
from gnuradio import soapy  # pytype: disable=import-error
from gnuradio import uhd  # pytype: disable=import-error
from gnuradio.fft import window  # pytype: disable=import-error

from gamutrf.utils import ETTUS_ANT


class grscan(gr.top_block):

    def __init__(self, freq_end=1e9, freq_start=100e6, igain=0, samp_rate=4e6, sweep_sec=30,
                 logaddr='127.0.0.1', logport=8001, sdr='ettus', ettusargs='', habets39=None):
        gr.top_block.__init__(self, 'scan', catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.freq_end = freq_end
        self.freq_start = freq_start
        self.igain = igain
        self.samp_rate = samp_rate
        self.sweep_sec = sweep_sec
        self.freq_update = 0
        self.no_freq_updates = 0

        ##################################################
        # Variables
        ##################################################
        self.sweep_freq = sweep_freq = 1/sweep_sec
        self.scan_samp_rate = scan_samp_rate = 32000
        self.fft_size = fft_size = 1024
        self.center_freq = freq_start

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
                ','.join((ettusargs, '')),
                uhd.stream_args(
                    cpu_format='fc32',
                    args='',
                    channels=list(range(0, 1)),
                ),
            )
            self.source.set_time_now(
                uhd.time_spec(time.time()), uhd.ALL_MBOARDS)
            self.source.set_antenna(ETTUS_ANT, 0)
            self.set_samp_rate(samp_rate)
            self.set_igain(igain)
            self.freq_setter = lambda x: self.source.set_center_freq(x, 0)
        elif sdr in ('bladerf', 'lime'):
            dev = f'driver={sdr}'
            stream_args = ''
            tune_args = ['']
            settings = ['']
            self.source = soapy.source(dev, 'fc32', 1, '',
                                       stream_args, tune_args, settings)
            self.source.set_sample_rate(0, samp_rate)
            self.source.set_bandwidth(0, 0.0)
            self.source.set_frequency_correction(0, 0)
            self.source.set_gain(0, igain)
            self.freq_setter = lambda x: self.source.set_frequency(0, x)

        self.center_freq_thread = threading.Thread(target=_center_freq_probe)
        self.center_freq_thread.daemon = True
        self.center_freq_thread.start()

        self.habets39_sweepsinkv_0 = None
        if habets39:
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
        if self.habets39_sweepsinkv_0:
            self.connect((self.blocks_complex_to_mag_0, 0),
                         (self.habets39_sweepsinkv_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0),
                     (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.blocks_throttle_0, 0),
                     (self.blocks_probe_signal_x_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_complex_to_mag_0, 0))
        if self.habets39_sweepsinkv_0:
            self.connect((self.habets39_sweepsinkv_0, 0),
                         (self.blocks_udp_sink_0, 0))
        if self.source:
            self.connect((self.source, 0),
                         (self.blocks_stream_to_vector_0, 0))

    def set_igain(self, igain):
        self.igain = igain
        self.source.set_gain(self.igain, 0)  # pytype: disable=attribute-error

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        # pytype: disable=attribute-error
        self.source.set_samp_rate(self.samp_rate)

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        if self.center_freq:
            self.freq_setter(self.center_freq)
            self.freq_update = time.time()
            self.no_freq_updates = 0

    def freq_updated(self, timeout):
        if (time.time() - self.freq_update) < timeout:
            return True
        self.no_freq_updates += 1
        return self.no_freq_updates < timeout
