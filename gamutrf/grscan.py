#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Derivative work from:
# https://github.com/ThomasHabets/radiostuff/blob/922944c9a7c9c51a15e369ac07a7f8963b5f67bd/broadband-scan/broadband_scan.grc
import logging
import sys
import threading
import time

import pandas as pd

try:
    from gnuradio import blocks  # pytype: disable=import-error
    from gnuradio import fft  # pytype: disable=import-error
    from gnuradio import gr  # pytype: disable=import-error
    from gnuradio import zeromq  # pytype: disable=import-error
    from gnuradio.fft import window  # pytype: disable=import-error
except ModuleNotFoundError:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme)"
    )
    sys.exit(1)

from gamutrf.grsource import get_source


class grscan(gr.top_block):
    def __init__(
        self,
        freq_end=1e9,
        freq_start=100e6,
        igain=0,
        samp_rate=4e6,
        sweep_sec=30,
        retune_hz=97,
        logaddr="0.0.0.0",  # nosec
        logport=8001,
        sdr="ettus",
        sdrargs=None,
        fft_size=1024,
        retune_intervals=1,
        habets39=None,
    ):
        gr.top_block.__init__(self, "scan", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.freq_end = freq_end
        self.freq_start = freq_start
        self.sweep_sec = sweep_sec
        self.freq_update = 0
        self.no_freq_updates = 0
        self.retune_hz = retune_hz
        self.retune_intervals = retune_intervals
        self.retune_timed_hz = retune_hz / self.retune_intervals

        ##################################################
        # Variables
        ##################################################
        self.fft_size = fft_size
        self.center_freq = freq_start

        ##################################################
        # Blocks
        ##################################################

        def _sweep_pos_to_freq(tune_time):  # nosemgrep
            sweep_pos = (tune_time % self.sweep_sec) / self.sweep_sec
            freq = ((self.freq_end - self.freq_start) * sweep_pos) + self.freq_start
            return freq

        def _retune_worker():
            freq_steps = []
            sweeps = 0
            retune_interval = 1.0 / self.retune_hz
            while True:
                rollover = False
                host_now = time.time()
                sdr_now = self.get_sdr_time_now(self.source_0)
                for i in range(self.retune_intervals):
                    tune_time = sdr_now + (i * retune_interval)
                    freq = _sweep_pos_to_freq(tune_time)
                    # logging.info(
                    #    "tune_time: %f now diff: %f %f %f MHz",
                    #    tune_time,
                    #    tune_time - sdr_now,
                    #    i,
                    #    freq / 1e6,
                    # )
                    if freq_steps and freq < freq_steps[-1]:
                        rollover = True
                        sweeps += 1
                    freq_steps.append(freq)
                    if tune_time == sdr_now:
                        self.set_center_freq(freq)
                    else:
                        self.set_command_time(self.source_0, tune_time)
                        self.set_center_freq(freq)
                        self.clear_command_time(self.source_0)
                if rollover and sweeps > 1:
                    freq_df = pd.DataFrame(freq_steps, columns=["freq"]).sort_values(
                        by=["freq"]
                    )
                    freq_df["freq"] /= 1e6
                    freq_df["diff"] = freq_df["freq"].diff()
                    dmin, dmean, dmax = (
                        freq_df["diff"].min(),
                        freq_df["diff"].mean(),
                        freq_df["diff"].max(),
                    )
                    freq_max = freq_df[freq_df["diff"] == dmax]["freq"].iat[0]
                    logging.info(
                        "tuning step min %f MHz mean %f MHz max %f MHz (at %f MHz)",
                        dmin,
                        dmean,
                        dmax,
                        freq_max,
                    )
                    if dmean > samp_rate / 1e6 / 2:
                        logging.warning(
                            "mean tuning step is greater than --samp-rate/2"
                        )
                remainder_interval_time = (1.0 / self.retune_timed_hz) - (
                    time.time() - host_now
                )
                if remainder_interval_time > 0:
                    time.sleep(remainder_interval_time)
                else:
                    logging.info(
                        "retune interval ran late near %.3f MHz by %.3fs (wanted interval %.3fs)",
                        freq_steps[-1] / 1e6,
                        abs(remainder_interval_time),
                        retune_interval,
                    )
                if rollover:
                    freq_steps = []

        logging.info(f"will scan from {freq_start} to {freq_end}")
        get_source(
            self,
            sdr,
            samp_rate,
            igain,
            agc=False,
            center_freq=freq_start,
            sdrargs=sdrargs,
        )

        self.retune_worker_thread = threading.Thread(target=_retune_worker)
        self.retune_worker_thread.daemon = True
        self.retune_worker_thread.start()

        self.habets39_sweepsinkv_0 = None
        if habets39:
            self.habets39_sweepsinkv_0 = habets39.sweepsinkv(
                "rx_freq", fft_size, samp_rate
            )
        self.fft_vxx_0 = fft.fft_vcc(
            fft_size, True, window.blackmanharris(fft_size), True, 1
        )
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(
            gr.sizeof_gr_complex, fft_size
        )
        zmq_addr = f"tcp://{logaddr}:{logport}"
        logging.info("serving FFT on %s", zmq_addr)
        self.zeromq_pub_sink_0 = zeromq.pub_sink(1, 1, zmq_addr, 100, False, -1, "")
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(fft_size)
        self.blocks_complex_to_mag_0.set_max_output_buffer(16)

        ##################################################
        # Connections
        ##################################################
        # FFT chain
        if self.habets39_sweepsinkv_0:
            self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
            self.connect((self.fft_vxx_0, 0), (self.blocks_complex_to_mag_0, 0))
            self.connect(
                (self.blocks_complex_to_mag_0, 0), (self.habets39_sweepsinkv_0, 0)
            )
            self.connect((self.habets39_sweepsinkv_0, 0), (self.zeromq_pub_sink_0, 0))
            self.connect((self.source_0, 0), (self.blocks_stream_to_vector_0, 0))

    def set_center_freq(self, center_freq):
        if center_freq == self.center_freq:
            return
        self.center_freq = center_freq
        if self.center_freq:
            self.freq_setter(self.source_0, self.center_freq)
            self.freq_update = time.time()
            self.no_freq_updates = 0

    def freq_updated(self, timeout):
        if time.time() - self.freq_update < timeout:
            return True
        self.no_freq_updates += 1
        return self.no_freq_updates < timeout
