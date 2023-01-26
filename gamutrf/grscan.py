#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys

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
        samp_rate=4.096e6,
        sweep_sec=30,
        logaddr="0.0.0.0",  # nosec
        logport=8001,
        sdr="ettus",
        sdrargs=None,
        fft_size=1024,
        tune_overlap=0.5,
        tune_step_fft=0,
        iqtlabs=None,
    ):
        gr.top_block.__init__(self, "scan", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.freq_end = freq_end
        self.freq_start = freq_start
        self.sweep_sec = sweep_sec
        self.fft_size = fft_size

        ##################################################
        # Blocks
        ##################################################

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

        self.retune_fft = None
        if iqtlabs:
            freq_range = freq_end - freq_start
            tune_step_hz = int(samp_rate * tune_overlap)
            if tune_step_fft:
                logging.info(
                    f"retuning across {freq_range/1e6}MHz every {tune_step_fft} FFTs"
                )
            else:
                target_retune_hz = freq_range / self.sweep_sec / tune_step_hz
                fft_rate = int(samp_rate / fft_size)
                tune_step_fft = int(fft_rate / target_retune_hz)
                logging.info(
                    f"retuning across {freq_range/1e6}MHz in {self.sweep_sec}s, requires retuning at {target_retune_hz}Hz in {tune_step_hz/1e6}MHz steps ({tune_step_fft} FFTs)"
                )
            self.retune_fft = iqtlabs.retune_fft(
                "rx_freq",
                fft_size,
                int(samp_rate),
                int(freq_start),
                int(freq_end),
                tune_step_hz,
                tune_step_fft,
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

        ##################################################
        # Connections
        ##################################################
        # FFT chain
        if self.retune_fft:
            self.msg_connect((self.retune_fft, "tune"), (self.source_0, self.cmd_port))
            self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
            self.connect((self.fft_vxx_0, 0), (self.blocks_complex_to_mag_0, 0))
            self.connect((self.blocks_complex_to_mag_0, 0), (self.retune_fft, 0))
            self.connect((self.retune_fft, 0), (self.zeromq_pub_sink_0, 0))
            self.connect((self.source_0, 0), (self.blocks_stream_to_vector_0, 0))
