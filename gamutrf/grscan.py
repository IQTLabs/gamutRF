#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys
from pathlib import Path

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
        skip_tune_step=0,
        write_samples=0,
        sample_dir="",
        inference_plan_file="",
        inference_output_dir="",
        inference_input_len=2048,
        bucket_range=1.0,
        tuning_ranges="",
        iqtlabs=None,
        wavelearner=None,
    ):
        gr.top_block.__init__(self, "scan", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.freq_end = freq_end
        self.freq_start = freq_start
        self.sweep_sec = sweep_sec
        self.fft_size = fft_size
        self.wavelearner = wavelearner
        self.samp_rate = samp_rate

        ##################################################
        # Blocks
        ##################################################

        logging.info(f"will scan from {freq_start} to {freq_end}")
        self.sources, cmd_port, self.workaround_start_hook = get_source(
            sdr,
            samp_rate,
            igain,
            agc=False,
            center_freq=freq_start,
            sdrargs=sdrargs,
        )

        fft_blocks, fft_roll = self.get_fft_blocks(fft_size, sdr)
        self.fft_blocks = fft_blocks + self.get_db_blocks(fft_size, samp_rate)
        self.samples_blocks = []
        if write_samples:
            Path(sample_dir).mkdir(parents=True, exist_ok=True)
            self.samples_blocks.extend(
                [
                    blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size),
                    iqtlabs.write_freq_samples(
                        "rx_freq",
                        gr.sizeof_gr_complex,
                        fft_size,
                        sample_dir,
                        "samples",
                        write_samples,
                        skip_tune_step,
                        int(samp_rate),
                    ),
                ]
            )
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
        retune_fft = iqtlabs.retune_fft(
            "rx_freq",
            fft_size,
            fft_size,
            int(samp_rate),
            int(freq_start),
            int(freq_end),
            tune_step_hz,
            tune_step_fft,
            skip_tune_step,
            fft_roll,
            -200,
            50,
            sample_dir,
            write_samples,
            bucket_range,
            tuning_ranges,
        )
        self.fft_blocks.append(retune_fft)
        zmq_addr = f"tcp://{logaddr}:{logport}"
        logging.info("serving FFT on %s", zmq_addr)
        self.fft_blocks.append((zeromq.pub_sink(1, 1, zmq_addr, 100, False, 65536, "")))

        self.inference_blocks = []
        if inference_plan_file and inference_output_dir:
            if not self.wavelearner:
                raise ValueError(
                    "trying to use inference but wavelearner not available"
                )
            inference_batch_size = 128
            output_len = 1
            self.inference_blocks = [
                blocks.stream_to_vector(
                    gr.sizeof_gr_complex * 1, inference_batch_size * inference_input_len
                ),
                self.wavelearner.inference(
                    inference_plan_file,
                    True,
                    inference_input_len * inference_batch_size,
                    output_len * inference_batch_size,
                    inference_batch_size,
                ),
                iqtlabs.write_freq_samples(
                    "rx_freq",
                    gr.sizeof_float * output_len,
                    inference_batch_size,
                    inference_output_dir,
                    "inference",
                    int(1e9),
                    0,
                    int(samp_rate),
                ),
            ]

        self.msg_connect((retune_fft, "tune"), (self.sources[0], cmd_port))
        self.connect_blocks(self.sources[0], self.sources[1:])
        for pipeline_blocks in (
            self.fft_blocks,
            self.samples_blocks,
            self.inference_blocks,
        ):
            self.connect_blocks(self.sources[-1], pipeline_blocks)

    def connect_blocks(self, source, other_blocks):
        last_block = source
        for block in other_blocks:
            self.connect((last_block, 0), (block, 0))
            last_block = block

    def get_db_blocks(self, fft_size, samp_rate):
        scale = 1.0 / (samp_rate * sum(self.get_window(fft_size)) ** 2)
        return [
            blocks.complex_to_mag_squared(fft_size),
            blocks.multiply_const_vff([scale] * fft_size),
            blocks.nlog10_ff(10, fft_size, 0),
        ]

    def get_window(self, fft_size):
        return window.hann(fft_size)

    def get_fft_blocks(self, fft_size, sdr):
        if self.wavelearner:
            fft_batch_size = 256
            return (
                [
                    blocks.stream_to_vector(
                        gr.sizeof_gr_complex, fft_batch_size * fft_size
                    ),
                    blocks.multiply_const_vff(
                        [val for val in self.get_window(fft_size) for _ in range(2)]
                        * fft_batch_size
                    ),
                    self.wavelearner.fft(
                        int(fft_batch_size * fft_size), (fft_size), True
                    ),
                    blocks.vector_to_stream(
                        gr.sizeof_gr_complex * fft_size, fft_batch_size
                    ),
                ],
                True,
            )
        return (
            [
                blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size),
                fft.fft_vcc(fft_size, True, self.get_window(fft_size), True, 1),
            ],
            False,
        )

    def start(self):
        super().start()
        self.workaround_start_hook(self)
