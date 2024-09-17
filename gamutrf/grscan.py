#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import logging
import sys
from pathlib import Path
import pbr.version
import webcolors

try:
    from gnuradio import analog  # pytype: disable=import-error
    from gnuradio import filter as grfilter  # pytype: disable=import-error
    from gnuradio import blocks  # pytype: disable=import-error
    from gnuradio import fft  # pytype: disable=import-error
    from gnuradio import gr  # pytype: disable=import-error
    from gnuradio import zeromq  # pytype: disable=import-error
    from gnuradio.fft import window  # pytype: disable=import-error
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)

from gamutrf.dc_spike import dc_spike_detrend, dc_spike_remove
from gamutrf.grsource import get_source
from gamutrf.grinferenceoutput import inferenceoutput
from gamutrf.grpduzmq import pduzmq
from gamutrf.utils import endianstr


class grscan(gr.top_block):
    def __init__(
        self,
        bucket_range=1.0,
        colormap=16,
        compass=False,
        correct_iq=False,
        db_clamp_ceil=50,
        db_clamp_floor=-200,
        dc_block_len=0,
        dc_block_long=False,
        dc_spike_detrend_length=0,
        dc_spike_remove_ratio=0.0,
        dc_ettus_auto_offset=True,
        description="",
        external_gps_server="",
        external_gps_server_port=8888,
        fft_batch_size=256,
        fft_processor_affinity=0,
        iq_zmq_addr="0.0.0.0",
        iq_zmq_port=10002,
        freq_end=1e9,
        freq_start=100e6,
        gps_server="",
        igain=0,
        inference_addr="0.0.0.0",  # nosec
        inference_batch=1,
        inference_min_confidence=0.5,
        inference_min_db=-200,
        inference_model_name="",
        inference_model_server="",
        inference_output_dir="",
        inference_port=10001,
        inference_text_color="",
        iq_inference_background=True,
        iq_inference_model_name="",
        iq_inference_model_server="",
        iq_inference_squelch_db=None,
        iq_inference_squelch_alpha=1e-4,
        iq_power_inference=False,
        iqtlabs=None,
        fft_zmq_addr="0.0.0.0",  # nosec
        fft_zmq_port=10000,
        low_power_hold_down=False,
        mqtt_server="",
        n_image=0,
        n_inference=0,
        nfft=1024,
        peak_fft_range=0,
        pretune=False,
        rotate_secs=0,
        samp_rate=4.096e6,
        sample_dir="",
        scaling="spectrum",
        sdr="ettus",
        sdrargs=None,
        sigmf=True,
        skip_tune_step=0,
        slew_rx_time=True,
        sweep_sec=30,
        tag_now=False,
        tune_dwell_ms=0,
        tune_jitter_hz=0,
        tune_step_fft=0,
        tuneoverlap=0.5,
        tuning_ranges="",
        use_external_gps=False,
        use_external_heading=False,
        vkfft=False,
        wavelearner=None,
        write_fft_points=False,
        write_samples=0,
    ):
        gr.top_block.__init__(self, "scan", catch_exceptions=True)

        if description:
            description = description.strip('"')

        ##################################################
        # Parameters
        ##################################################
        self.wavelearner = wavelearner
        self.iqtlabs = iqtlabs

        ##################################################
        # Blocks
        ##################################################
        if iqtlabs is not None:
            griqtlabs_path = os.path.realpath(
                glob.glob("/usr/local/**/libgnuradio-iqtlabs.so", recursive=True)[0]
            )
            pbr_version = pbr.version.VersionInfo("gamutrf").version_string()
            logging.info(f"gamutrf {pbr_version} with gr-iqtlabs {griqtlabs_path}")

        tune_step_fft, tune_step_hz, peak_fft_range = self.calc_rates(
            freq_start,
            freq_end,
            sweep_sec,
            samp_rate,
            nfft,
            tuneoverlap,
            tune_dwell_ms,
            tune_step_fft,
            peak_fft_range,
        )
        fft_batch_size, self.retune_pre_fft, self.fft_blocks = self.get_fft_blocks(
            samp_rate,
            tune_jitter_hz,
            vkfft,
            fft_batch_size,
            nfft,
            freq_start,
            freq_end,
            tune_step_hz,
            tune_step_fft,
            skip_tune_step,
            tuning_ranges,
            pretune,
            fft_processor_affinity,
            low_power_hold_down,
            slew_rx_time,
            tag_now,
        )
        self.fft_blocks = (
            self.get_dc_blocks(
                correct_iq,
                dc_block_len,
                dc_block_long,
                dc_spike_detrend_length,
                dc_spike_remove_ratio,
                fft_batch_size,
                nfft,
            )
            + self.fft_blocks
            + self.get_db_blocks(nfft, samp_rate, scaling)
        )
        fft_dir = ""
        if write_fft_points and write_samples:
            fft_dir = sample_dir
        retune_fft = self.iqtlabs.retune_fft(
            tag="rx_freq",
            nfft=nfft,
            samp_rate=int(samp_rate),
            tune_jitter_hz=int(tune_jitter_hz),
            freq_start=int(freq_start),
            freq_end=int(freq_end),
            tune_step_hz=tune_step_hz,
            tune_step_fft=tune_step_fft,
            skip_tune_step_fft=skip_tune_step,
            fft_min=db_clamp_floor,
            fft_max=db_clamp_ceil,
            sdir=fft_dir,
            write_step_fft=write_samples,
            bucket_range=bucket_range,
            tuning_ranges=tuning_ranges,
            description=description,
            rotate_secs=rotate_secs,
            pre_fft=pretune,
            tag_now=tag_now,
            low_power_hold_down=(not pretune and low_power_hold_down),
            slew_rx_time=slew_rx_time,
            peak_fft_range=peak_fft_range,
        )
        self.freq_start, self.freq_end, initial_freq, stare = self.get_freqs(
            self.retune_pre_fft, retune_fft
        )
        logging.info(
            f"will scan from {self.freq_start} to {self.freq_end} stare {stare}"
        )
        self.sources, cmd_port, self.workaround_start_hook = get_source(
            sdr,
            samp_rate,
            igain,
            nfft,
            tune_step_fft,
            agc=False,
            center_freq=initial_freq,
            sdrargs=sdrargs,
            dc_ettus_auto_offset=dc_ettus_auto_offset,
        )
        if iq_zmq_port:
            iq_zmq_block_addr = f"tcp://{iq_zmq_addr}:{iq_zmq_port}"
            logging.info("serving I/Q samples and tags on %s", iq_zmq_block_addr)
            iq_zmq_block = zeromq.pub_sink(
                gr.sizeof_gr_complex,
                fft_batch_size * nfft,
                iq_zmq_block_addr,
                100,
                True,
                65536,
                "",
            )
            self.connect((self.retune_pre_fft, 0), (iq_zmq_block, 0))
        last_db_block = self.fft_blocks[-1]
        self.samples_blocks = []
        self.write_samples_block = None
        if write_samples:
            Path(sample_dir).mkdir(parents=True, exist_ok=True)
            samples_vlen = fft_batch_size * nfft
            self.samples_blocks.extend(
                [
                    # blocks.vector_to_stream(
                    #    gr.sizeof_gr_complex, fft_batch_size * nfft
                    # ),
                    # blocks.complex_to_interleaved_short(False, 32767),
                    # blocks.stream_to_vector(gr.sizeof_short, nfft * 2),
                    self.iqtlabs.write_freq_samples(
                        "rx_freq",
                        gr.sizeof_gr_complex,
                        "_".join(("cf32", endianstr())),
                        samples_vlen,
                        sample_dir,
                        "samples",
                        int(write_samples / samples_vlen),
                        skip_tune_step,
                        int(samp_rate),
                        rotate_secs,
                        igain,
                        sigmf,
                        zstd=True,
                        rotate=False,
                        description=description,
                    ),
                ]
            )
            self.write_samples_block = self.samples_blocks[-1]

        self.fft_blocks.append(retune_fft)
        fft_zmq_block_addr = f"tcp://{fft_zmq_addr}:{fft_zmq_port}"
        self.pduzmq_block = pduzmq(fft_zmq_block_addr)
        logging.info("serving FFT on %s", fft_zmq_block_addr)

        self.image_inference_block, self.iq_inference_block, self.inference_blocks = (
            self.get_inference(
                colormap,
                inference_batch,
                inference_min_confidence,
                inference_min_db,
                inference_model_name,
                inference_model_server,
                inference_output_dir,
                inference_text_color,
                iq_inference_background,
                iq_inference_model_name,
                iq_inference_model_server,
                iq_power_inference,
                n_image,
                n_inference,
                nfft,
                rotate_secs,
                samp_rate,
                tune_step_fft,
            )
        )

        self.inference_output_block = self.connect_inference(
            compass,
            external_gps_server,
            external_gps_server_port,
            fft_batch_size,
            gps_server,
            self.image_inference_block,
            inference_addr,
            self.inference_blocks,
            inference_output_dir,
            inference_port,
            self.iq_inference_block,
            iq_inference_squelch_alpha,
            iq_inference_squelch_db,
            last_db_block,
            mqtt_server,
            nfft,
            retune_fft,
            self.retune_pre_fft,
            stare,
            use_external_gps,
            use_external_heading,
            self.write_samples_block,
        )

        if pretune:
            self.msg_connect((self.retune_pre_fft, "tune"), (self.sources[0], cmd_port))
            self.msg_connect((self.retune_pre_fft, "tune"), (retune_fft, "cmd"))
        else:
            self.msg_connect((retune_fft, "tune"), (self.sources[0], cmd_port))
        self.msg_connect((retune_fft, "json"), (self.pduzmq_block, "json"))
        self.connect_blocks(self.sources[0], self.sources[1:])
        self.connect_blocks(self.sources[-1], self.fft_blocks)
        self.connect_blocks(self.retune_pre_fft, self.samples_blocks)

    def get_inference(
        self,
        colormap,
        inference_batch,
        inference_min_confidence,
        inference_min_db,
        inference_model_name,
        inference_model_server,
        inference_output_dir,
        inference_text_color,
        iq_inference_background,
        iq_inference_model_name,
        iq_inference_model_server,
        iq_power_inference,
        n_image,
        n_inference,
        nfft,
        rotate_secs,
        samp_rate,
        tune_step_fft,
    ):
        inference_blocks = []
        iq_inference_block = None
        image_inference_block = None

        if inference_output_dir:
            Path(inference_output_dir).mkdir(parents=True, exist_ok=True)

        if (inference_model_server and inference_model_name) or inference_output_dir:
            if inference_text_color:
                wc = webcolors.name_to_rgb(inference_text_color, "css3")
                inference_text_color = ",".join(
                    [str(c) for c in [wc.blue, wc.green, wc.red]]
                )
            image_inference_block = self.iqtlabs.image_inference(
                tag="rx_freq",
                vlen=nfft,
                x=640,
                y=640,
                image_dir=inference_output_dir,
                convert_alpha=255,
                norm_alpha=0,
                norm_beta=1,
                norm_type=32,  # cv::NORM_MINMAX = 32
                colormap=colormap,  # cv::COLORMAP_VIRIDIS = 16, cv::COLORMAP_TURBO = 20,
                interpolation=1,  # cv::INTER_LINEAR = 1,
                flip=0,  # 0 means flipping around the x-axis
                min_peak_points=inference_min_db,
                model_server=inference_model_server,
                model_names=inference_model_name,
                confidence=inference_min_confidence,
                max_rows=tune_step_fft,
                rotate_secs=rotate_secs,
                n_image=n_image,
                n_inference=n_inference,
                samp_rate=int(samp_rate),
                text_color=inference_text_color,
            )
            inference_blocks.append(image_inference_block)
        if iq_inference_model_server and iq_inference_model_name:
            iq_inference_block = self.iqtlabs.iq_inference(
                tag="rx_freq",
                vlen=nfft,
                n_vlen=1,
                sample_buffer=tune_step_fft,
                min_peak_points=inference_min_db,
                model_server=iq_inference_model_server,
                model_names=iq_inference_model_name,
                confidence=inference_min_confidence,
                n_inference=n_inference,
                samp_rate=int(samp_rate),
                power_inference=iq_power_inference,
                background=iq_inference_background,
                batch=inference_batch,
            )
            inference_blocks.append(iq_inference_block)
        return (image_inference_block, iq_inference_block, inference_blocks)

    def connect_inference(
        self,
        compass,
        external_gps_server,
        external_gps_server_port,
        fft_batch_size,
        gps_server,
        image_inference_block,
        inference_addr,
        inference_blocks,
        inference_output_dir,
        inference_port,
        iq_inference_block,
        iq_inference_squelch_alpha,
        iq_inference_squelch_db,
        last_db_block,
        mqtt_server,
        nfft,
        retune_fft,
        retune_pre_fft,
        stare,
        use_external_gps,
        use_external_heading,
        write_samples_block,
    ):
        if not inference_blocks:
            return None
        inference_zmq_addr = f"tcp://{inference_addr}:{inference_port}"
        inference_output_block = inferenceoutput(
            "inferencemqtt",
            inference_zmq_addr,
            mqtt_server,
            compass,
            gps_server,
            use_external_gps,
            use_external_heading,
            external_gps_server,
            external_gps_server_port,
            inference_output_dir,
        )
        if iq_inference_block:
            if iq_inference_squelch_db is not None:
                squelch_blocks = self.wrap_batch(
                    [
                        analog.pwr_squelch_cc(
                            iq_inference_squelch_db,
                            iq_inference_squelch_alpha,
                            0,
                            False,
                        )
                    ],
                    fft_batch_size,
                    nfft,
                ) + [iq_inference_block]
                self.connect_blocks(retune_pre_fft, squelch_blocks)
            else:
                self.connect((retune_pre_fft, 0), (iq_inference_block, 0))
            self.connect((last_db_block, 0), (iq_inference_block, 1))
            if write_samples_block:
                self.msg_connect(
                    (iq_inference_block, "inference"),
                    (write_samples_block, "inference"),
                )

        if image_inference_block:
            if stare:
                self.connect((last_db_block, 0), (image_inference_block, 0))
            else:
                self.connect((retune_fft, 0), (image_inference_block, 0))
        for block in inference_blocks:
            self.msg_connect(
                (block, "inference"), (inference_output_block, "inference")
            )
        return inference_output_block

    def calc_rates(
        self,
        freq_start,
        freq_end,
        sweep_sec,
        samp_rate,
        nfft,
        tuneoverlap,
        tune_dwell_ms,
        tune_step_fft,
        peak_fft_range,
    ):
        stare = freq_end == 0
        tune_step_hz = int(samp_rate * tuneoverlap)
        if stare:
            freq_range = samp_rate
        else:
            freq_range = freq_end - freq_start
        fft_rate = int(samp_rate / nfft)

        if not tune_step_fft:
            if tune_dwell_ms:
                tune_step_fft = int(fft_rate * (tune_dwell_ms / 1e3))
            else:
                target_retune_hz = freq_range / sweep_sec / tune_step_hz
                tune_step_fft = int(fft_rate / target_retune_hz)
                logging.info(
                    f"retuning across {freq_range/1e6}MHz in {sweep_sec}s, requires retuning at {target_retune_hz}Hz in {tune_step_hz/1e6}MHz steps ({tune_step_fft} FFTs)"
                )
        if not tune_step_fft:
            logging.info("tune_step_fft cannot be 0 - defaulting to nfft")
            tune_step_fft = nfft
        tune_dwell_ms = tune_step_fft / fft_rate * 1e3
        logging.info(
            f"requested retuning across {freq_range/1e6}MHz every {tune_step_fft} FFTs, dwell time {tune_dwell_ms}ms"
        )
        if stare and tune_dwell_ms > 1e3:
            logging.warn(">1s dwell time in stare mode, updates will be slow!")
        peak_fft_range = min(peak_fft_range, tune_step_fft)
        return (
            tune_step_hz,
            tune_step_fft,
            peak_fft_range,
        )

    def get_freqs(self, retune_pre_fft, retune_fft):
        try:
            freq_start = retune_pre_fft.get_freq_start()
            freq_end = retune_pre_fft.get_freq_end()
            initial_freq = retune_pre_fft.get_tune_freq()
            stare = retune_pre_fft.get_stare_mode()
        except AttributeError:
            freq_start = retune_fft.get_freq_start()
            freq_end = retune_fft.get_freq_end()
            initial_freq = retune_fft.get_tune_freq()
            stare = retune_fft.get_stare_mode()
        return (freq_start, freq_end, initial_freq, stare)

    def connect_blocks(self, source, other_blocks, last_block_port=0):
        last_block = source
        for block in other_blocks:
            self.connect((last_block, last_block_port), (block, 0))
            last_block = block

    def get_db_blocks(self, nfft, samp_rate, scaling):
        if scaling == "density":
            scale = 1.0 / (samp_rate * sum(([x**2 for x in self.get_window(nfft)])))
        elif scaling == "spectrum":
            scale = 1.0 / (sum(self.get_window(nfft)) ** 2)
        else:
            raise ValueError("scaling must be 'spectrum' or 'density'")
        return [
            blocks.complex_to_mag_squared(nfft),
            blocks.multiply_const_ff(scale, nfft),
            blocks.nlog10_ff(10, nfft, 0),
        ]

    def get_window(self, nfft):
        return window.hann(nfft)

    def get_pretune_block(
        self,
        fft_batch_size,
        nfft,
        samp_rate,
        tune_jitter_hz,
        freq_start,
        freq_end,
        tune_step_hz,
        tune_step_fft,
        skip_tune_step,
        tuning_ranges,
        pretune,
        low_power_hold_down,
        slew_rx_time,
        tag_now,
    ):
        # if pretuning, the pretune block will also do the batching.
        if pretune:
            block = self.iqtlabs.retune_pre_fft(
                nfft=nfft,
                samp_rate=int(samp_rate),
                tune_jitter_hz=int(tune_jitter_hz),
                fft_batch_size=fft_batch_size,
                tag="rx_freq",
                freq_start=int(freq_start),
                freq_end=int(freq_end),
                tune_step_hz=tune_step_hz,
                tune_step_fft=tune_step_fft,
                skip_tune_step_fft=skip_tune_step,
                tuning_ranges=tuning_ranges,
                tag_now=tag_now,
                low_power_hold_down=low_power_hold_down,
                slew_rx_time=slew_rx_time,
            )
        else:
            # otherwise, the pretuning block will just do batching.
            block = blocks.stream_to_vector(gr.sizeof_gr_complex, fft_batch_size * nfft)
        return block

    def apply_window(self, nfft, fft_batch_size):
        window_constants = [val for val in self.get_window(nfft) for _ in range(2)]
        return blocks.multiply_const_vff(window_constants * fft_batch_size)

    def get_offload_fft_blocks(
        self,
        vkfft,
        fft_batch_size,
        nfft,
        fft_processor_affinity,
    ):
        fft_block = None
        fft_roll = False
        if self.wavelearner:
            fft_block = self.wavelearner.fft(int(fft_batch_size * nfft), nfft, True)
            fft_roll = True
        elif vkfft:
            # VkFFT handles batches by using set_multiple_output(), so we do not need
            # to wrap it.
            fft_block = self.iqtlabs.vkfft(fft_batch_size, nfft, True)
            fft_batch_size = 1
        else:
            logging.warning(
                "using software FFT - may not be deterministic, even with cached wisdon"
            )
            fft_block = fft.fft_vcc(nfft, True, [], True, 1)
            fft_batch_size = 1
        fft_block.set_thread_priority(99)
        fft_block.set_processor_affinity([fft_processor_affinity])

        fft_blocks = [
            self.apply_window(nfft, fft_batch_size),
            fft_block,
        ]
        if fft_batch_size > 1:
            fft_blocks.append(
                blocks.vector_to_stream(gr.sizeof_gr_complex * nfft, fft_batch_size)
            )
        if fft_roll:
            fft_blocks.append(self.iqtlabs.vector_roll(nfft))
        return fft_batch_size, fft_blocks

    def wrap_batch(self, wrap_blocks, fft_batch_size, nfft):
        # We prefer to deal with vector batches for efficiency, but some blocks
        # handle only single items. Wrap single-item blocks for batch compatibility
        # for now until batch-friendly blocks are available.
        return (
            [blocks.vector_to_stream(gr.sizeof_gr_complex, fft_batch_size * nfft)]
            + wrap_blocks
            + [blocks.stream_to_vector(gr.sizeof_gr_complex, fft_batch_size * nfft)]
        )

    def get_dc_blocks(
        self,
        correct_iq,
        dc_block_len,
        dc_block_long,
        dc_spike_detrend_length,
        dc_spike_remove_ratio,
        fft_batch_size,
        nfft,
    ):
        dc_blocks = []
        if dc_spike_detrend_length:
            logging.info(f"using dc_spike_detrend length={dc_spike_detrend_length}")
            dc_blocks.append(dc_spike_detrend(dc_spike_detrend_length))
        if dc_spike_remove_ratio:
            logging.info(f"using dc_spike_remove ratio={dc_spike_remove_ratio}")
            dc_blocks.append(dc_spike_remove(dc_spike_remove_ratio))
        if correct_iq:
            logging.info("using correct I/Q")
            dc_blocks.append(blocks.correctiq())
        if dc_block_len:
            logging.info(
                "using DC block length %u long %s", dc_block_len, dc_block_long
            )
            dc_blocks.append(grfilter.dc_blocker_cc(dc_block_len, dc_block_long))
        if dc_blocks:
            return self.wrap_batch(
                dc_blocks,
                fft_batch_size,
                nfft,
            )
        return []

    def get_fft_blocks(
        self,
        samp_rate,
        tune_jitter_hz,
        vkfft,
        fft_batch_size,
        nfft,
        freq_start,
        freq_end,
        tune_step_hz,
        tune_step_fft,
        skip_tune_step,
        tuning_ranges,
        pretune,
        fft_processor_affinity,
        low_power_hold_down,
        slew_rx_time,
        tag_now,
    ):
        fft_batch_size, fft_blocks = self.get_offload_fft_blocks(
            vkfft,
            fft_batch_size,
            nfft,
            fft_processor_affinity,
        )
        retune_pre_fft = self.get_pretune_block(
            fft_batch_size,
            nfft,
            samp_rate,
            tune_jitter_hz,
            freq_start,
            freq_end,
            tune_step_hz,
            tune_step_fft,
            skip_tune_step,
            tuning_ranges,
            pretune,
            low_power_hold_down,
            slew_rx_time,
            tag_now,
        )
        return (fft_batch_size, retune_pre_fft, [retune_pre_fft] + fft_blocks)

    def start(self):
        super().start()
        self.workaround_start_hook(self)
        logging.info("raw edge and message edge lists follow")
        logging.info(self.edge_list())
        logging.info(self.msg_edge_list())
