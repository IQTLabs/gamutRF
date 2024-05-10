import logging
import os
import sys
import time
from urllib.parse import urlparse
import numpy as np

try:
    import pmt
    from gnuradio import blocks
    from gnuradio import soapy
    from gnuradio import uhd
    from gnuradio import gr
    from gnuradio import iqtlabs
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)

from gamutrf.ettus_source import get_ettus_source
from gamutrf.soapy_source import get_soapy_source
from gamutrf.sample_reader import get_samples


def null_workaround_start_hook(self):
    return


def airt_workaround_start_hook(self):
    logging.info("applying AIRT workarounds")
    workaround_time = 0.5
    for rate in (15.625e6, self.samp_rate):
        time.sleep(workaround_time)
        self.sources[0].set_sample_rate(0, rate)
        self.sources[0].set_bandwidth(0, rate)


class file_source_tagger(gr.basic_block):
    def __init__(
        self,
        input_file,
        cmd_port,
        nfft,
        tune_step_fft,
    ):
        gr.basic_block.__init__(
            self,
            name="file_source_tagger",
            in_sig=None,
            out_sig=[np.complex64],
        )
        _, self.samples, meta = get_samples(input_file)
        self.timestamp = meta["timestamp"]
        self.center_freq = meta["center_frequency"]
        self.sample_rate = meta["sample_rate"]
        self.n_samples = len(self.samples)
        self.nfft = nfft
        self.tune_step_fft = tune_step_fft
        self.cmds_received = 0
        self.tags_sent = -1
        self.i = 0
        self.message_port_register_in(pmt.intern(cmd_port))
        self.set_msg_handler(pmt.intern(cmd_port), self.handle_cmd)
        self.set_output_multiple(self.nfft)
        self.tagged_interval = self.nfft * self.tune_step_fft
        self.n_samples = (
            int(self.n_samples / self.tagged_interval) * self.tagged_interval
        )
        logging.info("opened %s with %u samples", input_file, self.n_samples)
        self.work_guard = None

    def complete(self):
        return self.i >= self.n_samples

    def handle_cmd(self, _msg):
        self.cmds_received += 1

    def make_rx_time(self, sample_time):
        sample_sec = int(sample_time)
        sample_fsec = sample_time - sample_sec
        pmt_sample_time = pmt.make_tuple(
            pmt.from_long(sample_sec), pmt.from_double(sample_fsec)
        )
        return pmt_sample_time

    def add_tags(self):
        # Ideally, We want to add simulated rx_time tags in a consistent way between
        # runs. However, gnuradio's scheduler does not gaurantee when messages between
        # blocks are delivered. So more or less items may be processed before
        # a simulated tuning request is received by this block, between runs. This
        # will result in different timestamps.
        self.tags_sent += 1
        tag_pos = self.tags_sent * self.tagged_interval
        sample_time = self.timestamp + (tag_pos / float(self.sample_rate))
        logging.info(
            "tag %u at pos %u (nfft item %u), %.1f%%",
            self.tags_sent,
            tag_pos,
            int(tag_pos / self.nfft),
            self.i / self.n_samples * 100,
        )
        self.add_item_tag(
            0,
            tag_pos,
            pmt.intern("rx_freq"),
            pmt.from_double(self.center_freq),
        )
        self.add_item_tag(
            0,
            tag_pos,
            pmt.intern("rx_time"),
            self.make_rx_time(sample_time),
        )

    def general_work(self, input_items, output_items):
        if self.complete():
            # gnuradio will drop all undelivered messages to blocks when our source
            # returns done. cause gnuradio to repeatedly call us when we're done, to
            # give other blocks an opportunity to process undelivered messages.
            if self.work_guard is None:
                self.work_guard = time.time()
                logging.info("file ended, waiting for other blocks to finish")
                return 0
            if time.time() - self.work_guard < 3:
                return 0
            logging.info("complete")
            return -1

        n = min(self.nfft, len(output_items[0]))
        samples = self.samples[self.i : self.i + n]
        c = len(samples)
        self.i += c
        if c < len(output_items[0]):
            zeros = np.zeros(len(output_items[0]) - c, dtype=np.complex64)
            samples = np.append(samples, zeros)
        while (
            not self.complete() and int(self.i / self.tagged_interval) != self.tags_sent
        ):
            self.add_tags()
        output_items[0][:] = samples
        return c


def get_throttle(samp_rate, items):
    return blocks.throttle(gr.sizeof_gr_complex, samp_rate, True, items)


def get_source(
    sdr,
    samp_rate,
    gain,
    nfft,
    tune_step_fft,
    agc=False,
    center_freq=None,
    sdrargs=None,
    soapy_lib=soapy,
    uhd_lib=uhd,
    dc_ettus_auto_offset=True,
):
    logging.info(
        f"initializing SDR {sdr} with sample rate {samp_rate}, gain {gain}, agc {agc}"
    )

    workaround_start_hook = null_workaround_start_hook
    cmd_port = "command"
    sources = []

    url = urlparse(sdr)
    if url.scheme:
        if url.scheme == "file" and os.path.exists(url.path):
            sources = [
                file_source_tagger(url.path, cmd_port, nfft, tune_step_fft),
            ]
        else:
            raise ValueError("unsupported/missing file location")
    elif sdr == "tuneable_test_source":
        freq_divisor = 1e9
        cmd_port = "cmd"
        sources = [
            iqtlabs.tuneable_test_source(0, freq_divisor),
            get_throttle(samp_rate, nfft),
        ]
    elif sdr == "ettus":
        sources = get_ettus_source(
            sdrargs, samp_rate, center_freq, agc, gain, uhd_lib, dc_ettus_auto_offset
        )
    else:
        sources, cmd_port = get_soapy_source(
            sdr, sdrargs, samp_rate, center_freq, agc, gain, soapy_lib
        )
        if sdr == "SoapyAIRT":
            workaround_start_hook = airt_workaround_start_hook
    sources[0].set_thread_priority(99)
    return sources, cmd_port, workaround_start_hook
