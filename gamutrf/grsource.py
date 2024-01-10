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
        self.i = 0
        self.message_port_register_in(pmt.intern(cmd_port))
        self.set_msg_handler(pmt.intern(cmd_port), self.handle_cmd)
        self.need_tags = False
        self.tags_sent = 0
        logging.info("opened %s with %u samples", input_file, self.n_samples)

    def complete(self):
        return self.i >= self.n_samples

    def handle_cmd(self, _msg):
        self.need_tags = True

    def add_tags(self):
        self.tags_sent += 1
        self.add_item_tag(
            0,
            self.nitems_written(0),
            pmt.intern("rx_freq"),
            pmt.from_double(self.center_freq),
        )
        sample_time = self.timestamp + (
            (self.tags_sent * self.nfft * self.tune_step_fft) / float(self.sample_rate)
        )
        logging.info("%.1f%%", self.i / self.n_samples * 100)
        sample_sec = int(sample_time)
        sample_fsec = sample_time - sample_sec
        pmt_sample_time = pmt.make_tuple(
            pmt.from_long(sample_sec), pmt.from_double(sample_fsec)
        )
        self.add_item_tag(
            0,
            self.nitems_written(0),
            pmt.intern("rx_time"),
            pmt_sample_time,
        )

    def general_work(self, input_items, output_items):
        if self.complete():
            logging.info("100%%")
            return -1
        if self.need_tags:
            self.add_tags()
            self.need_tags = False
        if self.tags_sent:
            n = min(self.nfft, len(output_items[0]))
            samples = self.samples[self.i : self.i + n]
            c = len(samples)
            self.i += c
            if c < len(output_items[0]):
                zeros = np.zeros(len(output_items[0]) - c, dtype=np.complex64)
                samples = np.append(samples, zeros)
        else:
            # feed zeros until received first tag request
            samples = np.zeros(len(output_items[0]), dtype=np.complex64)
            c = len(samples)
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
            iqtlabs.tuneable_test_source(freq_divisor),
            get_throttle(samp_rate, nfft),
        ]
    elif sdr == "ettus":
        sources = get_ettus_source(sdrargs, samp_rate, center_freq, agc, gain, uhd_lib)
    else:
        sources, cmd_port = get_soapy_source(
            sdr, sdrargs, samp_rate, center_freq, agc, gain, soapy_lib
        )
        if sdr == "SoapyAIRT":
            workaround_start_hook = airt_workaround_start_hook
    sources[0].set_thread_priority(99)
    return sources, cmd_port, workaround_start_hook
