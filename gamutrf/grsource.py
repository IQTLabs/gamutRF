import logging
import os
import sys
import time
from urllib.parse import urlparse

try:
    import pmt
    from gnuradio import blocks
    from gnuradio import soapy
    from gnuradio import uhd
    from gnuradio.gr import sizeof_gr_complex
    from gnuradio import iqtlabs
except ModuleNotFoundError:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme)"
    )
    sys.exit(1)

from gamutrf.utils import ETTUS_ANT
from gamutrf.utils import ETTUS_ARGS


def null_workaround_start_hook(self):
    return


def airt_workaround_start_hook(self):
    logging.info("applying AIRT workarounds")
    workaround_time = 0.5
    for rate in (20e6, self.samp_rate):
        time.sleep(workaround_time)
        self.sources[0].set_sample_rate(0, rate)
        self.sources[0].set_bandwidth(0, rate)


def get_source(
    sdr,
    samp_rate,
    gain,
    agc=False,
    center_freq=None,
    sdrargs=None,
    soapy=soapy,
    uhd=uhd,
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
                blocks.file_source(sizeof_gr_complex, url.path, True, 0, 0),
                blocks.throttle(sizeof_gr_complex, samp_rate, True),
            ]
            sources[0].message_port_register_in(pmt.intern(cmd_port))
        else:
            raise ValueError("unsupported/missing file location")
    elif sdr == "tuneable_test_source":
        freq_divisor = 1e9
        cmd_port = "cmd"
        sources = [
            iqtlabs.tuneable_test_source(freq_divisor),
            blocks.throttle(sizeof_gr_complex, samp_rate, True),
        ]
    elif sdr == "ettus":
        if not sdrargs:
            sdrargs = ETTUS_ARGS
        source = uhd.usrp_source(
            ",".join((sdrargs, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=list(range(0, 1)),
            ),
        )
        source.set_antenna(ETTUS_ANT, 0)
        source.set_samp_rate(samp_rate)
        if center_freq is not None:
            source.set_center_freq(center_freq, 0)
        source.set_gain(gain, 0)
        source.set_rx_agc(agc, 0)

        now = time.time()
        now_sec = int(now)
        now_frac = now - now_sec
        uhd_now = uhd.time_spec(now_sec, now_frac)
        source.set_time_now(uhd_now, uhd.ALL_MBOARDS)
        time_diff = source.get_time_now().get_real_secs() - now
        if time_diff > 5:
            raise ValueError("could not set Ettus SDR time successfully")
        sources = [source]
    else:
        dev = f"driver={sdr}"
        stream_args = ""
        tune_args = [""]
        settings = [""]
        if sdrargs:
            settings = sdrargs
        source = soapy.source(dev, "fc32", 1, "", stream_args, tune_args, settings)
        source.set_sample_rate(0, samp_rate)
        source.set_bandwidth(0, samp_rate)
        if center_freq is not None:
            source.set_frequency(0, center_freq)
            source.set_frequency_correction(0, 0)
            source.set_gain_mode(0, agc)
            source.set_gain(0, gain)
            cmd_port = "cmd"
        if sdr == "SoapyAIRT":
            workaround_start_hook = airt_workaround_start_hook
        sources = [source]

    return sources, cmd_port, workaround_start_hook
