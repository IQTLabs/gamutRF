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
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)

from gamutrf.ettus_source import get_ettus_source
from gamutrf.soapy_source import get_soapy_source


def null_workaround_start_hook(self):
    return


def airt_workaround_start_hook(self):
    logging.info("applying AIRT workarounds")
    workaround_time = 0.5
    for rate in (15.625e6, self.samp_rate):
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
        sources = get_ettus_source(sdrargs, samp_rate, center_freq, agc, gain, uhd_lib)
    else:
        sources, cmd_port = get_soapy_source(
            sdr, sdrargs, samp_rate, center_freq, agc, gain, soapy_lib
        )
        if sdr == "SoapyAIRT":
            workaround_start_hook = airt_workaround_start_hook
    sources[0].set_thread_priority(99)
    return sources, cmd_port, workaround_start_hook
